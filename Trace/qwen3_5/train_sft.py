"""MSLoc-Qwen3.5 SFT：保留 DAM、EAM 与 LAA/CLoss。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ZeRO-2 使用 PyTorch AdamW，不需要编译 DeepSpeed 自定义 CUDA 算子。
# DeepSpeed 0.18.9 官方支持关闭本地 CUDA Toolkit 探测。
os.environ["DS_IGNORE_CUDA_DETECTION"] = "1"

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoProcessor, Trainer, TrainingArguments, set_seed

from Trace.qwen3_5.data import ProposalDataset, ProposalSample, build_messages, load_proposal_samples, sample_proposal_frames
from Trace.qwen3_5.modeling_msloc_qwen3_5 import IGNORE_INDEX, MSLocQwen35ForConditionalGeneration


class MSLocSFTTrainer(Trainer):
    """在通用 Trainer 日志中保留可审计的文本损失与 CLoss。"""

    @staticmethod
    def _base_model(model: Any) -> Any:
        current = model
        visited: set[int] = set()
        while id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "last_text_loss"):
                return current
            if hasattr(current, "module"):
                current = current.module
                continue
            break
        return current

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        base = self._base_model(self.model)
        for source_name, log_name in (
            ("last_text_loss", "text_loss"),
            ("last_closs", "closs"),
            ("last_total_loss", "model_total_loss"),
            ("last_supervised_tokens", "supervised_tokens"),
        ):
            value = getattr(base, source_name, None)
            if value is not None:
                logs[log_name] = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        super().log(logs, *args, **kwargs)


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered not in {"true", "false"}:
        raise argparse.ArgumentTypeError("必须显式传入 true 或 false")
    return lowered == "true"


def safe_clean_directory(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (WORKSPACE_ROOT.parent / "MSLoc_data" / "Trace").resolve()
    if not resolved.is_relative_to(allowed_root) or resolved == allowed_root:
        raise ValueError(f"只允许清理 {allowed_root} 下的具体实验目录：{resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_class_mapping(path: str) -> tuple[list[str], dict[str, int], torch.Tensor]:
    feature_path = Path(path)
    if not feature_path.is_file():
        raise FileNotFoundError(f"LAA 类别特征不存在：{feature_path}")
    payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    required = {"class_names", "class_features", "feat_dim"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise TypeError(f"class_features_bge.pt 必须包含字段：{sorted(required)}")
    if not isinstance(payload["class_names"], list):
        raise TypeError("class_features_bge.pt 的 class_names 必须是列表")
    class_names = [str(name) for name in payload["class_names"]]
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError("class_names 必须非空且不能重复")
    class_features = payload["class_features"]
    if not isinstance(class_features, torch.Tensor) or class_features.ndim != 2:
        raise TypeError("class_features 必须是二维 Tensor")
    if class_features.shape[0] != len(class_names):
        raise ValueError("class_features 行数必须与 class_names 数量一致")
    if int(payload["feat_dim"]) != class_features.shape[1]:
        raise ValueError("feat_dim 与 class_features 的特征维度不一致")
    if not torch.isfinite(class_features).all():
        raise ValueError("class_features 包含 NaN 或 Inf")
    return class_names, {name: index for index, name in enumerate(class_names)}, class_features.float()


@dataclass
class MSLocSFTCollator:
    processor: Any
    class_to_idx: dict[str, int]
    bnd_frames: int
    seg_frames: int
    bnd_ratio: float
    max_length: int
    max_pixels: int

    def _tokenize(self, sample: ProposalSample, include_answer: bool) -> dict[str, torch.Tensor]:
        return self.processor.apply_chat_template(
            build_messages(sample, include_answer=include_answer), tokenize=True,
            add_generation_prompt=not include_answer, return_dict=True, return_tensors="pt",
            enable_thinking=False, truncation=False,
        )

    def __call__(self, samples: list[ProposalSample]) -> dict[str, torch.Tensor]:
        text_rows: list[dict[str, torch.Tensor]] = []
        visual_rows: list[dict[str, torch.Tensor]] = []
        class_rows: list[list[int]] = []
        expected_frames = 2 * self.bnd_frames + self.seg_frames
        for sample in samples:
            prompt = self._tokenize(sample, include_answer=False)
            full = self._tokenize(sample, include_answer=True)
            prompt_ids, full_ids = prompt["input_ids"][0], full["input_ids"][0]
            if len(prompt_ids) >= len(full_ids) or not torch.equal(full_ids[: len(prompt_ids)], prompt_ids):
                raise ValueError(f"chat template 前缀不一致：{sample.sample_id}")
            if len(full_ids) > self.max_length:
                raise ValueError(
                    f"SFT 样本超过 max_length 且不允许截断 assistant 答案："
                    f"{sample.sample_id} -> {len(full_ids)} > {self.max_length}"
                )
            labels = full_ids.clone()
            labels[: len(prompt_ids)] = IGNORE_INDEX
            text_rows.append({"input_ids": full_ids, "attention_mask": full["attention_mask"][0], "labels": labels})

            frames, _ = sample_proposal_frames(
                sample.video_path, sample.proposal, self.bnd_frames, self.seg_frames, self.bnd_ratio
            )
            if len(frames) != expected_frames:
                raise ValueError(f"采样帧数错误：{sample.sample_id} -> {len(frames)}")
            visual = self.processor.image_processor(images=frames, return_tensors="pt", max_pixels=self.max_pixels)
            if "pixel_values" not in visual or "image_grid_thw" not in visual:
                raise KeyError("Qwen3.5 image_processor 缺少 pixel_values/image_grid_thw")
            if visual["image_grid_thw"].shape[0] != expected_frames:
                raise ValueError("必须把 40 帧作为 40 张独立图像编码，不能提前合并时间维")
            visual_rows.append(visual)

            indices = []
            for class_name in sample.closs_classes:
                if class_name not in self.class_to_idx:
                    raise KeyError(f"LAA 类别不在 class_features_bge.pt 中：{class_name}")
                indices.append(self.class_to_idx[class_name])
            class_rows.append(indices)

        max_tokens = max(row["input_ids"].shape[0] for row in text_rows)
        tokenizer = self.processor.tokenizer
        batch: dict[str, torch.Tensor] = {}
        for key, padding_value in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", IGNORE_INDEX)):
            padded = []
            for row in text_rows:
                tensor = row[key]
                padding = torch.full((max_tokens - tensor.shape[0],), padding_value, dtype=tensor.dtype)
                padded.append(torch.cat((tensor, padding)))
            batch[key] = torch.stack(padded)
        batch["pixel_values"] = torch.cat([row["pixel_values"] for row in visual_rows], dim=0)
        batch["image_grid_thw"] = torch.cat([row["image_grid_thw"] for row in visual_rows], dim=0)
        batch["frame_counts"] = torch.full((len(samples),), expected_frames, dtype=torch.long)
        batch["closs_labels"] = torch.tensor(class_rows, dtype=torch.long)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MSLoc-Qwen3.5 proposal-level SFT")
    parser.add_argument("--model", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--class-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deepspeed", required=True, help="DeepSpeed ZeRO 配置路径")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=112896)
    parser.add_argument("--bnd-frames", type=int, default=16)
    parser.add_argument("--seg-frames", type=int, default=8)
    parser.add_argument("--bnd-ratio", type=float, default=0.2)
    parser.add_argument("--closs-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--freeze-backbone",
        type=str_to_bool,
        default=False,
        help="仅供单卡冒烟测试：冻结 Qwen 官方骨干，只训练 MSLoc 新增模块；正式训练保持 false",
    )
    parser.add_argument("--freeze-vision", type=str_to_bool, required=True)
    parser.add_argument("--resume", default="none", help="none、auto 或 checkpoint 路径")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    if args.clean and args.resume != "none":
        raise ValueError("--clean 与 --resume 不能同时使用")
    if args.closs_weight < 0:
        raise ValueError("closs-weight 不能为负数")
    deepspeed_path = Path(args.deepspeed)
    if not deepspeed_path.is_file():
        raise FileNotFoundError(f"DeepSpeed 配置不存在：{deepspeed_path}")
    deepspeed_config = str(deepspeed_path)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    output = Path(args.output)
    if args.clean and rank == 0:
        safe_clean_directory(output)
    if world_size > 1:
        dist.barrier()

    manifest_path = output / "msloc_training_manifest.json"
    run_manifest = {
        "model": str(Path(args.model).resolve()),
        "annotation": str(Path(args.annotation).resolve()),
        "proposals": str(Path(args.proposals).resolve()),
        "video_root": str(Path(args.video_root).resolve()),
        "class_features": str(Path(args.class_features).resolve()),
        "deepspeed": str(Path(deepspeed_config).resolve()),
        "world_size": world_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "max_pixels": args.max_pixels,
        "bnd_frames": args.bnd_frames,
        "seg_frames": args.seg_frames,
        "bnd_ratio": args.bnd_ratio,
        "closs_weight": args.closs_weight,
        "max_samples": args.max_samples,
        "freeze_backbone": args.freeze_backbone,
        "freeze_vision": args.freeze_vision,
        "seed": 42,
    }
    if args.resume != "none":
        if not manifest_path.is_file():
            raise FileNotFoundError(f"训练恢复元数据不存在：{manifest_path}")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest != run_manifest:
            raise ValueError("恢复参数与原训练任务不一致")
    elif rank == 0:
        if manifest_path.exists():
            raise FileExistsError(f"输出目录已有训练元数据，请使用 --clean 或 --resume：{manifest_path}")
        atomic_write_json(manifest_path, run_manifest)
    if world_size > 1:
        dist.barrier()

    class_names, class_to_idx, class_features = load_class_mapping(args.class_features)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    is_msloc_checkpoint = bool(getattr(config, "msloc_class_names", None))
    config.msloc_class_names = class_names
    config.msloc_bnd_frames = args.bnd_frames
    config.msloc_seg_frames = args.seg_frames
    config.msloc_closs_weight = args.closs_weight
    config.msloc_class_feature_dim = int(class_features.shape[1])
    model = MSLocQwen35ForConditionalGeneration.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16, local_files_only=True
    )
    # 基础 Qwen checkpoint 不含 DAM/EAM、异常 token 和 CLoss。必须在
    # from_pretrained 完成缺失权重处理后最终初始化；Trainer 恢复时随后会用
    # checkpoint 覆盖这些初始值。已有 MSLoc checkpoint 则保留其训练权重。
    if not is_msloc_checkpoint:
        model.reset_msloc_parameters()
    model.validate_msloc_parameters()
    if rank == 0:
        print(
            "MSLoc新增模块："
            + ("从已有checkpoint加载" if is_msloc_checkpoint else "基础Qwen加载后重新初始化"),
            flush=True,
        )
    model.class_feature_bank.copy_(class_features)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        raise ValueError("Qwen3.5 tokenizer 必须提供 pad_token_id")
    if args.freeze_backbone:
        # Qwen 的 lm_head 位于 model.model 之外；必须先冻结整个官方模型，
        # 再显式解冻 MSLoc 新增模块，避免单卡冒烟测试仍为 lm_head 分配 Adam 状态。
        model.requires_grad_(False)
    if args.freeze_vision:
        model.model.visual.requires_grad_(False)
    model.msloc_projector.requires_grad_(True)
    model.anomaly_tokens.requires_grad_(True)
    model.closs_head.requires_grad_(True)
    if rank == 0:
        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(
            f"参数统计：总参数 {total_params:,}，可训练参数 {trainable_params:,} "
            f"({100.0 * trainable_params / total_params:.4f}%)",
            flush=True,
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    samples = load_proposal_samples(args.annotation, args.proposals, args.video_root, max_samples=args.max_samples)
    if not samples:
        raise ValueError("SFT 没有构造出任何 proposal 样本")
    dataset = ProposalDataset(samples)
    collator = MSLocSFTCollator(
        processor, class_to_idx, args.bnd_frames, args.seg_frames,
        args.bnd_ratio, args.max_length, args.max_pixels,
    )
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        # Transformers 5.x 将 warmup_ratio 合并到 warmup_steps；小于 1 的浮点数仍表示总步数比例。
        warmup_steps=args.warmup_ratio, lr_scheduler_type="cosine", logging_steps=1,
        save_strategy="epoch", save_total_limit=99, bf16=True, tf32=False,
        gradient_checkpointing=True, dataloader_num_workers=args.num_workers,
        remove_unused_columns=False, report_to="none", seed=42, data_seed=42,
        deepspeed=deepspeed_config, disable_tqdm=False,
    )
    trainer = MSLocSFTTrainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=collator, processing_class=processor,
    )
    resume: str | bool | None = None if args.resume == "none" else (True if args.resume == "auto" else args.resume)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))


if __name__ == "__main__":
    main()
