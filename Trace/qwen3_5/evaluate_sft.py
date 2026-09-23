"""Qwen3.5 SFT 推理：refine 第一阶段 proposal 并输出统一评测格式。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoProcessor, set_seed

from Trace.qwen3_5.data import (
    ProposalSample,
    build_messages,
    load_proposal_samples,
    parse_answer,
    sample_proposal_frames,
)
from Trace.qwen3_5.modeling_msloc_qwen3_5 import MSLocQwen35ForConditionalGeneration


def safe_clean_directory(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (WORKSPACE_ROOT.parent / "MSLoc_data" / "Trace").resolve()
    if not resolved.is_relative_to(allowed_root) or resolved == allowed_root:
        raise ValueError(f"只允许清理 {allowed_root} 下的具体评测目录：{resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5 proposal-level evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, default=40)
    parser.add_argument("--bnd-frames", type=int, default=16)
    parser.add_argument("--seg-frames", type=int, default=8)
    parser.add_argument("--bnd-ratio", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=112896)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


@torch.inference_mode()
def infer_one(
    sample: ProposalSample,
    model: MSLocQwen35ForConditionalGeneration,
    processor,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    frames, sampled_timestamps = sample_proposal_frames(
        sample.video_path,
        sample.proposal,
        args.bnd_frames,
        args.seg_frames,
        args.bnd_ratio,
    )
    text_inputs = processor.apply_chat_template(
        build_messages(sample, include_answer=False),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    visual_inputs = processor.image_processor(
        images=frames,
        return_tensors="pt",
        max_pixels=args.max_pixels,
    )
    if visual_inputs["image_grid_thw"].shape[0] != args.num_frames:
        raise ValueError("Qwen3.5 测试必须把每一帧作为独立图像编码")
    eos_set: set[int] = set()
    for eos_ids in (model.generation_config.eos_token_id, processor.tokenizer.eos_token_id):
        if eos_ids is None:
            continue
        if isinstance(eos_ids, (list, tuple, set)):
            eos_set.update(int(value) for value in eos_ids)
        else:
            eos_set.add(int(eos_ids))
    if not eos_set:
        raise ValueError("Qwen3.5 model/tokenizer 都没有提供 eos_token_id")
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Qwen3.5 tokenizer 没有提供 pad_token_id")
    generated = model.generate_msloc(
        input_ids=text_inputs["input_ids"].to(device),
        attention_mask=text_inputs["attention_mask"].to(device),
        pixel_values=visual_inputs["pixel_values"].to(device),
        image_grid_thw=visual_inputs["image_grid_thw"].to(device),
        frame_counts=torch.tensor([args.num_frames], dtype=torch.long, device=device),
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_set,
        pad_token_id=int(pad_token_id),
    )
    response = processor.decode(
        generated[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    win_start, win_end = sample.proposal
    record = {
        "sample_id": sample.sample_id,
        "video_path": sample.video_path,
        "proposal": [win_start, win_end],
        "sampled_timestamps": sampled_timestamps,
        "raw_response": response,
    }
    try:
        parsed = parse_answer(response, win_end - win_start)
        record["parse_status"] = "valid_event" if parsed["label"] == "fake" else "valid_no_event"
        record["label"] = parsed["label"]
        record["segments"] = [
            [win_start + segment[0], win_start + segment[1]] for segment in parsed["segments"]
        ]
        record["explanations"] = parsed["explanations"]
        record["manipulation_types"] = parsed["manipulation_types"]
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        record["parse_status"] = "format_failure"
        record["parse_error"] = str(error)
        record["label"] = "invalid"
        record["segments"] = []
        record["explanations"] = []
        record["manipulation_types"] = []
    return record


def aggregate_predictions(proposal_items: list[dict], records: list[dict]) -> list[dict]:
    records_by_video: dict[str, list[dict]] = {}
    for record in records:
        relative_video = record["sample_id"].split("::proposal-", 1)[0]
        records_by_video.setdefault(relative_video, []).append(record)

    outputs: list[dict] = []
    for item in proposal_items:
        video = str(item.get("video_path") or item.get("video") or item.get("image_id"))
        video_records = records_by_video.get(video, [])
        if not video_records:
            source_segments = item.get("model_inference", {}).get("segment")
            if source_segments == []:
                outputs.append(
                    {
                        "video_path": video,
                        "model_inference": {
                            "segment": [],
                            "response": [],
                            "event_types": [],
                            "type": "real",
                            "decision_status": "no_stage1_proposal",
                            "proposal_outputs": [],
                        },
                    }
                )
                continue
            raise KeyError(f"存在第一阶段 proposal，但缺少 Qwen 推理结果：{video}")
        fake_records = [record for record in video_records if record["label"] == "fake"]
        valid_records = [
            record for record in video_records
            if record["parse_status"] in {"valid_event", "valid_no_event"}
        ]
        segments = [segment for record in fake_records for segment in record["segments"]]
        responses = [
            explanation
            for record in fake_records
            for explanation in record["explanations"]
        ]
        event_types = [
            manipulation_type
            for record in fake_records
            for manipulation_type in record["manipulation_types"]
        ]
        if len(responses) != len(segments) or len(event_types) != len(segments):
            raise ValueError("Qwen 事件时间段、类型与解释数量不一致")
        if segments:
            prediction_type, decision_status = "fake", "semantic_event"
        elif len(valid_records) == len(video_records):
            prediction_type, decision_status = "real", "semantic_no_event"
        else:
            prediction_type, decision_status = "invalid", "format_failure"
        outputs.append(
            {
                "video_path": video,
                "model_inference": {
                    "segment": segments,
                    "response": responses,
                    "event_types": event_types,
                    "type": prediction_type,
                    "decision_status": decision_status,
                    "proposal_outputs": video_records,
                },
            }
        )
    return outputs


def main() -> None:
    args = parse_args()
    if args.num_frames != args.bnd_frames * 2 + args.seg_frames:
        raise ValueError("num_frames 必须等于 2*bnd_frames + seg_frames")
    if args.max_pixels <= 0:
        raise ValueError("max_pixels 必须为正数")
    if args.clean and args.resume:
        raise ValueError("--clean 与 --resume 不能同时使用")
    set_seed(42)
    rank, world_size, local_rank = init_distributed()
    output = Path(args.output)
    if args.clean and rank == 0:
        safe_clean_directory(output)
    if world_size > 1:
        dist.barrier()
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = output / "run_manifest.json"
    run_manifest = {
        "model": str(Path(args.model).resolve()),
        "annotation": str(Path(args.annotation).resolve()),
        "proposals": str(Path(args.proposals).resolve()),
        "video_root": str(Path(args.video_root).resolve()),
        "world_size": world_size,
        "num_frames": args.num_frames,
        "bnd_frames": args.bnd_frames,
        "seg_frames": args.seg_frames,
        "bnd_ratio": args.bnd_ratio,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "max_samples": args.max_samples,
        "seed": 42,
    }
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"恢复元数据不存在：{manifest_path}")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest != run_manifest:
            raise ValueError("恢复参数与原测试任务不一致，请保持模型、数据、GPU 数量和推理参数不变")
    elif rank == 0:
        if manifest_path.exists():
            raise FileExistsError(f"输出目录已有任务元数据，请使用 --clean 或 --resume：{manifest_path}")
        atomic_write_json(manifest_path, run_manifest)
    if world_size > 1:
        dist.barrier()

    samples = load_proposal_samples(
        args.annotation,
        args.proposals,
        args.video_root,
        max_samples=args.max_samples,
    )
    local_samples = samples[rank::world_size]
    progress_file = output / f"predictions_rank{rank}.jsonl"
    existing: dict[str, dict] = {}
    if args.resume:
        if not progress_file.is_file():
            raise FileNotFoundError(f"恢复文件不存在：{progress_file}")
        for line in progress_file.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            existing[record["sample_id"]] = record
    elif progress_file.exists():
        raise FileExistsError(f"进度文件已存在，请使用 --clean 或 --resume：{progress_file}")

    device = torch.device(f"cuda:{local_rank}")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = MSLocQwen35ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    model.validate_msloc_parameters()
    model_bnd_frames = int(getattr(model.config, "msloc_bnd_frames", -1))
    model_seg_frames = int(getattr(model.config, "msloc_seg_frames", -1))
    if (args.bnd_frames, args.seg_frames) != (model_bnd_frames, model_seg_frames):
        raise ValueError(
            "评测采样帧配置与 checkpoint 不一致："
            f"args=({args.bnd_frames},{args.seg_frames}), "
            f"checkpoint=({model_bnd_frames},{model_seg_frames})"
        )

    with progress_file.open("a", encoding="utf-8") as handle:
        iterator = tqdm(local_samples, disable=rank != 0, desc="Qwen3.5 测试")
        for sample in iterator:
            if sample.sample_id in existing:
                continue
            record = infer_one(sample, model, processor, device, args)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        all_records: list[dict] = []
        for worker_rank in range(world_size):
            worker_file = output / f"predictions_rank{worker_rank}.jsonl"
            if not worker_file.is_file():
                raise FileNotFoundError(f"缺少 rank 推理结果：{worker_file}")
            all_records.extend(
                json.loads(line)
                for line in worker_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if len(all_records) != len(samples):
            raise ValueError(f"推理结果数量不完整：{len(all_records)} != {len(samples)}")
        expected_ids = {sample.sample_id for sample in samples}
        actual_ids = {record["sample_id"] for record in all_records}
        if len(actual_ids) != len(all_records) or actual_ids != expected_ids:
            raise ValueError("推理结果存在重复或 sample_id 与当前任务不一致")
        proposal_items = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
        if args.max_samples:
            selected_videos = {sample.sample_id.split("::proposal-", 1)[0] for sample in samples}
            selected_counts: dict[str, int] = {}
            for sample in samples:
                video = sample.sample_id.split("::proposal-", 1)[0]
                selected_counts[video] = selected_counts.get(video, 0) + 1
            for item in proposal_items:
                video = str(item.get("video_path") or item.get("video") or item.get("image_id"))
                if video not in selected_videos:
                    continue
                proposal_count = len(item.get("model_inference", {}).get("segment", []))
                if selected_counts[video] != proposal_count:
                    raise ValueError(
                        "--max-samples 截断了某个视频的 proposal，不允许计算部分视频指标："
                        f"{video} -> {selected_counts[video]}/{proposal_count}"
                    )
            proposal_items = [
                item for item in proposal_items
                if str(item.get("video_path") or item.get("video") or item.get("image_id")) in selected_videos
            ]
        prediction_path = output / "predictions.json"
        metrics_path = output / "metrics.json"
        atomic_write_json(prediction_path, aggregate_predictions(proposal_items, all_records))
        print(f"[ok] 推理结果：{prediction_path}")

        # 与原 TRACE 测试流程完全复用同一个指标实现，Qwen 只负责生成兼容格式的预测。
        metrics_command = [
            sys.executable,
            str(WORKSPACE_ROOT / "evaluate_long.py"),
            "--gt_file",
            args.annotation,
            "--infer_file",
            str(prediction_path),
            "--output_file",
            str(metrics_path),
        ]
        if args.resume:
            metrics_command.append("--reuse-existing")
        subprocess.run(metrics_command, cwd=WORKSPACE_ROOT, check=True)
        print(f"[ok] 评测指标：{metrics_path}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
