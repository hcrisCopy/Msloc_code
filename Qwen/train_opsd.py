"""基于 ms-swift 4.5.3 GKD/OPSD 继续训练 SFT LoRA；教师为当前权重加文字真值。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def checkpoint(output: Path) -> Path:
    candidates = [path for path in output.glob("checkpoint-*") if path.is_dir() and path.name[11:].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"没有可继续的 OPSD checkpoint：{output}")
    return max(candidates, key=lambda path: int(path.name[11:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True, help="已评测的 SFT LoRA last/checkpoint-* 目录")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-completion-length", type=int, required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--resume", choices=["none", "auto"], required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    devices = [item.strip() for item in args.devices.split(",")]
    if not devices or any(not item.isdigit() for item in devices) or len(set(devices)) != len(devices):
        raise ValueError("--devices 应为不重复的 GPU 编号")
    if args.global_batch_size <= 0 or args.global_batch_size % len(devices):
        raise ValueError("global-batch-size 必须是正数且能被 GPU 数整除")
    if min(args.epochs, args.max_length, args.max_completion_length, args.save_steps) <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs、长度、save-steps 和 learning-rate 必须大于 0")
    if not Path(args.model).is_dir() or not Path(args.adapter).is_dir() or not Path(args.dataset).is_file():
        raise FileNotFoundError("模型、SFT adapter 或 OPSD 数据不存在")
    gate_path = Path(args.teacher_gate)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["passed"] is not True:
        raise ValueError("教师完整生成评测未优于学生，不能启动 OPSD")
    student_config = json.loads((Path(gate["student_eval"]) / "eval_config.json").read_text(encoding="utf-8"))
    if Path(student_config["adapter"]).resolve() != Path(args.adapter).resolve():
        raise ValueError("教师准入所用 SFT adapter 与 OPSD 初始 adapter 不同")
    if Path(student_config["model"]).resolve() != Path(args.model).resolve():
        raise ValueError("教师准入所用模型与 OPSD 模型不同")
    teacher_config = json.loads((Path(gate["teacher_eval"]) / "eval_config.json").read_text(encoding="utf-8"))
    data_config = json.loads((Path(args.dataset).parent / "data_config.json").read_text(encoding="utf-8"))
    if data_config["student_prompt_text"] != student_config["prompt_text"].strip():
        raise ValueError("OPSD 学生提示词与教师准入评测不一致")
    if data_config["teacher_prompt_text"] != teacher_config["teacher_prompt_text"].strip():
        raise ValueError("OPSD 教师提示词与教师准入评测不一致")
    for key in ("proposals", "annotation", "frames"):
        if data_config[key] != student_config[key]:
            raise ValueError(f"OPSD 数据与教师准入评测的 {key} 不一致")
    output = Path(args.output)
    if not output.resolve().is_relative_to(Path("../MSLoc_data/Qwen").resolve()):
        raise ValueError("OPSD 输出必须位于 ../MSLoc_data/Qwen/ 下")
    if args.clean and args.resume != "none":
        raise ValueError("--clean 与 --resume auto 不能同时使用")
    if args.clean and output.exists():
        shutil.rmtree(output)
    if args.resume == "none" and output.exists():
        raise FileExistsError(f"输出目录已存在：{output}；使用 --resume auto 或 --clean")
    resume_path = checkpoint(output) if args.resume == "auto" else None
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(devices),
        "NPROC_PER_NODE": str(len(devices)),
        "FORCE_QWENVL_VIDEO_READER": "torchcodec",
        "FPS_MAX_FRAMES": "16",
        "VIDEO_MAX_TOKEN_NUM": "128",
        "WANDB_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false",
    })
    command = [
        "swift", "rlhf",
        "--rlhf_type", "gkd",
        "--model", args.model,
        "--adapters", args.adapter,
        "--dataset", args.dataset,
        "--output_dir", args.output,
        "--add_version", "false",
        "--tuner_type", "lora",
        "--lora_rank", "16",
        "--lora_alpha", "32",
        "--target_modules", "all-linear",
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--torch_dtype", "bfloat16",
        "--attn_impl", "flash_attn",
        "--enable_thinking", "false",
        "--add_non_thinking_prefix", "true",
        "--lmbda", "1.0",
        "--beta", "1.0",
        "--temperature", "1.0",
        "--sft_alpha", "0",
        "--use_vllm", "false",
        "--use_liger_kernel", "false",
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", str(args.global_batch_size // len(devices)),
        "--learning_rate", str(args.learning_rate),
        "--max_length", str(args.max_length),
        "--max_completion_length", str(args.max_completion_length),
        "--split_dataset_ratio", "0",
        "--save_strategy", "steps",
        "--save_steps", str(args.save_steps),
        "--save_total_limit", "3",
        "--create_checkpoint_symlink", "true",
        "--logging_steps", "10",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--gradient_checkpointing", "true",
        "--dataloader_num_workers", "4",
        "--dataset_num_proc", "4",
        "--seed", "42",
        "--data_seed", "42",
        "--report_to", "none",
        "--check_model", "false",
    ]
    # 不传 teacher_model：官方动态 OPSD 会用当前 LoRA 权重加 teacher_prompt。
    if resume_path:
        command += ["--resume_from_checkpoint", str(resume_path)]
    print("OPSD:", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    state_path = checkpoint(output) / "trainer_state.json"
    history = json.loads(state_path.read_text(encoding="utf-8"))["log_history"]
    points = [(row["step"], row["loss"]) for row in history if "step" in row and "loss" in row]
    if not points:
        raise RuntimeError(f"训练结束但没有 loss 记录：{state_path}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot([step for step, _ in points], [loss for _, loss in points])
    axis.set(xlabel="Step", ylabel="GKD loss", title="Qwen3.5-4B OPSD")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "loss_curve.png", dpi=160)
    plt.close(figure)
    print(f"Adapter: {output / 'last'}\nLoss curve: {output / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
