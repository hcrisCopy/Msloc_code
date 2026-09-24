"""按 ms-swift 官方 Qwen3.5 LoRA 接口启动 SFT；一张或八张 GPU 使用同一全局 batch。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def checkpoint(output: Path) -> Path:
    found = list(output.glob("checkpoint-*"))
    found = [path for path in found if path.is_dir() and path.name.split("-")[-1].isdigit()]
    if not found:
        raise FileNotFoundError(f"没有可继续的 checkpoint：{output}")
    return max(found, key=lambda path: int(path.name.split("-")[-1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", required=True, help="逗号分隔，如 0 或 0,1,2,3,4,5,6,7")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--resume", choices=["none", "auto"], required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    devices = [item.strip() for item in args.devices.split(",")]
    if not devices or any(not item.isdigit() for item in devices) or len(set(devices)) != len(devices):
        raise ValueError("--devices 应为不重复的 GPU 编号")
    if args.global_batch_size <= 0 or args.global_batch_size % len(devices):
        raise ValueError("global-batch-size 必须是正数且能被 GPU 数整除")
    if args.epochs <= 0 or args.save_steps <= 0 or args.max_length <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs、save-steps、max-length 和 learning-rate 必须大于 0")
    if not Path(args.model).is_dir() or not Path(args.dataset).is_file():
        raise FileNotFoundError("模型目录或训练数据不存在")
    output = Path(args.output)
    data_root = Path("../MSLoc_data/Qwen").resolve()
    if not output.resolve().is_relative_to(data_root):
        raise ValueError("训练输出必须位于 ../MSLoc_data/Qwen/ 下")
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
        "swift", "sft",
        "--model", args.model,
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
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", str(args.global_batch_size // len(devices)),
        "--learning_rate", str(args.learning_rate),
        "--max_length", str(args.max_length),
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
    if resume_path:
        command += ["--resume_from_checkpoint", str(resume_path)]
    print("SFT:", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    state_path = checkpoint(output) / "trainer_state.json"
    with state_path.open(encoding="utf-8") as handle:
        history = json.load(handle)["log_history"]
    points = [(row["step"], row["loss"]) for row in history if "step" in row and "loss" in row]
    if not points:
        raise RuntimeError(f"训练结束但没有 loss 记录：{state_path}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot([step for step, _ in points], [loss for _, loss in points])
    axis.set(xlabel="Step", ylabel="Training loss", title="Qwen3.5-4B SFT")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "loss_curve.png", dpi=160)
    plt.close(figure)
    print(f"Adapter: {output / 'last'}\nLoss curve: {output / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
