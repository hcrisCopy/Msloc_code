"""基于 ms-swift 4.5.3 GKD/OPSD 继续训练 SFT LoRA；教师共享学生当前权重并读取文字真值。"""

from __future__ import annotations

import argparse
import hashlib
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True, help="已评测的 SFT LoRA last/checkpoint-* 目录")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True, help="-1 按 epochs 跑全量；正数用于限定更新步数")
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
    if args.max_steps != -1 and args.max_steps <= 0:
        raise ValueError("max-steps 只能是 -1 或正整数")
    if args.max_steps > 0 and args.save_steps > args.max_steps:
        raise ValueError("短跑时 save-steps 不能大于 max-steps，训练结束前须产生 checkpoint")
    if not Path(args.model).is_dir() or not Path(args.adapter).is_dir() or not Path(args.dataset).is_file():
        raise FileNotFoundError("模型、SFT adapter 或 OPSD 数据不存在")
    adapter = Path(args.adapter)
    adapter_weights = adapter / "adapter_model.safetensors"
    if not (adapter / "adapter_config.json").is_file() or not adapter_weights.is_file():
        raise FileNotFoundError(f"SFT LoRA 缺少 adapter 配置或权重：{adapter}")
    sft_config = json.loads((Path(args.adapter).parent / "sft_config.json").read_text(encoding="utf-8"))
    if (sft_config["frames"] != 40 or sft_config.get("sampling") != "trace16_8_16"
            or Path(sft_config["model"]).resolve() != Path(args.model).resolve()):
        raise ValueError("OPSD 必须从相同模型的 16/8/16 SFT adapter 开始")
    gate_path = Path(args.teacher_gate)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["passed"] is not True:
        raise ValueError("教师完整生成评测未优于学生，不能启动 OPSD")
    student_config = json.loads((Path(gate["student_eval"]) / "eval_config.json").read_text(encoding="utf-8"))
    if Path(student_config["adapter"]).resolve() != Path(args.adapter).resolve():
        raise ValueError("教师准入所用 SFT adapter 与 OPSD 初始 adapter 不同")
    if (student_config["adapter_config_sha256"] != file_sha256(adapter / "adapter_config.json")
            or student_config["adapter_weights_sha256"] != file_sha256(adapter_weights)):
        raise ValueError("教师准入后 SFT LoRA 权重发生变化")
    if Path(student_config["model"]).resolve() != Path(args.model).resolve():
        raise ValueError("教师准入所用模型与 OPSD 模型不同")
    teacher_config = json.loads((Path(gate["teacher_eval"]) / "eval_config.json").read_text(encoding="utf-8"))
    if (Path(teacher_config["model"]).resolve() != Path(args.model).resolve()
            or Path(teacher_config["adapter"]).resolve() != Path(args.adapter).resolve()):
        raise ValueError("教师准入必须使用与学生相同的 SFT 初始权重")
    if (student_config.get("sampling") != "trace16_8_16"
            or teacher_config.get("sampling") != "trace16_8_16"):
        raise ValueError("OPSD 师生预检必须使用同一 16/8/16 视频取帧")
    data_config = json.loads((Path(args.dataset).parent / "data_config.json").read_text(encoding="utf-8"))
    if data_config["frames"] != 40 or data_config.get("sampling") != "trace16_8_16":
        raise ValueError("OPSD 数据不是 16/8/16 版本；请重新运行 prepare_opsd.py")
    if data_config["student_prompt_text"] != student_config["prompt_text"].strip():
        raise ValueError("OPSD 学生提示词与教师准入评测不一致")
    if data_config["teacher_precheck_prompt_text"] != teacher_config["teacher_prompt_text"].strip():
        raise ValueError("OPSD 记录的教师预检提示词与教师准入评测不一致")
    if not data_config["teacher_opsd_prompt_text"]:
        raise ValueError("OPSD 训练教师提示词不能为空")
    for key in ("proposals", "annotation", "frames"):
        if data_config[key] != student_config[key]:
            raise ValueError(f"OPSD 数据与教师准入评测的 {key} 不一致")
    output = Path(args.output)
    resolved_output = output.resolve()
    data_root = Path("../MSLoc_data/Qwen").resolve()
    if resolved_output == data_root or not resolved_output.is_relative_to(data_root):
        raise ValueError("OPSD 输出必须位于 ../MSLoc_data/Qwen/ 下")
    if any(path.resolve() == resolved_output or path.resolve().is_relative_to(resolved_output)
           for path in (Path(args.model), Path(args.adapter), Path(args.dataset), gate_path,
                        Path(gate["student_eval"]), Path(gate["teacher_eval"]))):
        raise ValueError("OPSD 输出目录不能覆盖模型、权重、训练数据或教师预检结果")
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
        "LOG_LEVEL": "INFO",
        "NPROC_PER_NODE": str(len(devices)),
        "FORCE_QWENVL_VIDEO_READER": "torchcodec",
        "FPS_MAX_FRAMES": str(data_config["frames"]),
        "VIDEO_MAX_TOKEN_NUM": "128",
        "WANDB_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false",
    })
    command = [
        "swift", "rlhf",
        "--rlhf_type", "gkd",
        "--model", args.model,
        "--external_plugins", "Qwen/trace_video_template.py",
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
        "--max_steps", str(args.max_steps),
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
    # 记录完整训练指令和输入文件，防止 --resume auto 接到不同的实验。
    run_config = {
        "command": command.copy(),
        "devices": devices,
        "model": str(Path(args.model).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
        "adapter_weights_sha256": file_sha256(adapter_weights),
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_sha256": file_sha256(Path(args.dataset)),
        "teacher_gate": str(gate_path.resolve()),
        "teacher_gate_sha256": file_sha256(gate_path),
        "data_config": data_config,
        "sft_config": sft_config,
    }
    run_config_path = output / "opsd_config.json"
    if args.resume == "auto":
        previous_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if previous_config != run_config:
            raise ValueError("继续 OPSD 时训练参数、设备、数据、教师准入和 SFT adapter 路径必须与原运行一致")
    else:
        temporary_config = run_config_path.with_suffix(".json.tmp")
        temporary_config.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_config.replace(run_config_path)

    # 不传 teacher_model：同一模型在教师特权提示词下停止梯度，随后与学生一起更新 LoRA。
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
