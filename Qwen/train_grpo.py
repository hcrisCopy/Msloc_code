"""从 SFT 或 OPSD LoRA 继续训练 Qwen3.5-4B 的 proposal 级 GRPO。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_checkpoint(output: Path) -> Path:
    paths = [path for path in output.glob("checkpoint-*") if path.is_dir() and path.name[11:].isdigit()]
    if not paths:
        raise FileNotFoundError(f"没有 GRPO checkpoint：{output}")
    return max(paths, key=lambda path: int(path.name[11:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True, help="SFT 或 OPSD 的 last/checkpoint-* LoRA")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--nli-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True, help="-1 按 epochs 跑全量；正数用于限定更新步数")
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--num-generations", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-completion-length", type=int, required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--resume", choices=["none", "auto"], required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    devices = [value.strip() for value in args.devices.split(",")]
    if not devices or any(not value.isdigit() for value in devices) or len(devices) != len(set(devices)):
        raise ValueError("--devices 应为不重复的 GPU 编号")
    if (args.global_batch_size <= 0 or args.global_batch_size % len(devices)
            or args.num_generations < 2 or args.global_batch_size % args.num_generations):
        raise ValueError("global-batch-size 必须同时被 GPU 数和 num-generations 整除")
    if min(args.epochs, args.max_length, args.max_completion_length, args.save_steps) <= 0 or args.learning_rate <= 0:
        raise ValueError("训练轮次、长度、保存步数和学习率必须大于 0")
    if args.max_steps != -1 and args.max_steps <= 0:
        raise ValueError("max-steps 只能是 -1 或正整数")
    if args.max_steps > 0 and args.save_steps > args.max_steps:
        raise ValueError("短跑时 save-steps 不能大于 max-steps，训练结束前须产生 checkpoint")
    model, adapter, dataset, nli_model = map(Path, (args.model, args.adapter, args.dataset, args.nli_model))
    if any(not path.is_dir() for path in (model, adapter, nli_model)) or not dataset.is_file():
        raise FileNotFoundError("模型、SFT/OPSD adapter、GRPO 数据或本地 NLI 模型缺失")
    adapter_weights = adapter / "adapter_model.safetensors"
    if not (adapter / "adapter_config.json").is_file() or not adapter_weights.is_file():
        raise FileNotFoundError(f"LoRA 缺少 adapter 配置或权重：{adapter}")
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        if not (nli_model / name).is_file():
            raise FileNotFoundError(f"冻结 NLI 模型缺少 {name}：{nli_model}")
    data_config = json.loads((dataset.parent / "data_config.json").read_text(encoding="utf-8"))
    sft_config_path = adapter.parent / "sft_config.json"
    opsd_config_path = adapter.parent / "opsd_config.json"
    if sft_config_path.is_file() == opsd_config_path.is_file():
        raise ValueError("LoRA 所在目录必须恰有 sft_config.json 或 opsd_config.json")
    adapter_stage = "sft" if sft_config_path.is_file() else "opsd"
    stage_config_path = sft_config_path if adapter_stage == "sft" else opsd_config_path
    stage_config = json.loads(stage_config_path.read_text(encoding="utf-8"))
    if Path(stage_config["model"]).resolve() != model.resolve():
        raise ValueError(f"GRPO 模型必须与 {adapter_stage.upper()} 使用的模型相同")
    if (data_config.get("complete") is not True or data_config["sampling"] != "trace16_8_16"
            or data_config["frames"] != 40):
        raise ValueError("GRPO 数据必须使用 16/8/16 的 40 帧片段")
    if adapter_stage == "opsd":
        if data_config["prompt_text"] != stage_config["data_config"]["student_prompt_text"]:
            raise ValueError("GRPO 与 OPSD 学生提示词不一致")
        if Path(data_config["opsd_dataset"]).resolve() != Path(stage_config["dataset"]).resolve():
            raise ValueError("GRPO 数据来源与 OPSD 训练数据不一致")
    else:
        # 跳过 OPSD 训练时，仍复用 prepare_opsd 制作的真假片段；核对它与 SFT 的第一阶段输入相同。
        if stage_config["frames"] != 40 or stage_config["sampling"] != "trace16_8_16":
            raise ValueError("SFT LoRA 必须使用相同的 16/8/16 取帧方式")
        sft_data_path = Path(stage_config["dataset"]).parent / "data_config.json"
        opsd_data_path = Path(data_config["opsd_dataset"]).parent / "data_config.json"
        sft_data_config = json.loads(sft_data_path.read_text(encoding="utf-8"))
        opsd_data_config = json.loads(opsd_data_path.read_text(encoding="utf-8"))
        if sft_data_config.get("complete") is not True or opsd_data_config.get("complete") is not True:
            raise ValueError("SFT 与 OPSD 片段数据必须准备完成")
        for key in ("proposals_sha256", "annotation_sha256", "video_root", "frames", "sampling"):
            if sft_data_config["inputs"][key] != opsd_data_config["inputs"][key]:
                raise ValueError(f"SFT 与 GRPO 的第一阶段数据不一致：{key}")
        if data_config["prompt_text"] != opsd_data_config["student_prompt_text"]:
            raise ValueError("GRPO 必须使用 OPSD 片段数据的 student.txt 提示词")
        if Path(data_config["annotation"]).resolve() != Path(opsd_data_config["annotation"]).resolve():
            raise ValueError("GRPO 与 OPSD 片段数据的标注文件不一致")
    output = Path(args.output)
    data_root = Path("../MSLoc_data/Qwen").resolve()
    if output.resolve() == data_root or not output.resolve().is_relative_to(data_root):
        raise ValueError("GRPO 输出必须位于 ../MSLoc_data/Qwen/ 下")
    protected = (model.resolve(), adapter.resolve(), dataset.resolve(), nli_model.resolve())
    if any(path == output.resolve() or path.is_relative_to(output.resolve()) for path in protected):
        raise ValueError("GRPO 输出目录不能覆盖模型、初始 adapter、训练数据或 NLI 模型")
    if args.clean and args.resume != "none":
        raise ValueError("--clean 与 --resume auto 不能同时使用")
    if args.clean and output.exists():
        shutil.rmtree(output)
    if args.resume == "none" and output.exists():
        raise FileExistsError(f"输出目录已存在：{output}；使用 --resume auto 或 --clean")
    resume_path = latest_checkpoint(output) if args.resume == "auto" else None
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        # 数据预处理的 Unix socket 必须使用短临时路径。
        "TMPDIR": "/tmp",
        "CUDA_VISIBLE_DEVICES": ",".join(devices),
        "LOG_LEVEL": "INFO",
        "NPROC_PER_NODE": str(len(devices)),
        "FORCE_QWENVL_VIDEO_READER": "torchcodec",
        "FPS_MAX_FRAMES": "40",
        "VIDEO_MAX_TOKEN_NUM": "128",
        "MSLOC_GRPO_NLI_MODEL": str(nli_model.resolve()),
        "WANDB_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": os.pathsep.join((str(Path.cwd()), str(Path("Qwen").resolve()), env.get("PYTHONPATH", ""))),
    })
    command = [
        "swift", "rlhf", "--rlhf_type", "grpo",
        "--model", args.model,
        "--adapters", args.adapter,
        "--ref_adapters", args.adapter,
        "--dataset", args.dataset,
        "--external_plugins", "Qwen/trace_video_template.py", "Qwen/grpo_rewards.py",
        "--reward_funcs", "msloc_localization", "msloc_format", "msloc_explanation",
        "--reward_weights", "1.0", "0.1", "0.3",
        "--output_dir", args.output,
        "--add_version", "false",
        "--tuner_type", "lora",
        "--lora_rank", "16", "--lora_alpha", "32", "--target_modules", "all-linear",
        "--freeze_vit", "true", "--freeze_aligner", "true",
        "--torch_dtype", "bfloat16", "--attn_impl", "flash_attn",
        "--enable_thinking", "false", "--add_non_thinking_prefix", "true",
        "--use_vllm", "false", "--use_liger_kernel", "false",
        "--num_generations", str(args.num_generations),
        "--generation_batch_size", str(args.global_batch_size),
        "--num_iterations", "1", "--beta", "0.02",
        "--temperature", "0.8", "--top_p", "0.95",
        "--num_train_epochs", str(args.epochs),
        "--max_steps", str(args.max_steps),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", str(args.global_batch_size // len(devices)),
        "--learning_rate", str(args.learning_rate),
        "--max_length", str(args.max_length),
        "--max_completion_length", str(args.max_completion_length),
        "--truncation_strategy", "delete",
        "--split_dataset_ratio", "0",
        "--save_strategy", "steps", "--save_steps", str(args.save_steps),
        "--save_total_limit", "3", "--create_checkpoint_symlink", "false",
        "--logging_steps", "1", "--log_completions", "true",
        "--warmup_ratio", "0.03", "--lr_scheduler_type", "cosine",
        "--gradient_checkpointing", "true",
        "--dataloader_num_workers", "4", "--dataset_num_proc", "4",
        "--seed", "42", "--data_seed", "42",
        "--report_to", "none", "--check_model", "false",
    ]
    run_config = {
        "command": command.copy(),
        "devices": devices,
        "model": str(model.resolve()),
        "adapter": str(adapter.resolve()),
        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
        "adapter_weights_sha256": sha256(adapter_weights),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256(dataset),
        "nli_model": str(nli_model.resolve()),
        "nli_config_sha256": sha256(nli_model / "config.json"),
        "data_config": data_config,
    }
    if adapter_stage == "opsd":
        run_config["opsd_config"] = stage_config
    else:
        run_config["adapter_stage"] = "sft"
        run_config["sft_config"] = stage_config
        run_config["sft_data_config"] = sft_data_config
        run_config["opsd_data_config"] = opsd_data_config
    config_path = output / "grpo_config.json"
    if resume_path:
        if json.loads(config_path.read_text(encoding="utf-8")) != run_config:
            raise ValueError("GRPO 续训时数据、初始 adapter、NLI 模型和训练参数必须与原运行一致")
        command += ["--resume_from_checkpoint", str(resume_path)]
    else:
        config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GRPO 初始权重：{adapter_stage.upper()}", flush=True)
    print("GRPO:", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    state = json.loads((latest_checkpoint(output) / "trainer_state.json").read_text(encoding="utf-8"))
    history = [row for row in state["log_history"] if "step" in row and "reward" in row]
    if not history:
        raise RuntimeError("GRPO 结束但没有 reward 曲线记录")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot([row["step"] for row in history], [row["reward"] for row in history], label="Total reward")
    axis.set(xlabel="Step", ylabel="Reward", title="Qwen3.5-4B GRPO")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "reward_curve.png", dpi=160)
    plt.close(figure)
    last = output / "last"
    if last.is_symlink():
        last.unlink()
    elif last.exists():
        raise FileExistsError(f"不能覆盖已有的非链接目录：{last}")
    last.symlink_to(latest_checkpoint(output).name, target_is_directory=True)
    print(f"Adapter: {last}\nReward curve: {output / 'reward_curve.png'}")


if __name__ == "__main__":
    main()
