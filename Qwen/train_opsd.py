"""基于 ms-swift 4.5.3 GKD/OPSD 继续训练 SFT LoRA；教师固定为初始 SFT 权重。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from opsd_teacher_filter import filter_samples, teacher_results
from proposal_sample import ids_sha256, select_ids


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
    parser.add_argument("--teacher-model", required=True, help="由 merge_sft_teacher.py 导出的独立 SFT 模型")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True, help="-1 按 epochs 跑全量；正数用于限定更新步数")
    parser.add_argument("--max-samples", type=int, default=-1, help="-1 全量数据；正数抽样 fake/real 用于调试")
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True, help="JSD 的混合系数；正式训练使用 0.5")
    parser.add_argument("--teacher-min-iou", type=float, required=True,
                        help="fake 片段教师回答与目标区间的最低 IoU；正式训练使用 0.5")
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
    if not math.isfinite(args.beta) or not 0 < args.beta < 1:
        raise ValueError("--beta 必须在 0 和 1 之间；正式 JSD 训练使用 0.5")
    if not math.isfinite(args.teacher_min_iou) or not 0 < args.teacher_min_iou <= 1:
        raise ValueError("--teacher-min-iou 必须在 0 和 1 之间，包含 1")
    if args.max_steps != -1 and args.max_steps <= 0:
        raise ValueError("max-steps 只能是 -1 或正整数")
    if args.max_samples != -1 and args.max_samples < 2:
        raise ValueError("max-samples 只能是 -1 或至少 2")
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
    for role in ("student", "teacher"):
        evaluation = Path(gate[f"{role}_eval"])
        for name, filename in (("config", "eval_config.json"), ("predictions", "predictions.json")):
            key = f"{role}_eval_{name}_sha256"
            if gate.get(key) != file_sha256(evaluation / filename):
                raise ValueError(f"{role} 预检结果与教师准入记录不一致；请重新运行 check_teacher.py")
    if args.max_samples == -1 and gate["max_proposals"] != -1:
        raise ValueError("全量 OPSD 必须先完成全量教师预检；抽样准入仅供短跑调试")
    if gate["max_proposals"] != -1 and gate["max_proposals"] != args.max_samples:
        raise ValueError("抽样 OPSD 与抽样教师预检须使用相同 proposal 数量")
    student_config = json.loads((Path(gate["student_eval"]) / "eval_config.json").read_text(encoding="utf-8"))
    if Path(student_config["adapter"]).resolve() != Path(args.adapter).resolve():
        raise ValueError("教师准入所用 SFT adapter 与 OPSD 初始 adapter 不同")
    if (student_config["adapter_config_sha256"] != file_sha256(adapter / "adapter_config.json")
            or student_config["adapter_weights_sha256"] != file_sha256(adapter_weights)):
        raise ValueError("教师准入后 SFT LoRA 权重发生变化")
    teacher_model = Path(args.teacher_model).resolve()
    if teacher_model == Path(args.model).resolve() or not (teacher_model / "config.json").is_file():
        raise ValueError("固定教师必须是独立合并的 SFT 模型，不能使用原始底座")
    source_path = teacher_model / "teacher_source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (Path(source["model"]).resolve() != Path(args.model).resolve()
            or Path(source["adapter"]).resolve() != adapter.resolve()
            or source["adapter_config_sha256"] != file_sha256(adapter / "adapter_config.json")
            or source["adapter_weights_sha256"] != file_sha256(adapter_weights)):
        raise ValueError("固定教师不是从当前预检的 SFT 权重合并得到的")
    if not list(teacher_model.glob("*.safetensors")):
        raise FileNotFoundError(f"固定教师缺少模型权重：{teacher_model}")
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
    if (teacher_config["teacher_prompt_file"] is None
            or not isinstance(teacher_config["teacher_prompt_text"], str)
            or teacher_config["teacher_prompt_text"].strip() != data_config["teacher_opsd_prompt_text"]):
        raise ValueError("教师预检和 OPSD 训练必须使用同一份教师提示词")
    for key in ("proposals", "annotation", "frames"):
        if data_config[key] != student_config[key]:
            raise ValueError(f"OPSD 数据与教师准入评测的 {key} 不一致")
    output = Path(args.output)
    resolved_output = output.resolve()
    data_root = Path("../MSLoc_data/Qwen").resolve()
    if resolved_output == data_root or not resolved_output.is_relative_to(data_root):
        raise ValueError("OPSD 输出必须位于 ../MSLoc_data/Qwen/ 下")
    if resolved_output.is_relative_to(teacher_model):
        raise ValueError("OPSD 输出目录不能位于固定教师模型内部")
    if any(path.resolve() == resolved_output or path.resolve().is_relative_to(resolved_output)
           for path in (Path(args.model), Path(args.adapter), teacher_model, Path(args.dataset), gate_path,
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
    # 训练教师提示词已对同批 proposal 贪心生成；逐条核对其回答，再建立训练数据。
    source_dataset = Path(args.dataset)
    audit_path = source_dataset.parent / "targets.jsonl"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    samples = source_dataset.read_text(encoding="utf-8").splitlines()
    audit = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    selected_ids = select_ids([(row["id"], row["target_kind"]) for row in audit], args.max_samples)
    if gate["max_proposals"] != -1 and gate["selected_ids_sha256"] != ids_sha256(selected_ids):
        raise ValueError("抽样教师预检与 OPSD 数据选中的 proposal 不一致")
    teacher_predictions_path = Path(gate["teacher_eval"]) / "predictions.json"
    expected_teacher_ids = (selected_ids if gate["max_proposals"] != -1
                            else {row["id"] for row in audit})
    filtered_text, decision_text, filter_summary = filter_samples(
        samples, audit, teacher_results(teacher_predictions_path), selected_ids,
        expected_teacher_ids, args.teacher_min_iou)
    train_dataset = output / "train_teacher_filtered.jsonl"
    decision_path = output / "teacher_filter.jsonl"
    summary_path = output / "teacher_filter_summary.json"
    summary_text = json.dumps(filter_summary, ensure_ascii=False, indent=2)
    for path, content in ((train_dataset, filtered_text), (decision_path, decision_text),
                          (summary_path, summary_text)):
        if args.resume == "auto":
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"续训时教师筛选结果发生变化：{path}")
        else:
            path.write_text(content, encoding="utf-8")
    print(f"OPSD 教师逐条筛选：{filter_summary['selected']} -> {filter_summary['retained']}；"
          f"排除 {filter_summary['excluded']}。记录：{decision_path}", flush=True)
    env = os.environ.copy()
    env.update({
        # 数据预处理的 Unix socket 必须使用短临时路径。
        "TMPDIR": "/tmp",
        "CUDA_VISIBLE_DEVICES": ",".join(devices),
        "LOG_LEVEL": "INFO",
        "NPROC_PER_NODE": str(len(devices)),
        "FORCE_QWENVL_VIDEO_READER": "torchcodec",
        "FPS_MAX_FRAMES": str(data_config["frames"]),
        "VIDEO_MAX_TOKEN_NUM": "128",
        "WANDB_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false",
    })
    # ms-swift 4.5.3 GKD 原生支持 0 < beta < 1 的 JSD。保留完整词表，
    # 使 Real、Interval 和时间数字也参与师生分布匹配。
    command = [
        "swift", "rlhf",
        "--rlhf_type", "gkd",
        "--model", args.model,
        "--teacher_model", str(teacher_model),
        "--external_plugins", "Qwen/trace_video_template.py",
        "--adapters", args.adapter,
        "--dataset", str(train_dataset),
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
        "--beta", str(args.beta),
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
        "--create_checkpoint_symlink", "false",
        "--logging_steps", "1",
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
        "teacher_model": str(teacher_model),
        "teacher_source_sha256": file_sha256(source_path),
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
        "adapter_weights_sha256": file_sha256(adapter_weights),
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_sha256": file_sha256(Path(args.dataset)),
        "max_samples": args.max_samples,
        "selected_ids_sha256": ids_sha256(selected_ids),
        "teacher_gate": str(gate_path.resolve()),
        "teacher_gate_sha256": file_sha256(gate_path),
        "teacher_eval_predictions_sha256": file_sha256(teacher_predictions_path),
        "teacher_min_iou": args.teacher_min_iou,
        "teacher_filter_sha256": file_sha256(decision_path),
        "filtered_dataset_sha256": file_sha256(train_dataset),
        "teacher_filter_summary": filter_summary,
        "teacher_filter_code_sha256": file_sha256(Path("Qwen/opsd_teacher_filter.py")),
        "divergence": "jsd",
        "beta": args.beta,
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

    # 独立合并的 SFT 模型作为冻结教师；只有学生 LoRA 随优化步骤更新。
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
    axis.set(xlabel="Step", ylabel="JSD loss", title="Qwen3.5-4B OPSD")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "loss_curve.png", dpi=160)
    plt.close(figure)
    last = output / "last"
    if last.is_symlink():
        last.unlink()
    elif last.exists():
        raise FileExistsError(f"不能覆盖已有的非链接目录：{last}")
    last.symlink_to(checkpoint(output).name, target_is_directory=True)
    print(f"Adapter: {last}\nLoss curve: {output / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
