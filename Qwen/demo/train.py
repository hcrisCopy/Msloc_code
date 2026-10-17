"""Short SFT/OPSD/GRPO runs with interchangeable HF weights and rollout backends."""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from demo_utils import (check_versions, data_identity, device_ids, environment, latest_checkpoint,
                        model_identity, prepare_output, read_json, read_rows, relative, require_root,
                        require_matching_processors,
                        sha256, source_identity, write_json, write_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=["sft", "opsd", "grpo"], required=True)
    p.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    p.add_argument("--model", required=True, help="Complete HF base OR fine-tuned/merged model directory")
    p.add_argument("--adapter", default="none", help="Optional LoRA directory; no old project metadata required")
    p.add_argument("--teacher-model", default="none")
    p.add_argument("--teacher-adapter", default="none")
    p.add_argument("--teacher-gate", default="none")
    p.add_argument("--alignment", default="none", help="Passed inspect_model.py alignment.json for vLLM")
    p.add_argument("--data", required=True)
    p.add_argument("--nli-model", default="none")
    p.add_argument("--output", required=True)
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--tuner", choices=["lora", "full"], default="lora")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--save-steps", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--vllm-memory", type=float, default=0.35)
    p.add_argument("--resume", choices=["none", "auto"], default="none")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--plan-only", action="store_true", help="Validate inputs and print command without loading GPUs or writing output")
    args = p.parse_args()
    require_root()
    devices = device_ids(args.devices)
    if (min(args.steps, args.save_steps, args.global_batch_size, args.max_new_tokens,
            args.lora_rank, args.lora_alpha) <= 0 or args.learning_rate <= 0
            or args.steps % args.save_steps or args.global_batch_size % len(devices)
            or args.max_length <= args.max_new_tokens or not 0 < args.vllm_memory < 1):
        raise ValueError("Invalid parameters: steps must be divisible by save-steps; batch must divide across GPUs.")
    if args.stage == "sft" and args.backend != "transformers":
        raise ValueError("SFT has no on-policy rollout; use --backend transformers.")
    if args.stage == "grpo" and (args.num_generations < 2 or args.global_batch_size % args.num_generations):
        raise ValueError("GRPO batch counts completions and must be divisible by num-generations >=2.")
    if args.tuner == "full" and args.adapter != "none":
        raise ValueError("For full tuning, export/merge the adapter first, then pass the complete model.")
    if args.adapter != "none":
        adapter_cfg = read_json(Path(args.adapter) / "adapter_config.json")
        if adapter_cfg["r"] != args.lora_rank or adapter_cfg["lora_alpha"] != args.lora_alpha:
            raise ValueError("Set --lora-rank/--lora-alpha to the supplied adapter's actual configuration.")
        targets = adapter_cfg.get("target_modules", [])
        if isinstance(targets, str) and targets != "all-linear":
            raise ValueError("Regex LoRA targets need explicit vLLM compatibility review; use a standard language LoRA.")
        if any("visual" in str(t) or "vision" in str(t) for t in targets):
            raise ValueError("This demo freezes vision and does not support vision LoRA.")
    weights = model_identity(args.model, args.adapter)
    data_hash = data_identity(args.data)
    versions = check_versions(args.backend == "vllm")
    sources = source_identity()
    extra_inputs = {}
    protected = [args.model, args.data]
    if args.adapter != "none":
        protected.append(args.adapter)
    dataset = Path(args.data) / f"{args.stage}.jsonl"
    rows = read_rows(dataset)
    if args.stage == "opsd":
        if args.teacher_model == "none" or args.teacher_gate == "none":
            raise ValueError("OPSD requires explicit --teacher-model and --teacher-gate.")
        teacher = model_identity(args.teacher_model, args.teacher_adapter)
        require_matching_processors(weights, teacher)
        same_base = Path(args.model).resolve() == Path(args.teacher_model).resolve()
        if same_base and (args.tuner != "lora" or args.adapter != "none" or args.teacher_adapter != "none"):
            raise ValueError("For this combination export a separate complete frozen teacher directory; same-path swift self-distillation would change teacher semantics.")
        gate = read_json(args.teacher_gate)
        if (gate["passed"] is not True or gate["student"]["weights"] != weights
                or gate["teacher"]["weights"] != teacher or gate["student"]["data"] != data_hash
                or gate["student"]["source"] != sources):
            raise ValueError("Teacher gate does not match the current student, teacher, data or source.")
        dataset = Path(gate["dataset"])
        if sha256(dataset) != gate["dataset_sha256"]:
            raise ValueError("Teacher-filtered dataset changed.")
        rows = read_rows(dataset)
        extra_inputs.update(teacher=teacher, gate_sha256=sha256(args.teacher_gate))
        protected += [args.teacher_model, args.teacher_gate, dataset]
        if args.teacher_adapter != "none":
            protected.append(args.teacher_adapter)
    if args.stage == "grpo":
        if args.nli_model == "none":
            raise ValueError("GRPO requires --nli-model.")
        nli = Path(args.nli_model)
        extra_inputs["nli"] = {name: sha256(nli / name) for name in
                               ("config.json", "model.safetensors", "tokenizer.json")}
        protected.append(nli)
    if args.backend == "vllm":
        if args.alignment == "none":
            raise ValueError("Run inspect_model.py --mode alignment before vLLM training.")
        alignment = read_json(args.alignment)
        expected_base = model_identity(args.model)
        if (alignment["passed"] is not True or alignment["config"]["weights"] != expected_base
                or alignment["config"]["data"] != data_hash or alignment["config"]["source"] != sources
                or alignment["config"]["max_length"] != args.max_length):
            raise ValueError("Alignment report does not match the current model, data, source or max-length.")
        extra_inputs["alignment_sha256"] = sha256(args.alignment)
        protected.append(args.alignment)
    unique_per_batch = args.global_batch_size // (args.num_generations if args.stage == "grpo" else 1)
    if len(rows) < unique_per_batch:
        raise ValueError(f"Need >= {unique_per_batch} samples after filtering; got {len(rows)}.")
    output = Path(args.output)
    train_dataset = output / "train.jsonl"
    module = "swift.cli.sft" if args.stage == "sft" else "swift.cli.rlhf"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={len(devices)}", "--module", module]

    def add(key, *values):
        command.extend(["--" + key, *map(str, values)])

    if args.stage != "sft":
        add("rlhf_type", "gkd" if args.stage == "opsd" else "grpo")
    for key, value in {"model": args.model, "model_type": "qwen3_5", "dataset": relative(train_dataset),
                       "output_dir": args.output, "tuner_type": args.tuner, "add_version": "false",
                       "load_args": "false",
                       "freeze_vit": "true", "freeze_aligner": "true", "torch_dtype": "bfloat16",
                       "attn_impl": "flash_attn", "enable_thinking": "false", "add_non_thinking_prefix": "true",
                       "max_steps": args.steps, "per_device_train_batch_size": 1,
                       "gradient_accumulation_steps": args.global_batch_size // len(devices),
                       "learning_rate": args.learning_rate, "max_length": args.max_length,
                       "truncation_strategy": "delete", "strict": "true", "split_dataset_ratio": 0,
                       "save_strategy": "steps", "save_steps": args.save_steps, "save_total_limit": 3,
                       "save_only_model": "false", "create_checkpoint_symlink": "false",
                       "logging_steps": 1, "warmup_ratio": 0.03, "lr_scheduler_type": "cosine",
                       "gradient_checkpointing": "true", "dataloader_num_workers": 4, "dataset_num_proc": 4,
                       "seed": 42, "data_seed": 42, "report_to": "none", "check_model": "false",
                       "use_liger_kernel": "false"}.items():
        add(key, value)
    if args.tuner == "lora":
        add("lora_rank", args.lora_rank)
        add("lora_alpha", args.lora_alpha)
        add("target_modules", "all-linear")
    if args.adapter != "none":
        add("adapters", args.adapter)
    plugins = ["Qwen/demo/video_plugin.py", "Qwen/demo/monitor_plugin.py"]
    if args.stage == "grpo":
        plugins += ["Qwen/grpo_rewards.py", "Qwen/grpo_logging.py"]
        add("reward_funcs", "msloc_localization", "msloc_format", "msloc_explanation")
        add("reward_weights", 1.0, 0.1, 0.3)
        for key, value in {"num_generations": args.num_generations, "num_iterations": 1,
                           "beta": 0.02, "temperature": 0.8, "top_p": 0.95,
                           "loss_type": "grpo", "scale_rewards": "group", "epsilon": 0.2,
                           "importance_sampling_level": "token", "kl_in_reward": "false",
                           "sync_ref_model": "false"}.items():
            add(key, value)
        if args.adapter != "none":
            add("ref_adapters", args.adapter)
    if args.stage == "opsd":
        add("teacher_model", args.teacher_model)
        if args.teacher_adapter != "none":
            add("teacher_adapters", args.teacher_adapter)
        for key, value in {"lmbda": 1.0, "beta": 0.5, "sft_alpha": 0, "temperature": 1.0, "top_p": 1.0}.items():
            add(key, value)
    if args.stage != "sft":
        add("remove_unused_columns", "false")
        add("async_generate", "false")
        add("generation_batch_size", args.global_batch_size)
        add("max_completion_length", args.max_new_tokens)
        add("log_completions", "true")
        add("top_k", -1)
        add("use_vllm", str(args.backend == "vllm").lower())
    if args.backend == "vllm":
        for key, value in {"vllm_mode": "colocate", "vllm_tensor_parallel_size": 1,
                           "vllm_gpu_memory_utilization": args.vllm_memory,
                           "vllm_max_model_len": args.max_length, "vllm_enable_lora": "false",
                           "vllm_enable_prefix_caching": "false", "vllm_mm_processor_cache_gb": 0,
                           "sleep_level": 0, "offload_model": "false", "offload_optimizer": "false"}.items():
            add(key, value)
    add("external_plugins", *plugins)
    config = {"kind": "training", "parameters": {k: v for k, v in vars(args).items()
              if k not in ("clean", "resume", "plan_only")}, "weights": weights,
              "data": data_hash, "dataset_sha256": sha256(dataset), "sources": sources,
              "versions": versions, "extra_inputs": extra_inputs, "command": command,
              "samples": len(rows), "unique_inputs_per_batch": unique_per_batch}
    print(f"Stage={args.stage} | backend={args.backend} | GPUs={len(devices)} | tuner={args.tuner}")
    print(f"Samples={len(rows)} | batch={args.global_batch_size} | unique inputs/batch={unique_per_batch}")
    print("Command:\n" + shlex.join(command))
    if args.plan_only:
        return
    output = prepare_output(output, config, args.resume, args.clean, protected)
    if args.resume == "auto":
        if sha256(train_dataset) != sha256(dataset):
            raise ValueError("Saved training rows differ from the current dataset.")
        add("resume_from_checkpoint", relative(latest_checkpoint(output)))
    else:
        write_rows(train_dataset, rows)
    env = environment(args.devices)
    env["MSLOC_DEMO_RUN"] = relative(output)
    if args.stage == "grpo":
        env["MSLOC_GRPO_NLI_MODEL"] = relative(args.nli_model)
    subprocess.run([sys.executable, "Qwen/demo/check_lengths.py", "--model", args.model,
                    "--dataset", relative(train_dataset), "--stage", args.stage,
                    "--max-length", str(args.max_length), "--max-new-tokens", str(args.max_new_tokens),
                    "--output", relative(output / "length_check.json")], env=env, check=True)
    write_json(output / "status.json", {"state": "running"})
    started = time.perf_counter()
    try:
        subprocess.run(command, env=env, check=True)
    except BaseException as exc:
        write_json(output / "status.json", {"state": "interrupted_or_failed", "error": str(exc)})
        raise
    checkpoint = latest_checkpoint(output)
    state = read_json(checkpoint / "trainer_state.json")
    if state["global_step"] != args.steps:
        raise RuntimeError("Training returned without the expected final checkpoint.")
    write_json(output / "status.json", {"state": "complete", "checkpoint": relative(checkpoint),
               "model": relative(args.model), "tuner": args.tuner,
               "process_wall_seconds": time.perf_counter() - started})
    subprocess.run([sys.executable, "Qwen/demo/report.py", "--runs", relative(output),
                    "--output", relative(output / "report"), "--warmup-steps", "4"], check=True)


if __name__ == "__main__":
    main()
