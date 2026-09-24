"""Qwen3.5 原模型/SFT LoRA 共用评测：逐 proposal 推理后调用 Trace 同款 evaluate_long.py。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from common import clip_name, clip_timestamps_path, make_clip, parse_answer, proposal_segments, read_records
from opsd_common import proposal_target, student_message, teacher_message


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True, help="none 或 SFT 的 last/checkpoint-* 目录")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--teacher-prompt-file", help="仅教师完整生成评测使用；学生评测不传")
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--resume", choices=["none", "auto"], required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    if args.frames != 40 or args.max_new_tokens <= 0:
        raise ValueError("Qwen 输入 frames 必须是 40，max-new-tokens 必须大于 0")
    if not Path(args.model).is_dir():
        raise FileNotFoundError(args.model)
    if args.adapter != "none" and not Path(args.adapter).is_dir():
        raise FileNotFoundError(args.adapter)
    if args.adapter != "none" and any(
        not (Path(args.adapter) / name).is_file()
        for name in ("adapter_config.json", "adapter_model.safetensors")
    ):
        raise FileNotFoundError(f"LoRA 配置或权重缺失：{args.adapter}")
    if not Path(args.prompt_file).is_file():
        raise FileNotFoundError(args.prompt_file)
    if args.teacher_prompt_file and not Path(args.teacher_prompt_file).is_file():
        raise FileNotFoundError(args.teacher_prompt_file)
    proposals, gt = read_records(args.proposals), read_records(args.annotation)
    if {row["video_path"] for row in proposals} != {row["video_path"] for row in gt}:
        raise ValueError("测试 proposal 与 GT 的 video_path 集合不同")
    return proposals, gt


def launch(args: argparse.Namespace) -> None:
    validate(args)
    devices = [item.strip() for item in args.devices.split(",")]
    if not devices or any(not item.isdigit() for item in devices) or len(set(devices)) != len(devices):
        raise ValueError("--devices 应为不重复的 GPU 编号")
    output = Path(args.output)
    data_root = Path("../MSLoc_data/Qwen").resolve()
    resolved_output = output.resolve()
    if resolved_output == data_root or not resolved_output.is_relative_to(data_root):
        raise ValueError("评测输出必须位于 ../MSLoc_data/Qwen/ 下")
    protected = [Path(args.model), Path(args.proposals), Path(args.annotation), Path(args.video_root),
                 Path(args.prompt_file)]
    if args.adapter != "none":
        protected.append(Path(args.adapter))
    if args.teacher_prompt_file:
        protected.append(Path(args.teacher_prompt_file))
    if any(path.resolve() == resolved_output or path.resolve().is_relative_to(resolved_output)
           for path in protected):
        raise ValueError("评测输出目录不能覆盖模型、权重、输入数据或提示词")
    if args.clean and args.resume != "none":
        raise ValueError("--clean 与 --resume auto 不能同时使用")
    if args.clean and output.exists():
        shutil.rmtree(output)
    if args.resume == "none" and output.exists():
        raise FileExistsError(f"输出目录已存在：{output}；使用 --resume auto 或 --clean")
    if args.resume == "auto" and not output.exists():
        raise FileNotFoundError(f"没有可继续的评测目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "eval_config.json"
    adapter_config_hash = None
    adapter_weights_hash = None
    if args.adapter != "none":
        adapter_config_hash = file_sha256(Path(args.adapter) / "adapter_config.json")
        adapter_weights_hash = file_sha256(Path(args.adapter) / "adapter_model.safetensors")
    config = {
        "model": args.model,
        "adapter": args.adapter,
        "adapter_config_sha256": adapter_config_hash,
        "adapter_weights_sha256": adapter_weights_hash,
        "proposals": args.proposals,
        "proposals_sha256": file_sha256(Path(args.proposals)),
        "annotation": args.annotation,
        "annotation_sha256": file_sha256(Path(args.annotation)),
        "video_root": args.video_root,
        "prompt_file": args.prompt_file,
        "prompt_text": Path(args.prompt_file).read_text(encoding="utf-8"),
        "teacher_prompt_file": args.teacher_prompt_file,
        "teacher_prompt_text": Path(args.teacher_prompt_file).read_text(encoding="utf-8") if args.teacher_prompt_file else None,
        "devices": args.devices,
        "frames": args.frames,
        "sampling": "trace16_8_16",
        "max_new_tokens": args.max_new_tokens,
    }
    if args.resume == "auto":
        if json.loads(manifest_path.read_text(encoding="utf-8")) != config:
            raise ValueError("继续评测时模型、设备、数据和提示词必须与原运行一致")
    else:
        manifest_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(devices),
        "LOG_LEVEL": "INFO",
        "FORCE_QWENVL_VIDEO_READER": "torchcodec",
        "FPS_MAX_FRAMES": str(args.frames),
        "VIDEO_MAX_TOKEN_NUM": "128",
        "TOKENIZERS_PARALLELISM": "false",
    })
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={len(devices)}", str(Path(__file__)),
        "--model", args.model, "--adapter", args.adapter,
        "--proposals", args.proposals, "--annotation", args.annotation,
        "--video-root", args.video_root, "--prompt-file", args.prompt_file,
        "--output", args.output, "--devices", args.devices,
        "--frames", str(args.frames), "--max-new-tokens", str(args.max_new_tokens),
        "--resume", args.resume, "--worker",
    ]
    if args.teacher_prompt_file:
        command += ["--teacher-prompt-file", args.teacher_prompt_file]
    print("Eval:", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def saved_results(path: Path) -> dict[str, dict]:
    results = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["id"] in results:
                    raise ValueError(f"重复评测 ID：{row['id']}")
                results[row["id"]] = row
    return results


def run_worker(args: argparse.Namespace) -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    proposals, gt_rows = validate(args)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    teacher_prompt = Path(args.teacher_prompt_file).read_text(encoding="utf-8").strip() if args.teacher_prompt_file else None
    annotations = {row["video_path"]: row for row in gt_rows}
    output = Path(args.output)
    shard = output / f"rank_{rank}.jsonl"
    completed = saved_results(shard)
    tasks = []
    for row in proposals:
        video_name = row["video_path"]
        relative_video = Path(video_name)
        if relative_video.is_absolute() or ".." in relative_video.parts:
            raise ValueError(f"video_path 必须是视频根目录下的相对路径：{video_name}")
        for index, segment in enumerate(proposal_segments(row)):
            task_id = f"{video_name}::{index}"
            tasks.append((task_id, video_name, relative_video, index, segment))
    tasks = [task for index, task in enumerate(tasks) if index % world == rank and task[0] not in completed]
    if tasks:
        from trace_video_template import register_trace_video_template
        from swift import get_template
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

        register_trace_video_template()
        adapters = [] if args.adapter == "none" else [args.adapter]
        engine = TransformersEngine(
            args.model,
            adapters=adapters,
            adapter_names=["sft"] if adapters else None,
            torch_dtype=torch.bfloat16,
            attn_impl="flash_attn",
            device_map={"": f"cuda:{int(os.environ['LOCAL_RANK'])}"},
        )
        if adapters:
            engine.model.set_adapter("sft")
        engine.template = get_template(engine.processor, enable_thinking=False)
        config = RequestConfig(max_tokens=args.max_new_tokens, temperature=0)
        with shard.open("a", encoding="utf-8") as handle:
            for task_id, video_name, relative_video, index, proposal in tqdm(
                tasks, desc=f"Qwen eval rank {rank}", unit="proposal", disable=rank != 0
            ):
                clip = output / "clips" / relative_video.with_suffix("") / clip_name(index, proposal)
                make_clip(Path(args.video_root) / relative_video, proposal, clip, args.frames)
                duration = proposal[1] - proposal[0]
                message = student_message(prompt, duration)
                # 真值只用于教师特权提示词和事后审计；普通学生请求不包含它。
                target = proposal_target(proposal, annotations[video_name])
                if teacher_prompt:
                    message = teacher_message(message, teacher_prompt, target, mode="precheck")
                request = InferRequest(
                    messages=[{"role": "user", "content": message}],
                    videos=[str(clip)],
                    chat_template_kwargs={"nframes": 40},
                )
                response = engine.infer([request], request_config=config)[0].choices[0].message.content
                raw = "" if response is None else str(response)
                parsed = parse_answer(raw, duration)
                absolute = None
                if parsed["relative_segment"] is not None:
                    absolute = [proposal[0] + value for value in parsed["relative_segment"]]
                result = {
                    "id": task_id,
                    "video_path": video_name,
                    "proposal_index": index,
                    "proposal": list(proposal),
                    "clip": str(clip),
                    "timestamps": str(clip_timestamps_path(clip)),
                    "raw_response": raw,
                    "status": parsed["status"],
                    "target_kind": target["kind"],
                    "target_relative": target["relative_segment"],
                    "source_video_type": target["source_video_type"],
                    "replay_bucket": target["replay_bucket"],
                    "explanation": parsed["explanation"],
                    "segment": absolute,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
    dist.barrier()
    if rank == 0:
        merge(args, proposals, world)
    dist.barrier()
    dist.destroy_process_group()


def merge(args: argparse.Namespace, proposals: list[dict], world: int) -> None:
    output = Path(args.output)
    all_results = {}
    for rank in range(world):
        for task_id, row in saved_results(output / f"rank_{rank}.jsonl").items():
            if task_id in all_results:
                raise ValueError(f"重复评测 ID：{task_id}")
            all_results[task_id] = row
    expected = {f"{row['video_path']}::{index}" for row in proposals for index, _ in enumerate(proposal_segments(row))}
    if set(all_results) != expected:
        raise ValueError(f"评测不完整：期望 {len(expected)} 条，实际 {len(all_results)} 条")
    predictions = []
    counts = defaultdict(int)
    proposal_detection = defaultdict(int)
    invalid_ids = []
    for source in proposals:
        video_name = source["video_path"]
        results = [all_results[f"{video_name}::{index}"] for index, _ in enumerate(proposal_segments(source))]
        for result in results:
            # 续跑时也按当前解析规则重算已有原文，避免旧的格式判定污染最终指标。
            proposal = result["proposal"]
            parsed = parse_answer(result["raw_response"], proposal[1] - proposal[0])
            result["status"] = parsed["status"]
            result["explanation"] = parsed["explanation"]
            result["segment"] = (
                [proposal[0] + value for value in parsed["relative_segment"]]
                if parsed["relative_segment"] is not None else None
            )
            counts[result["status"]] += 1
            if result["status"] in {"format_error", "range_error"}:
                outcome = "invalid_positive" if result["target_kind"] == "fake" else "invalid_negative"
            elif result["target_kind"] == "fake":
                outcome = "true_positive" if result["status"] == "fake" else "false_negative"
            else:
                outcome = "false_positive" if result["status"] == "fake" else "true_negative"
            result["proposal_detection"] = outcome
            proposal_detection[outcome] += 1
            if result["status"] in {"format_error", "range_error"}:
                invalid_ids.append(result["id"])
        segments = [result["segment"] for result in results if result["status"] == "fake"]
        if segments:
            decision, decision_status = "fake", "semantic_event"
        elif results and all(result["status"] == "real" for result in results):
            decision, decision_status = "real", "semantic_no_event"
        elif not results:
            decision, decision_status = "real", "no_proposal"
        else:
            decision, decision_status = "invalid", "format_failure"
        predictions.append({
            "video_path": video_name,
            "model_inference": {"type": decision, "segment": segments, "decision_status": decision_status},
            "proposal_results": results,
        })
    predictions_path = output / "predictions.json"
    temporary = output / "predictions.json.tmp"
    temporary.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(predictions_path)
    summary = {
        "counts": dict(counts), "proposal_detection": dict(proposal_detection),
        "invalid_proposal_ids": invalid_ids,
    }
    (output / "parse_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Parse:", dict(counts), flush=True)
    print("Proposal detection:", dict(proposal_detection), flush=True)
    subprocess.run([
        sys.executable, "evaluate_long.py",
        "--gt_file", args.annotation,
        "--infer_file", str(predictions_path),
        "--output_file", str(output / "metrics.json"),
    ], check=True)


if __name__ == "__main__":
    parsed_args = arguments()
    if parsed_args.worker:
        run_worker(parsed_args)
    else:
        launch(parsed_args)
