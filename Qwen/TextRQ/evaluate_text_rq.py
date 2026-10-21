"""Prepare, run and summarize local TextRQ without changing the legacy scripts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

from text_rq_data import prepare_video, required_stages, select_tasks
from text_rq_metrics import association_analysis, iou_pairs, selection_stats, summarize
from text_rq_report import build_report, write_report
from text_rq_storage import (
    atomic_json, atomic_jsonl, checked_result, fingerprint, model_identity,
    read_json, request_path, result_path, sha256,
)
from common import read_records


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Qwen/evaluate.py predictions.json")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--model", required=True, help="Local frozen Qwen3 / Qwen3.5 checkpoint")
    for name in ("common", "fluency", "consistency", "factual"):
        parser.add_argument(f"--{name}-prompt", default=f"Qwen/prompts/text_rq_{name}.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--min-iou", type=float, default=0.7, help="High-IoU subset only; not an input filter")
    parser.add_argument("--sample-count", type=int, default=-1,
                        help="-1: all valid TP/TN proposals with Explanation")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000, help="Video bootstrap; 0 disables CI")
    parser.add_argument("--batch-size", type=int, default=32, help="Requests per GPU batch")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--resume", choices=("none", "auto"), default="none")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare tasks without loading a judge")
    return parser.parse_args()


def configuration(args) -> dict:
    if not 0 <= args.min_iou < 1 or args.sample_count == 0 or args.sample_count < -1:
        raise ValueError("Require 0 <= min-iou < 1 and sample-count=-1 or a positive integer")
    if (args.batch_size <= 0 or args.max_new_tokens <= 0
            or args.max_model_len <= args.max_new_tokens or args.max_num_batched_tokens <= 0
            or not 0 < args.gpu_memory_utilization < 1):
        raise ValueError("Invalid batch, context, output budget, or GPU memory fraction")
    if (not 0 <= args.temperature < float("inf") or not 0 < args.top_p <= 1
            or (args.top_k != -1 and args.top_k <= 0)):
        raise ValueError("Invalid sampling parameters")
    devices = args.devices.split(",")
    if any(not item.isdigit() for item in devices) or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain distinct GPU indices")
    if args.clean and args.resume != "none":
        raise ValueError("--clean requires --resume none")
    if args.bootstrap_repeats < 0:
        raise ValueError("--bootstrap-repeats must be nonnegative")
    code_root = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve() != code_root:
        raise ValueError("Run this command from MSLoc_code/")
    for field in ("predictions", "annotation", "model", "common_prompt", "fluency_prompt",
                  "consistency_prompt", "factual_prompt", "output"):
        if Path(getattr(args, field)).is_absolute():
            raise ValueError(f"--{field.replace('_', '-')} must use a relative path")
    versions = {name: importlib.metadata.version(name)
                for name in ("vllm", "transformers", "torch", "scipy")}
    if versions["vllm"] != "0.19.1":
        raise RuntimeError(
            "TextRQ requires vLLM 0.19.1 from the msloc_qwen35 environment (Qwen/requirements.txt)"
        )
    sources = list(Path(__file__).parent.glob("text_rq_*.py")) + [Path(__file__)]
    sources += [code_root / "Qwen/common.py"]
    return {
        "protocol": "text_rq_v3", "seed": 42,
        "predictions": {"path": args.predictions, "sha256": sha256(Path(args.predictions))},
        "annotation": {"path": args.annotation, "sha256": sha256(Path(args.annotation))},
        "model": model_identity(Path(args.model)), "versions": versions,
        "source_sha256": {path.relative_to(code_root).as_posix(): sha256(path)
                          for path in sorted(sources)},
        "prompts": {name: {"path": getattr(args, f"{name}_prompt"),
                           "text": Path(getattr(args, f"{name}_prompt")).read_text(encoding="utf-8").strip()}
                    for name in ("common", "fluency", "consistency", "factual")},
        "min_iou": args.min_iou, "sample_count": args.sample_count,
        "bootstrap_repeats": args.bootstrap_repeats,
        "matching": "single_visible_gt_else_unique_positive_max_prediction_iou_allow_gt_reuse",
        "scope": "valid_tp_tn_with_explanation_no_iou_filter",
        "aggregation": "equal_weight_per_scored_proposal",
        "inference": {
            "thinking": False, "dtype": "bfloat16", "tensor_parallel_size": 1,
            "batch_size": args.batch_size, "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        },
    }


def initialize(output: Path, config: dict, args) -> None:
    manifest = output / "run_config.json"
    if args.clean and output.exists():
        # A clean operation is limited to a recognized TextRQ run directory.
        if (not manifest.is_file()
                or read_json(manifest)["config"]["protocol"] not in {
                    "text_rq_v1", "text_rq_v2", "text_rq_v3"}):
            raise ValueError("--clean refuses a directory without a TextRQ run_config.json")
        shutil.rmtree(output)
    if args.resume == "auto":
        saved = read_json(manifest)
        if saved != {"config": config, "run_id": fingerprint(config)}:
            raise ValueError("Resume requires identical inputs, model files, prompts and scoring settings")
    else:
        if output.exists():
            raise FileExistsError(f"{output}: use --resume auto, or --resume none --clean")
        output.mkdir(parents=True)
        atomic_json(manifest, {"config": config, "run_id": fingerprint(config)})
    for name in ("preparation", "requests", "results", "failures", "logs"):
        (output / name).mkdir(exist_ok=True)


def prepare(output: Path, config: dict) -> tuple[list[dict], list[dict], int]:
    predictions = read_records(config["predictions"]["path"])
    annotations = read_records(config["annotation"]["path"])
    ground_truth = {row["video_path"]: row for row in annotations}
    if not {row["video_path"] for row in predictions} <= set(ground_truth):
        raise ValueError("Prediction video_path missing from annotation")
    tasks, audit, no_proposal = [], [], 0
    for index, prediction in enumerate(tqdm(predictions, desc="Prepare TextRQ", unit="video")):
        path = output / "preparation" / f"video_{index:08d}.json"
        if path.exists():
            prepared = read_json(path)
            if prepared["video_path"] != prediction["video_path"]:
                raise ValueError(f"Preparation checkpoint mismatch: {path}")
        else:
            prepared = prepare_video(prediction, ground_truth[prediction["video_path"]], config["min_iou"])
            atomic_json(path, prepared)
        tasks.extend(prepared["tasks"])
        audit.extend(prepared["audit"])
        no_proposal += int(prepared["no_proposal"])
    chosen = select_tasks(tasks, config["sample_count"])
    chosen = [{**task, "sample_index": index} for index, task in enumerate(chosen)]
    tasks_path = output / "tasks.json"
    if tasks_path.exists() and read_json(tasks_path) != chosen:
        raise ValueError("Saved task list differs from current fixed selection")
    atomic_json(tasks_path, chosen)
    atomic_jsonl(output / "selection.jsonl", audit)
    selection = {
        **selection_stats(audit), "videos_without_proposals": no_proposal,
        "by_tool_domain": {
            domain: selection_stats([row for row in audit if row["tool_domain"] == domain])
            for domain in sorted({row["tool_domain"] for row in audit})
        },
    }
    selection.update(input_videos=len(predictions), annotation_videos=len(annotations),
                     selected_count=len(chosen),
                     selected_request_count=sum(len(required_stages(task)) for task in chosen))
    atomic_json(output / "selection_summary.json", selection)
    print("Selection:", json.dumps(selection["selection_counts"], ensure_ascii=False), flush=True)
    print("Classification:", json.dumps(selection["outcome_counts"], ensure_ascii=False), flush=True)
    print("Scoring scope: valid TP/TN only; low-IoU TP retained.", flush=True)
    print(f"Selected: {len(chosen)} / {len(tasks)} eligible proposals", flush=True)
    return chosen, audit, no_proposal


def stop_workers(processes) -> None:
    # Each worker owns a process group, including any vLLM engine subprocesses.
    for process, _ in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for process, _ in processes:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process, _ in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def launch(output: Path, config: dict, tasks: list[dict], devices: str) -> None:
    run_id = fingerprint(config)
    existing = 0
    expected_files = {result_path(output, task).name for task in tasks}
    actual_files = {path.name for path in (output / "results").glob("*.json")}
    if not actual_files <= expected_files:
        raise ValueError("Unexpected saved result files")
    expected_requests = {request_path(output, task, stage).name
                         for task in tasks for stage in required_stages(task)}
    if not {path.name for path in (output / "requests").glob("*.json")} <= expected_requests:
        raise ValueError("Unexpected saved judge request files")
    for task in tasks:
        path = result_path(output, task)
        if path.exists():
            checked_result(path, task, run_id)
            existing += 1
    if existing == len(tasks):
        return
    device_list = devices.split(",")
    worker = Path(__file__).with_name("text_rq_worker.py")
    processes, handles = [], []
    try:
        for rank, device in enumerate(device_list):
            atomic_json(output / f"progress_rank_{rank}.json", {"new_completed": 0})
            log_path = output / "logs" / f"rank_{rank}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=device, TOKENIZERS_PARALLELISM="false",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       VLLM_WORKER_MULTIPROC_METHOD="spawn", PYTHONHASHSEED="42")
            command = [sys.executable, str(worker), "--output", str(output),
                       "--rank", str(rank), "--world-size", str(len(device_list))]
            process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            processes.append((process, log_path))
        with tqdm(total=len(tasks), initial=existing, desc="TextRQ", unit="proposal") as bar:
            while True:
                states = [process.poll() for process, _ in processes]
                for state, (_, log_path) in zip(states, processes):
                    if state is not None and state != 0:
                        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
                        raise RuntimeError(f"TextRQ worker failed; {log_path}\n{tail}")
                completed = existing + sum(
                    read_json(output / f"progress_rank_{rank}.json")["new_completed"]
                    for rank in range(len(device_list))
                )
                bar.update(completed - bar.n)
                if all(state is not None for state in states):
                    break
                time.sleep(0.5)
    except BaseException:
        stop_workers(processes)
        raise
    finally:
        for handle in handles:
            handle.close()


def main() -> None:
    args = arguments()
    if sys.platform != "linux":
        raise RuntimeError("Run TextRQ on the Linux GPU server")
    config = configuration(args)
    if any(not value["text"] for value in config["prompts"].values()):
        raise ValueError("Judge prompt cannot be empty")
    output = Path(args.output)
    resolved = output.resolve()
    root = Path("../MSLoc_data/Qwen/text_rq").resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("--output must be a run subdirectory of ../MSLoc_data/Qwen/text_rq/")
    protected = [Path(config[key]["path"]).resolve() for key in ("predictions", "annotation", "model")]
    protected += [Path(prompt["path"]).resolve() for prompt in config["prompts"].values()]
    if any(path == resolved or path.is_relative_to(resolved) for path in protected):
        raise ValueError("Output must not contain inputs, prompts or model weights")
    # The lock lives outside the cleanable directory and is released by the OS.
    import fcntl

    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize(output, config, args)
        tasks, audit, no_proposal = prepare(output, config)
        if args.prepare_only:
            print(f"Prepared only: {output / 'tasks.json'}", flush=True)
            return
        started = time.monotonic()
        launch(output, config, tasks, args.devices)
        results = [checked_result(result_path(output, task), task, fingerprint(config))
                   for task in tasks]
        atomic_jsonl(output / "scores.jsonl", results)
        summary = summarize(results, audit, no_proposal)
        summary.update(run_id=fingerprint(config), selected_count=len(tasks))
        atomic_json(output / "summary.json", summary)
        atomic_jsonl(output / "iou_pairs.jsonl", iou_pairs(results))
        print("Analyzing TextRQ / IoU association (video bootstrap)...", flush=True)
        atomic_json(output / "iou_analysis.json",
                    association_analysis(results, config["bootstrap_repeats"], show_progress=True))
        atomic_json(output / "last_execution.json", {
            "devices": args.devices, "seconds": time.monotonic() - started,
            "selected_count": len(tasks),
        })
        report = build_report(results, audit, no_proposal, args.min_iou)
        report.update(run_id=fingerprint(config), source_run=output.as_posix(),
                      protocol=config["protocol"])
        write_report(output / "reports" / "default", report)
        print(f"Results: {output / 'scores.jsonl'}; {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
