"""在同一批 SFT 评测片段上比较贪心解码与 Qwen3.5 官方采样设置。

参考 Qwen/Qwen3.5-4B 模型卡的 non-thinking 参数，以及正式评测 evaluate.py
使用的 ms-swift TransformersEngine、40 帧模板和答案解析方式。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm


QWEN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QWEN_DIR))

from common import clip_timestamps_path, parse_answer  # noqa: E402
from opsd_common import student_message  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT 重复输出的独立小实验")
    parser.add_argument("--source-eval", required=True, help="已有 SFT 评测目录，需包含 clips/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", required=True, help="例如 0 或 0,1,2,3,4,5,6,7")
    parser.add_argument("--format-errors", type=int, required=True)
    parser.add_argument("--positive-controls", type=int, required=True)
    parser.add_argument("--negative-controls", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--resume", choices=["none", "auto"], required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def source_data(args: argparse.Namespace) -> tuple[dict, list[dict], dict]:
    source = Path(args.source_eval)
    config_path = source / "eval_config.json"
    predictions_path = source / "predictions.json"
    if not config_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"缺少 SFT 评测配置或逐条结果：{source}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["teacher_prompt_file"] is not None or config["adapter"] == "none":
        raise ValueError("实验输入必须是学生 SFT LoRA 评测结果")
    if config["frames"] != 40 or config["sampling"] != "trace16_8_16":
        raise ValueError("实验只复用 16/8/16 的 40 帧片段")
    model = Path(config["model"])
    adapter = Path(config["adapter"])
    if not model.is_dir():
        raise FileNotFoundError(model)
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    if file_sha256(adapter / "adapter_config.json") != config["adapter_config_sha256"]:
        raise ValueError("LoRA 配置已不同于原 SFT 评测")
    if file_sha256(adapter / "adapter_model.safetensors") != config["adapter_weights_sha256"]:
        raise ValueError("LoRA 权重已不同于原 SFT 评测")
    if Path(config["prompt_file"]).read_text(encoding="utf-8") != config["prompt_text"]:
        raise ValueError("学生提示词已不同于原 SFT 评测")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    rows = [item for video in predictions for item in video["proposal_results"]]
    ids = [item["id"] for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("原评测中 proposal ID 重复")
    return config, rows, {
        "eval_config_sha256": file_sha256(config_path),
        "predictions_sha256": file_sha256(predictions_path),
    }


def select_rows(args: argparse.Namespace, rows: list[dict]) -> list[dict]:
    groups = [
        ("format_error", args.format_errors,
         [row for row in rows if row["status"] == "format_error"]),
        ("positive_control", args.positive_controls,
         [row for row in rows if row["status"] == "fake" and row["target_kind"] == "fake"]),
        ("negative_control", args.negative_controls,
         [row for row in rows if row["status"] == "fake" and row["target_kind"] == "real"]),
    ]
    rng = random.Random(42)
    selected = []
    for name, count, candidates in groups:
        if count < 0 or count > len(candidates):
            raise ValueError(f"{name} 请求 {count} 条，可选 {len(candidates)} 条")
        ordered = sorted(candidates, key=lambda row: row["id"])
        if name == "format_error" and count:
            # 确保包含最明显的循环重复案例，其余仍按固定种子抽样。
            worst = max(ordered, key=lambda row: max_eightgram_count(row["raw_response"]))
            chosen = [worst] + rng.sample([row for row in ordered if row["id"] != worst["id"]], count - 1)
        else:
            chosen = rng.sample(ordered, count)
        selected.extend({"group": name, **row} for row in chosen)
    if not selected:
        raise ValueError("至少选择一条样本")
    return selected


def validate_paths(args: argparse.Namespace, config: dict) -> list[str]:
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens 必须大于 0")
    devices = [item.strip() for item in args.devices.split(",")]
    if not devices or any(not item.isdigit() for item in devices) or len(set(devices)) != len(devices):
        raise ValueError("--devices 应为不重复的 GPU 编号")
    output = Path(args.output).resolve()
    data_root = Path("../MSLoc_data/Qwen").resolve()
    if output == data_root or not output.is_relative_to(data_root):
        raise ValueError("实验输出必须位于 ../MSLoc_data/Qwen/ 下")
    protected = [Path(args.source_eval), Path(config["model"]), Path(config["adapter"]),
                 Path(config["prompt_file"])]
    if any(path.resolve() == output or path.resolve().is_relative_to(output) for path in protected):
        raise ValueError("输出目录不能覆盖原评测、模型、权重或提示词")
    if args.clean and args.resume != "none":
        raise ValueError("--clean 与 --resume auto 不能同时使用")
    return devices


def launch(args: argparse.Namespace) -> None:
    config, rows, source_hashes = source_data(args)
    if args.max_new_tokens != config["max_new_tokens"]:
        raise ValueError("为只比较解码方式，max-new-tokens 必须与原 SFT 评测一致")
    selected = select_rows(args, rows)
    devices = validate_paths(args, config)
    output = Path(args.output)
    if args.clean and output.exists():
        shutil.rmtree(output)
    if args.resume == "none" and output.exists():
        raise FileExistsError(f"输出目录已存在：{output}；使用 --resume auto 或 --clean")
    if args.resume == "auto" and not output.exists():
        raise FileNotFoundError(f"没有可继续的实验目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_eval": args.source_eval,
        **source_hashes,
        "selected_ids": [row["id"] for row in selected],
        "model": config["model"],
        "adapter": config["adapter"],
        "devices": args.devices,
        "max_new_tokens": args.max_new_tokens,
        "settings": {
            "greedy": {"temperature": 0},
            "qwen_sampling": {"temperature": 0.7, "top_p": 0.8,
                              "top_k": 20, "repetition_penalty": 1.0},
        },
        "seed": 42,
    }
    manifest_path = output / "experiment_config.json"
    if args.resume == "auto":
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("续跑配置与已有实验不同")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(devices),
        "FORCE_QWENVL_VIDEO_READER": "torchcodec",
        "FPS_MAX_FRAMES": "40",
        "VIDEO_MAX_TOKEN_NUM": "128",
        "TOKENIZERS_PARALLELISM": "false",
    })
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={len(devices)}", str(Path(__file__)),
               "--source-eval", args.source_eval, "--output", args.output,
               "--devices", args.devices,
               "--format-errors", str(args.format_errors),
               "--positive-controls", str(args.positive_controls),
               "--negative-controls", str(args.negative_controls),
               "--max-new-tokens", str(args.max_new_tokens),
               "--resume", args.resume, "--worker"]
    print("Experiment:", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def completed_rows(path: Path) -> dict[str, dict]:
    result = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["id"] in result:
                    raise ValueError(f"实验结果 ID 重复：{row['id']}")
                result[row["id"]] = row
    return result


def max_eightgram_count(raw: str) -> int:
    body = raw.split("Explanation:", 1)[-1]
    words = re.findall(r"[A-Za-z0-9]+", body.lower())
    if len(words) < 8:
        return 0
    return max(Counter(tuple(words[index:index + 8])
                       for index in range(len(words) - 7)).values())


def interval_iou(predicted: list[float] | None, target: list[float] | None) -> float | None:
    if target is None:
        return None
    if predicted is None:
        return 0.0
    intersection = max(0.0, min(predicted[1], target[1]) - max(predicted[0], target[0]))
    union = (predicted[1] - predicted[0]) + (target[1] - target[0]) - intersection
    return intersection / union


def run_worker(args: argparse.Namespace) -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    config, rows, _ = source_data(args)
    selected = select_rows(args, rows)
    output = Path(args.output)
    shard = output / f"rank_{rank}.jsonl"
    completed = completed_rows(shard)
    tasks = [row for index, row in enumerate(selected)
             if index % world == rank and row["id"] not in completed]
    if tasks:
        from swift import get_template
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
        from trace_video_template import register_trace_video_template

        register_trace_video_template()
        engine = TransformersEngine(
            config["model"], adapters=[config["adapter"]], adapter_names=["sft"],
            torch_dtype=torch.bfloat16, attn_impl="flash_attn",
            device_map={"": f"cuda:{int(os.environ['LOCAL_RANK'])}"},
        )
        engine.model.set_adapter("sft")
        engine.template = get_template(engine.processor, enable_thinking=False)
        settings = {
            "greedy": RequestConfig(max_tokens=args.max_new_tokens, temperature=0),
            "qwen_sampling": RequestConfig(max_tokens=args.max_new_tokens,
                                           temperature=0.7, top_p=0.8, top_k=20,
                                           repetition_penalty=1.0),
        }
        with shard.open("a", encoding="utf-8") as handle:
            for row in tqdm(tasks, desc=f"Repetition rank {rank}", unit="proposal", disable=rank != 0):
                clip = Path(row["clip"])
                if not clip.is_file() or not clip_timestamps_path(clip).is_file():
                    raise FileNotFoundError(f"缺少原评测生成的片段或时间戳：{clip}")
                duration = row["proposal"][1] - row["proposal"][0]
                request = InferRequest(
                    messages=[{"role": "user", "content": student_message(config["prompt_text"].strip(), duration)}],
                    videos=[str(clip)], chat_template_kwargs={"nframes": 40},
                )
                alternatives = {}
                for name, request_config in settings.items():
                    # 每条样本固定随机状态，与使用几张 GPU 无关。
                    seed = 42 + int.from_bytes(hashlib.sha256(row["id"].encode()).digest()[:4], "big")
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    response = engine.infer([request], request_config=request_config)[0]
                    choice = response.choices[0]
                    raw = "" if choice.message.content is None else str(choice.message.content)
                    parsed = parse_answer(raw, duration)
                    alternatives[name] = {
                        "raw_response": raw,
                        "status": parsed["status"],
                        "relative_segment": parsed["relative_segment"],
                        "label_correct": parsed["status"] == row["target_kind"],
                        "interval_iou": interval_iou(parsed["relative_segment"], row["target_relative"]),
                        "finish_reason": choice.finish_reason,
                        "max_eightgram_count": max_eightgram_count(raw),
                    }
                record = {
                    "id": row["id"], "group": row["group"],
                    "target_kind": row["target_kind"], "target_relative": row["target_relative"],
                    "proposal": row["proposal"],
                    "saved_status": row["status"], "saved_raw_response": row["raw_response"],
                    "greedy": alternatives["greedy"],
                    "qwen_sampling": alternatives["qwen_sampling"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    dist.barrier()
    if rank == 0:
        merge(selected, output, world)
    dist.barrier()
    dist.destroy_process_group()


def merge(selected: list[dict], output: Path, world: int) -> None:
    results = {}
    for rank in range(world):
        for task_id, row in completed_rows(output / f"rank_{rank}.jsonl").items():
            if task_id in results:
                raise ValueError(f"实验结果 ID 重复：{task_id}")
            results[task_id] = row
    expected = {row["id"] for row in selected}
    if set(results) != expected:
        raise ValueError(f"实验不完整：期望 {len(expected)} 条，实际 {len(results)} 条")
    ordered = [results[row["id"]] for row in selected]
    with (output / "comparisons.jsonl").open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"samples": len(ordered), "groups": dict(Counter(row["group"] for row in ordered)),
               "greedy_matches_saved": sum(row["greedy"]["raw_response"] == row["saved_raw_response"]
                                           for row in ordered), "settings": {}}
    for name in ("greedy", "qwen_sampling"):
        fake_ious = [row[name]["interval_iou"] for row in ordered if row["target_kind"] == "fake"]
        summary["settings"][name] = {
            "status": dict(Counter(row[name]["status"] for row in ordered)),
            "correct_label": sum(row[name]["label_correct"] for row in ordered),
            "mean_interval_iou_on_fake_targets": sum(fake_ious) / len(fake_ious) if fake_ious else None,
            "repeated_8gram_at_least_3": sum(row[name]["max_eightgram_count"] >= 3 for row in ordered),
            "stopped_by_length": sum(row[name]["finish_reason"] == "length" for row in ordered),
            "format_error_group": {
                "valid_output": sum(row["group"] == "format_error" and row[name]["status"] in {"fake", "real"}
                                    for row in ordered),
                "repeated_8gram_at_least_3": sum(row["group"] == "format_error"
                                                  and row[name]["max_eightgram_count"] >= 3 for row in ordered),
            },
        }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    args = arguments()
    if args.worker:
        run_worker(args)
    else:
        launch(args)
