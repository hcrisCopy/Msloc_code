"""Atomic checkpoints and input identities for resumable TextRQ evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_jsonl(path: Path, rows) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_identity(path: Path) -> dict:
    """Hash small configs; record weight names, sizes and modification times."""
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    weights = sorted(path.glob("*.safetensors"))
    if not weights or (path / "adapter_config.json").exists():
        raise ValueError("Judge must be a downloaded full model with safetensors, not a LoRA")
    configuration = read_json(path / "config.json")
    if configuration["model_type"] not in {"qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}:
        raise ValueError("This runner supports Qwen3 / Qwen3.5 with thinking disabled")
    if configuration.get("quantization_config") is not None:
        raise ValueError("Use original BF16 weights; this runner does not enable quantization")
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        expected = set(read_json(index_path)["weight_map"].values())
        if not expected <= {weight.name for weight in weights}:
            raise FileNotFoundError("The judge checkpoint is missing indexed weight shards")
    return {
        "path": path.as_posix(), "model_type": configuration["model_type"],
        "configs": {file.name: sha256(file) for file in sorted(path.iterdir())
                    if file.is_file() and file.suffix in {".json", ".jinja", ".txt"}},
        "weights": [{"name": file.name, "bytes": file.stat().st_size,
                     "mtime_ns": file.stat().st_mtime_ns} for file in weights],
    }


def result_path(output: Path, task: dict) -> Path:
    return output / "results" / f"sample_{task['sample_index']:08d}.json"


def request_path(output: Path, task: dict, stage: str) -> Path:
    return output / "requests" / f"sample_{task['sample_index']:08d}_{stage}.json"


def checked_request(path: Path, task: dict, stage: str, run_id: str) -> dict:
    from text_rq_data import validate_stage

    row = read_json(path)
    if (row["run_id"] != run_id or row["task_sha256"] != fingerprint(task)
            or row["stage"] != stage or row["finish_reason"] != "stop"):
        raise ValueError(f"Judge request checkpoint mismatch: {path}")
    validate_stage(row["scores"], stage)
    if json.loads(row["judge_output"]) != row["scores"]:
        raise ValueError(f"Scores differ from raw judge output: {path}")
    return row


def assemble_result(output: Path, task: dict, run_id: str) -> dict:
    from text_rq_data import ALL_DIMENSIONS, required_stages, validate_scores

    scores = dict.fromkeys(ALL_DIMENSIONS)
    requests = {}
    for stage in required_stages(task):
        record = checked_request(request_path(output, task, stage), task, stage, run_id)
        requests[stage] = record
        scores.update(record["scores"])
    return {**task, "run_id": run_id, "task_sha256": fingerprint(task),
            "scores": scores, "score": validate_scores(scores, task),
            "judge_requests": requests}


def checked_result(path: Path, task: dict, run_id: str) -> dict:
    from text_rq_data import required_stages, validate_scores, validate_stage

    row = read_json(path)
    if (row["run_id"] != run_id or row["task_sha256"] != fingerprint(task)
            or any(row[key] != task[key] for key in task)):
        raise ValueError(f"Result does not belong to the current task: {path}")
    if row["score"] != validate_scores(row["scores"], task):
        raise ValueError(f"Inconsistent score: {path}")
    if set(row["judge_requests"]) != set(required_stages(task)):
        raise ValueError(f"Missing judge request records: {path}")
    for stage, record in row["judge_requests"].items():
        if (record["run_id"] != run_id or record["task_sha256"] != fingerprint(task)
                or record["stage"] != stage or record["finish_reason"] != "stop"):
            raise ValueError(f"Inconsistent request identity: {path}")
        validate_stage(record["scores"], stage)
        if (json.loads(record["judge_output"]) != record["scores"]
                or any(row["scores"][key] != value for key, value in record["scores"].items())):
            raise ValueError(f"Inconsistent request scores: {path}")
    return row
