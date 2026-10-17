"""Shared demo utilities; all user paths are relative to the repository root."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QWEN = ROOT / "Qwen"
OUTPUT_ROOT = ROOT.parent / "MSLoc_data" / "Qwen" / "demo"
sys.path.insert(0, str(QWEN))


def require_root():
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from Msloc_code: python Qwen/demo/<script>.py ...")


def relative(path):
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def device_ids(value):
    ids = value.split(",")
    if not ids or any(not i.isdigit() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("--devices must be distinct GPU indices, e.g. 0 or 0,1,2,3,4,5,6,7")
    return ids


def environment(devices):
    env = os.environ.copy()
    visible = env.get("MSLOC_DEMO_RAY_VISIBLE", ",".join(device_ids(devices)))
    if len(visible.split(",")) != len(device_ids(devices)):
        raise ValueError("Ray allocation does not match the requested GPU count.")
    env.update(CUDA_VISIBLE_DEVICES=visible,
               PYTHONPATH=os.pathsep.join([str(QWEN), str(QWEN / "demo"), str(ROOT), env.get("PYTHONPATH", "")]),
               FORCE_QWENVL_VIDEO_READER="decord", FPS_MAX_FRAMES="40", VIDEO_MAX_TOKEN_NUM="128",
               WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false")
    return env


def check_versions(vllm=False):
    expected = {"ms-swift": "4.5.3", "transformers": "5.9.0", "trl": "0.27.0", "peft": "0.20.0"}
    if vllm:
        expected["vllm"] = "0.19.1"
    for package, wanted in expected.items():
        actual = importlib.metadata.version(package)
        if actual != wanted:
            raise RuntimeError(f"{package}: expected {wanted}, got {actual}. See demo/README.md.")
    return expected


def processor_identity(model):
    """Read effective local settings through Transformers, independent of export layout.

    Transformers 5.9 saves nested image/video settings in processor_config.json,
    while older exports use separate preprocessor files. Do not copy base assets
    or guess defaults to repair an incomplete export.
    """
    from transformers import AutoProcessor

    model = Path(model)
    nested_path = model / "processor_config.json"
    nested = read_json(nested_path) if nested_path.is_file() else {}
    for component, filename in (("image_processor", "preprocessor_config.json"),
                                ("video_processor", "video_preprocessor_config.json")):
        if component in nested:
            if not isinstance(nested[component], dict) or not nested[component]:
                raise ValueError(f"Invalid {component} in {nested_path}")
        elif not (model / filename).is_file():
            raise FileNotFoundError(
                f"Missing {component}: need {nested_path} containing '{component}' or {model / filename}.")

    processor = AutoProcessor.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    for component in ("tokenizer", "image_processor", "video_processor"):
        if getattr(processor, component, None) is None:
            raise ValueError(f"Loaded processor has no {component}: {model}")
    tokenizer = processor.tokenizer
    settings = {
        "processor": json.loads(processor.to_json_string()),
        "chat_template": processor.chat_template,
        "tokenizer": json.loads(tokenizer.backend_tokenizer.to_str()),
        "special_tokens": tokenizer.special_tokens_map,
        "tokenizer_settings": {key: getattr(tokenizer, key) for key in
                               ("padding_side", "truncation_side", "model_max_length")},
    }
    return {key: digest(value) for key, value in settings.items()}


def require_matching_processors(student, teacher):
    """Compare loaded settings rather than filenames or JSON whitespace."""
    for key, value in student["processor"].items():
        if value != teacher["processor"][key]:
            raise ValueError(f"Teacher/student processing differs: {key}")


def model_identity(model, adapter="none"):
    """Validate HF exports. Weight file stats detect changes without repeatedly hashing GBs."""
    model = Path(model)
    config = read_json(model / "config.json")
    if config.get("model_type") != "qwen3_5":
        raise ValueError("This demo targets dense Qwen3.5 (model_type=qwen3_5), not another architecture/MoE.")
    for name in ("tokenizer_config.json", "tokenizer.json"):
        if not (model / name).is_file():
            raise FileNotFoundError(f"HF export missing tokenizer file: {model / name}")
    weights = sorted(model.glob("*.safetensors"))
    if not weights or any(p.name.startswith("adapter_model") for p in weights):
        raise ValueError(f"--model needs complete HF model weights, not an adapter or raw training checkpoint: {model}")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        for name in set(read_json(index)["weight_map"].values()):
            if not (model / name).is_file():
                raise FileNotFoundError(model / name)
    elif not (model / "model.safetensors").is_file():
        raise FileNotFoundError(f"Sharded model needs model.safetensors.index.json: {model}")
    metadata = {p.name: sha256(p) for p in sorted(model.iterdir())
                if p.is_file() and p.suffix in (".json", ".jinja", ".txt")}
    result = {"model": relative(model), "metadata": metadata, "processor": processor_identity(model),
              "weights": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in weights}, "adapter": None}
    if adapter != "none":
        adapter = Path(adapter)
        cfg = read_json(adapter / "adapter_config.json")
        if cfg.get("peft_type") != "LORA":
            raise ValueError("Only LoRA adapters are supported.")
        result["adapter"] = {"path": relative(adapter), "config": sha256(adapter / "adapter_config.json"),
                             "weights": sha256(adapter / "adapter_model.safetensors")}
    return result


def source_identity():
    paths = list((QWEN / "demo").glob("*.py")) + [QWEN / name for name in
            ("common.py", "opsd_common.py", "check_teacher.py", "trace_video_template.py",
             "grpo_rewards.py", "grpo_logging.py", "trl_compat.py")]
    paths += [ROOT / "Trace/trace/opd_grpo.py", ROOT / "Trace/trace/text_explanation_reward.py"]
    return {relative(p): sha256(p) for p in sorted(paths)}


def prepare_output(output, config, resume, clean, protected=()):
    output = Path(output)
    resolved = output.resolve()
    allowed = OUTPUT_ROOT.resolve()
    if resolved == allowed or not resolved.is_relative_to(allowed):
        raise ValueError("Output must be a subdirectory of ../MSLoc_data/Qwen/demo/.")
    for item in protected:
        candidate = Path(item).resolve()
        if candidate == resolved or candidate.is_relative_to(resolved) or resolved.is_relative_to(candidate):
            raise ValueError(f"Output would overwrite an input: {item}")
    if clean and resume != "none":
        raise ValueError("--clean and --resume auto cannot be combined.")
    if clean and output.exists():
        # The resolved output and every protected input have been checked above.
        shutil.rmtree(output)
    if resume == "auto":
        if read_json(output / "run.json") != config:
            raise ValueError("Resume configuration/inputs changed; use a new output directory.")
    elif output.exists():
        raise FileExistsError(f"{output}: choose another output or explicitly use --clean.")
    else:
        output.mkdir(parents=True)
        write_json(output / "run.json", config)
    return output


def latest_checkpoint(output):
    candidates = [p for p in Path(output).glob("checkpoint-*")
                  if p.is_dir() and p.name[11:].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {output}; restart with --resume none --clean.")
    return max(candidates, key=lambda p: int(p.name[11:]))


def data_identity(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("complete") is not True:
        raise ValueError("Demo preparation is incomplete.")
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Prepared file changed: {name}")
    for item in manifest["media"]:
        clip = Path(item["clip"])
        if clip.stat().st_size != item["size"] or clip.stat().st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"Prepared video changed: {clip}")
        if sha256(clip.with_suffix(".timestamps.json")) != item["timestamps_sha256"]:
            raise ValueError(f"Timestamps changed: {clip}")
    return digest(manifest)
