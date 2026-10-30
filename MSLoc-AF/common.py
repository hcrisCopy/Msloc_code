"""Shared data and time-coordinate contract for the isolated MSLoc-AF experiments."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

VARIANTS = {"xclip": "xclip", "mamba": "mamba", "fusion": "fusion"}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_id(video_path):
    return hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:24]


def load_records(path):
    records = read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a nonempty annotation list: {path}")
    seen = set()
    for record in records:
        video = record["video_path"]
        if video in seen:
            raise ValueError(f"Duplicate video_path: {video}")
        seen.add(video)
        if record["type"] not in {"real", "fake"}:
            raise ValueError(f"Unknown video type: {video}")
        duration = float(record["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid duration: {video}")
    return records


def fake_segments(record):
    """Keep the released PG convention, including whole-fake fallback."""
    if record["type"] == "real":
        return []
    duration = float(record["duration"])
    result = []
    for ann in record.get("annotations", []):
        if "segment" not in ann:
            continue
        if len(ann["segment"]) != 2:
            raise ValueError(f"Invalid segment in {record['video_path']}")
        start, end = map(float, ann["segment"])
        if not math.isfinite(start + end) or end <= start:
            raise ValueError(f"Invalid segment in {record['video_path']}: {ann['segment']}")
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            raise ValueError(f"Segment outside video: {record['video_path']}")
        result.append([start, end])
    return sorted(result) if result else [[0.0, duration]]


def make_windows(duration, window_seconds, stride_seconds):
    if not (0 < stride_seconds <= window_seconds):
        raise ValueError("Require 0 < stride_seconds <= window_seconds")
    return [(i * stride_seconds, min(i * stride_seconds + window_seconds, duration))
            for i in range(math.ceil(duration / stride_seconds))]


def sample_indices(start, end, fps, frame_count, samples):
    """Exactly mirror PG's inclusive-endpoint sampling, including short-tail padding.

    Frame array index 0 is frame_1.jpg. This is intentionally not linspace:
    changing the released PG sampler would change the feature experiment.
    """
    first = int(start * fps)
    last = min(int(end * fps), frame_count - 1)
    if first < 0 or first >= frame_count or last < first or samples <= 0:
        raise ValueError("Cannot sample this window; check duration, FPS and extracted frames")
    if last - first + 1 >= samples:
        step = max(1, (last - first) // samples)
        indices = list(range(first, last + 1, step))[:samples]
    else:
        indices = list(range(first, last + 1))
        indices += [indices[-1]] * (samples - len(indices))
    return indices


def seconds_to_grid(segments, stride_seconds, window_seconds):
    """Feature i has nominal center i * stride + window / 2, even at the tail."""
    values = np.asarray(segments, dtype=np.float32).reshape(-1, 2)
    return (values - window_seconds / 2.0) / stride_seconds


def grid_to_seconds(segments, stride_seconds, window_seconds, duration):
    values = np.asarray(segments, dtype=np.float32).reshape(-1, 2)
    return np.clip(values * stride_seconds + window_seconds / 2.0, 0, duration)


def check_manifest(manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or incomplete feature manifest")
    spec = manifest["spec"]
    if spec["frame_fps"] <= 0 or spec["frames_per_window"] <= 0:
        raise ValueError("Invalid FPS/frame count")
    if not 0 < spec["stride_seconds"] <= spec["window_seconds"]:
        raise ValueError("Invalid window/stride")
    if spec["aggregation"] != "legacy_mamba_sequence_mean_then_flatten_no_global":
        raise ValueError("Unexpected aggregation contract")
    if not manifest["videos"]:
        raise ValueError("Empty feature cache")


def feature_signature(manifest):
    """Reject mixing caches made from different checkpoints, timing or code."""
    check_manifest(manifest)
    spec = dict(manifest["spec"])
    # Absolute paths are informational; content hashes define identity.
    spec.pop("checkpoint_path", None)
    spec.pop("xclip_weights_path", None)
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def validate_entry(root, manifest, record):
    entry = manifest["videos"].get(record["video_path"])
    if entry is None:
        raise ValueError(f"Missing cached video: {record['video_path']}")
    if not math.isclose(float(record["duration"]), entry["duration"], abs_tol=1e-5):
        raise ValueError(f"Duration mismatch: {record['video_path']}")
    spec = manifest["spec"]
    expected = len(make_windows(entry["duration"], spec["window_seconds"], spec["stride_seconds"]))
    if entry["num_windows"] != expected:
        raise ValueError(f"Window count mismatch: {record['video_path']}")
    for name in ("xclip", "mamba"):
        path = Path(root) / name / (entry["feature_id"] + ".npy")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (expected, spec["feature_dim"]):
            raise ValueError(f"Feature shape mismatch: {path}: {array.shape}")
        if array.dtype != np.dtype(spec["storage_dtype"]):
            raise ValueError(f"Feature dtype mismatch: {path}")
    return entry


def load_features(root, entry, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    def load(name):
        path = Path(root) / name / (entry["feature_id"] + ".npy")
        return np.load(path, allow_pickle=False).astype(np.float32)
    features = (np.concatenate([load("xclip"), load("mamba")], axis=1)
                if variant == "fusion" else load(variant))
    if not np.isfinite(features).all():
        raise ValueError(f"Nonfinite features: {entry['feature_id']}")
    return features
