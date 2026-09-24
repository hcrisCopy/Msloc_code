"""第二阶段共用的 proposal、片段与答案格式；参考 Trace 的 replay 构建和 evaluate_long 输入。"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path


EXPLANATION_LINE = r"Explanation: (?P<explanation>[^\r\n]+)\n"
ANSWER_PATTERN = re.compile(
    rf"\A{EXPLANATION_LINE}"
    r"Interval: \[(?P<start>\d+(?:\.\d+)?), (?P<end>\d+(?:\.\d+)?)\]\Z"
)
REAL_PATTERN = re.compile(
    rf"\A{EXPLANATION_LINE}"
    r"Real\Z"
)


def read_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"JSON 顶层必须是列表：{path}")
    if any(not isinstance(row, dict) or not row.get("video_path") for row in rows):
        raise ValueError(f"每条记录必须有 video_path：{path}")
    names = [row["video_path"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"video_path 重复：{path}")
    return rows


def proposal_segments(row: dict) -> list[tuple[float, float]]:
    inference = row.get("model_inference")
    if not isinstance(inference, dict):
        raise ValueError(f"缺少 model_inference：{row['video_path']}")
    segments = inference.get("segment")
    if not isinstance(segments, list):
        raise ValueError(f"model_inference.segment 必须是列表：{row['video_path']}")
    result = []
    for segment in segments:
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError(f"非法 proposal：{row['video_path']} {segment!r}")
        start, end = map(float, segment)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"非法 proposal 边界：{row['video_path']} {segment!r}")
        result.append((start, end))
    return result


def gt_segments(row: dict) -> list[tuple[float, float, dict]]:
    result = []
    for ann in row.get("annotations", []):
        segment = ann.get("segment")
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError(f"非法 GT 区间：{row['video_path']} {segment!r}")
        start, end = map(float, segment)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"非法 GT 边界：{row['video_path']} {segment!r}")
        result.append((start, end, ann))
    return result


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def explanation(ann: dict) -> list[str]:
    def first_text(key: str, field: str) -> str:
        items = ann.get(key) or []
        return " ".join(str(items[0].get(field, "")).split()) if items else ""

    object_text = first_text("obj_cot", "obj_caption")
    if not object_text:
        raise ValueError("正样本缺少 obj_cot[0].obj_caption")
    if "Round4" in str(ann.get("combine_dir", "")):
        return [object_text]
    start_text = first_text("bnd_cot_st", "bnd_caption")
    end_text = first_text("bnd_cot_ed", "bnd_caption")
    if not start_text or not end_text:
        return [object_text]
    return [start_text, object_text, end_text]


def number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") if value else "0"


def target_text(captions: list[str], relative: tuple[float, float]) -> str:
    if len(captions) not in (1, 3):
        raise ValueError("解释必须由一条或三条标注组成")
    parts = [caption.strip().rstrip(" .!?。") for caption in captions]
    if any(not part for part in parts):
        raise ValueError("解释标注不能为空")
    body = ". ".join(parts) + "."
    return f"Explanation: {body}\nInterval: [{number(relative[0])}, {number(relative[1])}]"


def clip_name(index: int, segment: tuple[float, float]) -> str:
    return f"proposal_{index:05d}_{segment[0]:.6f}_{segment[1]:.6f}.mp4"


def parse_answer(raw: str, duration: float) -> dict:
    text = raw.strip()
    real_match = REAL_PATTERN.fullmatch(text)
    if real_match is not None:
        return {"status": "real", "relative_segment": None, "explanation": real_match["explanation"]}
    match = ANSWER_PATTERN.fullmatch(text)
    if match is None:
        return {"status": "format_error", "relative_segment": None, "explanation": ""}
    start, end = float(match["start"]), float(match["end"])
    if not (0 <= start < end <= duration + 0.001):
        return {"status": "range_error", "relative_segment": None, "explanation": match["explanation"]}
    return {
        "status": "fake",
        "relative_segment": [start, end],
        "explanation": match["explanation"],
    }


def make_clip(video: Path, segment: tuple[float, float], output: Path, frames: int) -> None:
    """精确截取 proposal，均匀抽帧成短视频；已有片段供断点继续复用。"""
    if output.is_file():
        if output.stat().st_size == 0:
            raise RuntimeError(f"已有片段为空：{output}")
        return
    if not video.is_file():
        raise FileNotFoundError(video)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.mp4")
    duration = segment[1] - segment[0]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", number(segment[0]), "-i", str(video), "-t", number(duration),
        "-vf", f"fps={frames / duration:.8f}", "-frames:v", str(frames),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", str(temporary),
    ]
    subprocess.run(command, check=True)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"截取片段失败：{video} {segment}")
    temporary.replace(output)
