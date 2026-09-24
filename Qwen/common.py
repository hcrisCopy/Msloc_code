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
        # 拼接标注可能从 0 秒前约两帧开始；与非负 proposal 求交时才落入视频范围。
        # Trace/scripts/build_opd_grpo_replay.py 也保留原始 GT 边界再计算交集。
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise ValueError(f"非法 GT 边界：{row['video_path']} {segment!r}")
        result.append((start, end, ann))
    return result


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def first_cot_entry(ann: dict, key: str) -> dict:
    """CoT 标注兼容单个对象与对象列表；参考 Trace 的 first() 读取方式。"""
    entries = ann.get(key)
    if entries is None or entries == []:
        return {}
    if isinstance(entries, dict):
        return entries
    if isinstance(entries, list) and isinstance(entries[0], dict):
        return entries[0]
    raise ValueError(f"{key} 必须是对象或对象列表，实际为 {type(entries).__name__}")


def explanation(ann: dict) -> list[str]:
    def first_text(key: str, field: str) -> str:
        return " ".join(str(first_cot_entry(ann, key).get(field, "")).split())

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
    return f"proposal_{index:05d}_{segment[0]:.6f}_{segment[1]:.6f}_trace16_8_16.mp4"


def clip_timestamps_path(clip: Path) -> Path:
    return clip.with_suffix(".timestamps.json")


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
    """参考 Trace/trace/mm_utils.py 的 16/8/16 取帧，另存片段内真实时间戳。"""
    if frames != 40:
        raise ValueError("Qwen 输入固定为 40 帧")
    timestamps_path = clip_timestamps_path(output)
    if output.is_file() and timestamps_path.is_file():
        if output.stat().st_size == 0:
            raise RuntimeError(f"已有片段为空：{output}")
        from torchcodec.decoders import VideoDecoder

        if VideoDecoder(str(output)).metadata.num_frames != 40:
            raise ValueError(f"已有片段不是 40 帧：{output}")
        saved = json.loads(timestamps_path.read_text(encoding="utf-8"))
        if (saved["sampling"] != "trace16_8_16" or saved["proposal"] != list(segment)
                or len(saved["relative_milliseconds"]) != 40):
            raise ValueError(f"已有片段的采样元数据与 proposal 不一致：{timestamps_path}")
        return
    if not video.is_file():
        raise FileNotFoundError(video)
    import numpy as np
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video))
    total = decoder.metadata.num_frames
    fps = decoder.metadata.average_fps
    if total is None or total <= 0 or fps is None or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"视频帧数或帧率无效：{video}")
    start = max(0.0, segment[0])
    end = min((total - 1) / fps, segment[1])
    if end <= start:
        raise ValueError(f"proposal 超出视频有效时间范围：{video} {segment}")
    first = max(0, min(round(start * fps), total - 1))
    last = max(0, min(round(end * fps), total - 1))
    duration = end - start
    left_end = start + 0.2 * duration
    right_start = end - 0.2 * duration

    def region_indices(begin: float, finish: float, count: int) -> list[int]:
        # 与 Trace 一样，先把分区端点按原视频 FPS 映射到帧，再在分区内 linspace。
        begin_frame = max(0, min(round(begin * fps), total - 1))
        finish_frame = max(0, min(round(finish * fps), total - 1))
        return np.linspace(begin_frame, finish_frame, count, dtype=int).tolist()

    indices = (region_indices(start, left_end, 16)
               + region_indices(left_end, right_start, 8)
               + region_indices(right_start, end, 16))
    if indices[0] != first or indices[-1] != last or len(indices) != 40:
        raise RuntimeError(f"16/8/16 采样结果无效：{video} {segment}")
    # Trace 使用 idx/fps - window_start；整数毫秒仅用于 Qwen 的 frames_indices/fps 接口。
    proposal_duration = segment[1] - segment[0]
    relative_milliseconds = [round(max(0.0, min(proposal_duration, index / fps - start)) * 1000)
                             for index in indices]
    batch = decoder.get_frames_at(indices=indices).data
    if batch.shape[0] != 40 or batch.shape[1] != 3:
        raise ValueError(f"解码帧形状错误：{video} {tuple(batch.shape)}")
    height, width = batch.shape[2:]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", f"{frames / proposal_duration:.8f}", "-i", "pipe:0", "-frames:v", str(frames),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", str(temporary),
    ]
    pixels = batch.permute(0, 2, 3, 1).contiguous().numpy().tobytes()
    subprocess.run(command, input=pixels, check=True)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"截取片段失败：{video} {segment}")
    if VideoDecoder(str(temporary)).metadata.num_frames != 40:
        raise RuntimeError(f"片段编码后不是 40 帧：{temporary}")
    timestamp_tmp = timestamps_path.with_name(timestamps_path.name + ".tmp")
    timestamp_tmp.write_text(json.dumps({
        "sampling": "trace16_8_16",
        "proposal": list(segment),
        "relative_milliseconds": relative_milliseconds,
        "source_frame_indices": indices,
        "source_fps": fps,
        "timebase_hz": 1000,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    timestamp_tmp.replace(timestamps_path)
