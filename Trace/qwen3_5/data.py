"""Qwen3.5 SFT 的样本构造、视频采样和严格输出协议。

数据定义沿用现有 TRACE ref2：每个 DeMamba proposal 是一个训练样本，
与 GT 有交集时学习 proposal 内的相对边界，否则学习 real/no-forgery。
视频采样复现 ``trace.mm_utils.process_video_ref_split`` 的 16+8+16 策略，
并把原始 RGB 帧逐帧交给 Qwen3.5 官方图像处理器，随后再进入 DAM/EAM。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import decord
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


SYSTEM_PROMPT = (
    "You are a video forgery localization system. Analyze only the candidate clip represented "
    "by the visual prefix. Return exactly one JSON object without Markdown or extra text. "
    "All times are seconds relative to the beginning of the candidate clip."
)

USER_PROMPT_TEMPLATE = (
    "The candidate clip duration is {duration:.6f} seconds. Localize every forged event and "
    "explain its visible evidence. Every segment boundary must be in seconds and satisfy "
    "0 <= start < end <= {duration:.6f}. Return exactly two keys: "
    '"label" ("fake" or "real") and "events". Each fake event must contain exactly '
    '"type" ("temporal" or "spatio-temporal"), "segment" ([start,end]), and '
    '"explanation". For a real clip, events must be empty.'
)


@dataclass(frozen=True)
class ProposalEvent:
    segment: tuple[float, float]
    manipulation_type: str
    start_caption: str
    object_caption: str
    end_caption: str
    start_class: str
    object_class: str
    end_class: str

    @property
    def explanation(self) -> str:
        parts = (self.start_caption, self.object_caption, self.end_caption)
        return " ".join(part.strip() for part in parts if part.strip())


@dataclass(frozen=True)
class ProposalSample:
    sample_id: str
    video_path: str
    proposal: tuple[float, float]
    events: tuple[ProposalEvent, ...]

    @property
    def is_positive(self) -> bool:
        return bool(self.events)

    @property
    def target_segments(self) -> tuple[tuple[float, float], ...]:
        return tuple(event.segment for event in self.events)

    @property
    def closs_classes(self) -> tuple[str, str, str]:
        if not self.events:
            return ("Normal", "Normal", "Normal")
        event = self.events[0]
        return (
            event.start_class or "none",
            event.object_class or "none",
            event.end_class or "none",
        )


def _video_key(item: dict[str, Any]) -> str:
    value = item.get("video_path") or item.get("video") or item.get("image_id")
    if not value:
        raise ValueError(f"记录缺少 video_path/video/image_id：{item}")
    return str(value)


def _first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _annotation_event(annotation: dict[str, Any], segment: tuple[float, float]) -> ProposalEvent:
    obj = _first_dict(annotation.get("obj_cot"))
    if "Round4" in str(annotation.get("combine_dir", "")):
        event = ProposalEvent(
            segment=segment,
            manipulation_type="spatio-temporal",
            start_caption="",
            object_caption=str(obj.get("obj_caption", "")).strip(),
            end_caption="",
            start_class="none",
            object_class=str(obj.get("bnd_sub_class", "")).strip() or "none",
            end_class="none",
        )
    else:
        start = _first_dict(annotation.get("bnd_cot_st"))
        end = _first_dict(annotation.get("bnd_cot_ed"))
        event = ProposalEvent(
            segment=segment,
            manipulation_type="temporal",
            start_caption=str(start.get("bnd_caption", "")).strip(),
            object_caption=str(obj.get("obj_caption", "")).strip(),
            end_caption=str(end.get("bnd_caption", "")).strip(),
            start_class=str(start.get("bnd_class", "")).strip() or "none",
            object_class=str(obj.get("bnd_sub_class", "")).strip() or "none",
            end_class=str(end.get("bnd_class", "")).strip() or "none",
        )
    if not event.explanation:
        raise ValueError(f"GT 事件缺少解释：{annotation}")
    return event


def load_proposal_samples(
    annotation_path: str,
    proposal_path: str,
    video_root: str,
    max_samples: int = 0,
) -> list[ProposalSample]:
    """严格读取 GT 与第一阶段 proposal，并构造与 TRACE ref2 一致的样本。"""

    annotation_file = Path(annotation_path)
    proposal_file = Path(proposal_path)
    root = Path(video_root)
    if not annotation_file.is_file():
        raise FileNotFoundError(f"标注文件不存在：{annotation_file}")
    if not proposal_file.is_file():
        raise FileNotFoundError(f"proposal 文件不存在：{proposal_file}")
    if not root.is_dir():
        raise FileNotFoundError(f"视频目录不存在：{root}")

    annotations = json.loads(annotation_file.read_text(encoding="utf-8"))
    proposals = json.loads(proposal_file.read_text(encoding="utf-8"))
    if not isinstance(annotations, list) or not isinstance(proposals, list):
        raise TypeError("标注文件和 proposal 文件的根节点都必须是列表")

    gt_map: dict[str, dict[str, Any]] = {}
    for item in annotations:
        video_key = _video_key(item)
        if video_key in gt_map:
            raise ValueError(f"GT 中出现重复视频：{video_key}")
        gt_map[video_key] = item
    samples: list[ProposalSample] = []
    for proposal_item in proposals:
        video_key = _video_key(proposal_item)
        if video_key not in gt_map:
            raise KeyError(f"proposal 在 GT 中找不到同名视频：{video_key}")
        video_file = root / video_key
        if not video_file.is_file():
            raise FileNotFoundError(f"视频文件不存在：{video_file}")

        segments = proposal_item.get("model_inference", {}).get("segment")
        if not isinstance(segments, list):
            raise TypeError(f"model_inference.segment 必须是列表：{video_key}")
        for proposal_index, proposal in enumerate(segments):
            if not isinstance(proposal, list) or len(proposal) != 2:
                raise ValueError(f"非法 proposal：{video_key} -> {proposal}")
            win_start, win_end = float(proposal[0]), float(proposal[1])
            if not np.isfinite(win_start) or not np.isfinite(win_end):
                raise ValueError(f"proposal 时间必须是有限数值：{video_key} -> {proposal}")
            if win_start < 0 or win_end <= win_start:
                raise ValueError(f"proposal 必须满足 0 <= start < end：{video_key} -> {proposal}")

            events: list[ProposalEvent] = []
            for annotation in gt_map[video_key].get("annotations", []):
                segment = annotation.get("segment")
                if not isinstance(segment, list) or len(segment) != 2:
                    raise ValueError(f"GT segment 非法：{video_key} -> {segment}")
                gt_start, gt_end = float(segment[0]), float(segment[1])
                if not np.isfinite(gt_start) or not np.isfinite(gt_end):
                    raise ValueError(f"GT segment 时间必须是有限数值：{video_key} -> {segment}")
                # 与原 TRACE ref2 数据逻辑一致：GT 允许因标注换算产生轻微负起点，
                # 随后通过与非负 proposal 求交自然裁到有效视频时间范围。
                if gt_end <= gt_start:
                    raise ValueError(f"GT segment 必须满足 start < end：{video_key} -> {segment}")
                inter_start, inter_end = max(gt_start, win_start), min(gt_end, win_end)
                if inter_end > inter_start:
                    relative = (inter_start - win_start, inter_end - win_start)
                    events.append(_annotation_event(copy.deepcopy(annotation), relative))

            samples.append(
                ProposalSample(
                    sample_id=f"{video_key}::proposal-{proposal_index}",
                    video_path=str(video_file),
                    proposal=(win_start, win_end),
                    events=tuple(events),
                )
            )

    if max_samples < 0:
        raise ValueError("max_samples 不能为负数")
    return samples[:max_samples] if max_samples else samples


def sample_proposal_frames(
    video_path: str,
    proposal: tuple[float, float],
    bnd_frames: int,
    seg_frames: int,
    bnd_ratio: float,
) -> tuple[list[Image.Image], list[float]]:
    """按左边界/中间/右边界三段确定性采样，不允许坏视频静默回退。"""

    if bnd_frames <= 0 or seg_frames <= 0:
        raise ValueError("bnd_frames 和 seg_frames 必须为正数")
    if not 0.0 < bnd_ratio < 0.5:
        raise ValueError("bnd_ratio 必须在 (0, 0.5) 内")

    reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(reader)
    fps = float(reader.get_avg_fps())
    if total_frames <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"无法读取有效视频元数据：{video_path}")

    last_frame_time = (total_frames - 1) / fps
    video_duration = total_frames / fps
    start, end = float(proposal[0]), float(proposal[1])
    if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
        raise ValueError(f"proposal 时间非法：{video_path} -> {proposal}")
    if end > video_duration + 1e-6:
        raise ValueError(
            f"proposal 超出视频有效时间，不允许裁剪后继续训练："
            f"{video_path} -> {proposal}, video_duration={video_duration:.6f}"
        )

    duration = end - start
    split_1 = start + bnd_ratio * duration
    split_2 = end - bnd_ratio * duration

    def indices(left: float, right: float, count: int) -> np.ndarray:
        first = max(0, min(int(round(left * fps)), total_frames - 1))
        last = max(0, min(int(round(min(right, last_frame_time) * fps)), total_frames - 1))
        return np.linspace(first, last, count, dtype=np.int64)

    frame_ids = np.concatenate(
        [
            indices(start, split_1, bnd_frames),
            indices(split_1, split_2, seg_frames),
            indices(split_2, end, bnd_frames),
        ]
    )
    frames = reader.get_batch(frame_ids).asnumpy()
    timestamps = [(int(frame_id) / fps) - start for frame_id in frame_ids]
    return [Image.fromarray(frame).convert("RGB") for frame in frames], timestamps


def build_answer(sample: ProposalSample) -> str:
    payload = {
        "label": "fake" if sample.is_positive else "real",
        "events": [
            {
                "type": event.manipulation_type,
                "segment": [round(event.segment[0], 3), round(event.segment[1], 3)],
                "explanation": event.explanation,
            }
            for event in sample.events
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_messages(
    sample: ProposalSample,
    include_answer: bool,
) -> list[dict[str, Any]]:
    duration = sample.proposal[1] - sample.proposal[0]
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"proposal 时长必须为正有限数值：{sample.sample_id} -> {sample.proposal}")
    user_prompt = USER_PROMPT_TEMPLATE.format(duration=duration)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        },
    ]
    if include_answer:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": build_answer(sample)}]}
        )
    return messages


def parse_answer(text: str, proposal_duration: float) -> dict[str, Any]:
    """严格解析 Qwen 输出；格式错误直接暴露为异常。"""

    if not np.isfinite(proposal_duration) or proposal_duration <= 0:
        raise ValueError("proposal_duration 必须是正有限数值")
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or set(payload) != {"label", "events"}:
        raise ValueError("输出必须是且仅是包含 label/events 的 JSON 对象")
    if payload["label"] not in {"fake", "real"}:
        raise ValueError("label 必须是 fake 或 real")
    if not isinstance(payload["events"], list):
        raise ValueError("events 必须是列表")
    if payload["label"] == "real" and payload["events"]:
        raise ValueError("real 输出的 events 必须为空")
    if payload["label"] == "fake" and not payload["events"]:
        raise ValueError("fake 输出至少需要一个 event")

    parsed_segments: list[list[float]] = []
    explanations: list[str] = []
    manipulation_types: list[str] = []
    for event in payload["events"]:
        if not isinstance(event, dict) or set(event) != {"type", "segment", "explanation"}:
            raise ValueError("每个 event 必须且仅包含 type/segment/explanation")
        manipulation_type = event["type"]
        if manipulation_type not in {"temporal", "spatio-temporal"}:
            raise ValueError("event.type 必须是 temporal 或 spatio-temporal")
        segment = event["segment"]
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError("每个 segment 必须是 [start, end]")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in segment):
            raise ValueError("segment 的 start/end 必须是 JSON 数值")
        start, end = float(segment[0]), float(segment[1])
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("segment 时间必须为有限数值")
        if start < 0 or end <= start or end > proposal_duration + 1e-6:
            raise ValueError("segment 必须位于 proposal 相对时间范围内")
        explanation = event["explanation"]
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("fake event 的 explanation 必须是非空字符串")
        parsed_segments.append([start, end])
        explanations.append(explanation.strip())
        manipulation_types.append(manipulation_type)
    payload["segments"] = parsed_segments
    payload["explanations"] = explanations
    payload["manipulation_types"] = manipulation_types
    return payload


class ProposalDataset(Dataset):
    def __init__(self, samples: list[ProposalSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ProposalSample:
        return self.samples[index]
