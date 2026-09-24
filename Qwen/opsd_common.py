"""OPSD 的 proposal 真值选择与教师提示词；与 SFT 使用同一单区间规则。"""

from __future__ import annotations

from common import explanation, first_cot_entry, gt_segments, number, overlap


def proposal_target(proposal: tuple[float, float], annotation: dict) -> dict:
    """按 Trace replay 标记 proposal；有交集时选最长交集作为单区间监督。"""
    source_type = annotation["type"]
    if source_type not in ("fake", "real"):
        raise ValueError(f"未知视频类型：{source_type}")
    hits = []
    all_gt = gt_segments(annotation)
    if source_type == "real" and all_gt:
        raise ValueError(f"真实视频包含伪造 GT：{annotation['video_path']}")
    for start, end, ann in all_gt:
        shared = overlap(proposal, (start, end))
        if shared > 0:
            hits.append((shared, start, end, ann))
    max_iou = max((overlap(proposal, (start, end)) /
                   (max(proposal[1], end) - min(proposal[0], start)) for start, end, _ in all_gt), default=0.0)
    nearest_gap = min((max(start - proposal[1], proposal[0] - end, 0.0)
                       for start, end, _ in all_gt), default=float("inf"))
    if not hits:
        # Trace/scripts/build_opd_grpo_replay.py：fake 视频未命中 GT 的 proposal 也作为负例保留。
        bucket = "near_hard_negative" if nearest_gap <= 1.0 else "real_false_positive"
        return {
            "kind": "real", "relative_segment": None, "overlapping_gt_count": 0,
            "source_video_type": source_type, "replay_bucket": bucket, "max_gt_iou": max_iou,
            "nearest_gt_gap_seconds": None if nearest_gap == float("inf") else nearest_gap,
        }
    _, start, end, ann = max(hits, key=lambda hit: hit[0])
    clipped = (max(start, proposal[0]), min(end, proposal[1]))
    captions = explanation(ann)
    if clipped != (start, end) and len(captions) == 3:
        captions = [captions[1]]
    sentence_count = len(captions)
    object_entry = first_cot_entry(ann, "obj_cot")
    object_name = str(object_entry.get("obj", "")).strip()
    object_class = str(object_entry.get("bnd_class", "")).strip()
    object_subclass = str(object_entry.get("bnd_sub_class", "")).strip()
    round4 = "Round4" in str(ann.get("combine_dir", ""))
    has_visible_boundaries = not round4 and clipped == (start, end) and sentence_count == 3
    start_class = str(first_cot_entry(ann, "bnd_cot_st").get("bnd_class", "")).strip() if has_visible_boundaries else ""
    end_class = str(first_cot_entry(ann, "bnd_cot_ed").get("bnd_class", "")).strip() if has_visible_boundaries else ""
    return {
        "kind": "fake",
        "relative_segment": [clipped[0] - proposal[0], clipped[1] - proposal[0]],
        "overlapping_gt_count": len(hits),
        "source_video_type": source_type,
        "replay_bucket": "positive" if max_iou >= 0.5 else "hard_positive",
        "max_gt_iou": max_iou,
        "nearest_gt_gap_seconds": 0.0,
        "object_name": object_name,
        "object_class": object_class,
        "manipulation_type": "spatio-temporal" if round4 else "temporal",
        "object_subclass": object_subclass,
        "start_class": start_class,
        "end_class": end_class,
        "explanation_sentence_count": sentence_count,
    }


def student_message(prompt: str, duration: float) -> str:
    return f"<video>{prompt}\n\nClip duration: {number(duration)} seconds."


def teacher_message(student: str, teacher_template: str, target: dict, mode: str) -> str:
    """预检与训练教师均沿用学生任务格式；训练前缀由训练器接入。"""
    if mode not in ("precheck", "opsd"):
        raise ValueError(f"未知教师模式：{mode}")
    fake_marker, real_marker = "=== FAKE ===", "=== REAL ==="
    if teacher_template.count(fake_marker) != 1 or teacher_template.count(real_marker) != 1:
        raise ValueError("教师提示词必须各有一个 FAKE 和 REAL 段落")
    before_fake, _, remaining = teacher_template.partition(fake_marker)
    fake_prompt, _, real_prompt = remaining.partition(real_marker)
    if before_fake.strip() or not fake_prompt.strip() or not real_prompt.strip():
        raise ValueError("教师提示词必须先写 FAKE 段落，再写 REAL 段落")
    opening = student
    if target["kind"] == "fake":
        start, end = target["relative_segment"]
        main_fields = ["the visible sign of forgery in the main event"]
        if target["object_name"]:
            main_fields.append(f"object: {target['object_name']}")
        if target["object_class"]:
            main_fields.append(f"object category: {target['object_class']}")
        if target["object_subclass"]:
            main_fields.append(f"object subcategory: {target['object_subclass']}")
        main_point = "; ".join(main_fields)
        if target["explanation_sentence_count"] == 3:
            start_point = "the visible start of the forgery"
            end_point = "the visible end of the forgery"
            if target["start_class"]:
                start_point += f"; start category: {target['start_class']}"
            if target["end_class"]:
                end_point += f"; end category: {target['end_class']}"
            explanation_clues = (
                f"<what is visible at the beginning> should include {start_point}. "
                f"<main visible event> should include {main_point}. "
                f"<what is visible at the end> should include {end_point}."
            )
        else:
            explanation_clues = f"<main visible event> should include {main_point}."
        fact = (
            "Verified label for this clip: fake. "
            f"Verified interval within this clip: [{number(start)}, {number(end)}] seconds from the clip start. "
            f"Forgery category: {target['manipulation_type']}. "
            f"Explanation guidance: {explanation_clues}"
        )
        instruction = fake_prompt.strip()
    elif target["kind"] == "real":
        fact = (
            "Verified label for this clip: real. "
            "Describe ordinary visible actions or scene changes in one or three sentences."
        )
        instruction = real_prompt.strip()
    else:
        raise ValueError(f"未知 proposal 类型：{target['kind']}")
    return opening + "\n\nVerified information for this clip:\n" + fact + "\n\n" + instruction
