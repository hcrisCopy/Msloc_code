"""OPSD 的 proposal 真值选择与教师提示词；与 SFT 使用同一单区间规则。"""

from __future__ import annotations

from common import explanation, gt_segments, number, overlap


def proposal_target(proposal: tuple[float, float], annotation: dict) -> dict:
    """选交集最长的 GT，并转换为 proposal 内的相对秒数；无交集即正常片段。"""
    hits = []
    for start, end, ann in gt_segments(annotation):
        shared = overlap(proposal, (start, end))
        if shared > 0:
            hits.append((shared, start, end, ann))
    if not hits:
        return {"kind": "real", "relative_segment": None, "overlapping_gt_count": 0}
    _, start, end, ann = max(hits, key=lambda hit: hit[0])
    clipped = (max(start, proposal[0]), min(end, proposal[1]))
    captions = explanation(ann)
    if clipped != (start, end) and len(captions) == 3:
        captions = [captions[1]]
    object_items = ann.get("obj_cot") or []
    manipulation_class = str(object_items[0].get("bnd_sub_class", "")).strip() if object_items else ""
    return {
        "kind": "fake",
        "relative_segment": [clipped[0] - proposal[0], clipped[1] - proposal[0]],
        "overlapping_gt_count": len(hits),
        "manipulation_class": manipulation_class,
        "annotation_hints": captions,
    }


def student_message(prompt: str, duration: float) -> str:
    return f"<video>{prompt}\n\nClip duration: {number(duration)} seconds."


def teacher_message(student: str, teacher_template: str, target: dict, duration: float) -> str:
    """异常教师沿用学生任务；正常教师用中性任务措辞，输出协议保持一致。"""
    if not teacher_template.strip():
        raise ValueError("教师提示词不能为空")
    for key in ("{privileged_fact}", "{continuation_examples}"):
        if teacher_template.count(key) != 1:
            raise ValueError(f"教师模板必须恰有一个 {key} 占位符")
    if target["kind"] == "fake":
        opening = student
        start, end = target["relative_segment"]
        fact = (
            "Verified ground truth for THIS clip: it contains a manipulation. "
            f"The true manipulated interval is [{number(start)}, {number(end)}] seconds "
            "relative to this clip's start. "
            f"The final line must be Interval: [{number(start)}, {number(end)}]. "
            "If a student's earlier Explanation calls this clip normal, "
            "ignore that mistaken conclusion. Describe only visible signs in the Explanation. "
        )
        if target["manipulation_class"]:
            fact += f"Annotation category: {target['manipulation_class']}. "
        fact += "Annotation points to check against the visible clip: " + " | ".join(target["annotation_hints"])
        fact += ". Use a point only when the video supports it; do not invent details from the annotation."
        examples = (
            "- Verified interval [10, 20], prefix 'Explanation: The reflection changes unexpectedly.\\n' "
            "-> continue 'Interval: [10, 20]'.\n"
            "- Verified interval [10, 20], prefix 'Explanation: The reflection changes unexpectedly.\\nInterval: [10, 1' "
            "-> continue '9]'; the completed end 19 is closer to 20 than 11.\n"
            "- Verified interval [4, 8], prefix 'Explanation: An object appears suddenly.\\nInterval: [4, ' "
            "-> continue '8]'.\n"
            "- Verified interval [3, 7], prefix 'Explanation: A person walks past the camera.\\n' "
            "-> continue 'Interval: [3, 7]' even if the earlier sentence missed the anomaly."
        )
    elif target["kind"] == "real":
        opening = (
            "<video>Describe ordinary visible events in this clip in one sentence, or in three sentences "
            "for the beginning, main event, and end when all three are visible. "
            "Write exactly two lines with no extra text. The first line is 'Explanation: <one or three sentences>'. "
            "The second line is 'Real'. "
            f"Clip duration: {number(duration)} seconds."
        )
        fact = (
            "Verified ground truth for THIS clip: it is normal. "
            "Earlier text may be incorrect; describe what is actually visible in one or three sentences. "
            "The final line must be Real."
        )
        examples = (
            "- Verified normal clip, prefix 'Explanation: A person moves across the room.\\n' "
            "-> continue 'Real'.\n"
            "- Verified normal clip, no answer prefix -> write 'Explanation: A person walks across the room.\\nReal' "
            "when that is what the video shows."
        )
    else:
        raise ValueError(f"未知 proposal 类型：{target['kind']}")
    supplement = teacher_template.strip().replace("{privileged_fact}", fact).replace("{continuation_examples}", examples)
    return opening + "\n\n" + supplement
