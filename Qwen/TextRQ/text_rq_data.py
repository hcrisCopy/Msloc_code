"""TextRQ data and scores, adapted from evaluate_RQ.py and Qwen/common.py."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import gt_segments, overlap, parse_answer


DIMENSIONS = (
    "fluency", "object_accuracy", "anomaly_type_accuracy", "detail_consistency",
)
ALL_DIMENSIONS = (*DIMENSIONS, "conclusion_consistency")
STAGE_FIELDS = {
    "fluency": ("fluency",),
    "consistency": ("conclusion_consistency",),
    "factual": DIMENSIONS[1:],
}


def interval(value, name: str) -> tuple[float, float]:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(x) not in (int, float) for x in value)):
        raise ValueError(f"{name}: expected two numeric endpoints")
    start, end = map(float, value)
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError(f"{name}: invalid interval {value!r}")
    return start, end


def temporal_iou(first, second) -> float:
    intersection = overlap(first, second)
    union = first[1] - first[0] + second[1] - second[0] - intersection
    return intersection / union


def cot_entries(annotation: dict, field: str, caption_field: str) -> list[dict]:
    """Preserve every annotated object, its classes, and its matching caption."""
    entries = annotation.get(field, [])
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        raise ValueError(f"{field}: expected an object or a list")
    output = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{field}: entry must be an object")
        required = ("obj", "bnd_class", "bnd_sub_class", caption_field)
        if any(not isinstance(entry.get(key), str) for key in required):
            raise ValueError(f"{field}: expected string fields {required}")
        if not entry[caption_field].strip():
            raise ValueError(f"{field}: empty caption")
        # Keep the annotation score out of the judge input; its meaning is undocumented.
        output.append({
            "subject": entry["obj"].strip(),
            "category": entry["bnd_class"].strip(),
            "subcategory": entry["bnd_sub_class"].strip(),
            "description": entry[caption_field].strip(),
        })
    return output


def reference_text(annotation: dict, proposal, target) -> dict:
    main = cot_entries(annotation, "obj_cot", "obj_caption")
    start = cot_entries(annotation, "bnd_cot_st", "bnd_caption")
    end = cot_entries(annotation, "bnd_cot_ed", "bnd_caption")
    if not main:
        raise ValueError("Matched fake annotation has no obj_cot")
    # Use the same boundary eligibility as common.explanation / prepare_sft.
    full_event = proposal[0] <= target[0] and proposal[1] >= target[1]
    round4 = "Round4" in annotation["combine_dir"]
    include_boundaries = full_event and not round4 and bool(start and end)
    return {
        "boundary_reference_available": include_boundaries,
        "start": start if include_boundaries else [],
        "main": main,
        "end": end if include_boundaries else [],
    }


def annotation_domain(annotation: dict) -> tuple[str, str]:
    """Support the two documented annotation schemas without guessing a domain."""
    field = "tool_domain" if "tool_domain" in annotation else "model"
    value = annotation[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or empty annotation {field}")
    return value.strip(), field


def video_domain(ground_truth: dict, targets) -> tuple[str, str, list[str]]:
    domains = sorted({annotation_domain(ann)[0] for _, _, ann in targets})
    for field in ("tool_domain", "model"):
        if field in ground_truth:
            value = ground_truth[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid video {field}")
            return value.strip(), f"video.{field}", domains
    if len(domains) == 1:
        return domains[0], "unique_annotation_domain", domains
    if domains:
        return "mixed", "multiple_annotation_domains", domains
    return "real", "real_without_domain", []


def extract_explanation(raw: str, parsed: dict) -> str:
    """Retain an exact Explanation line even if the conclusion is malformed."""
    if parsed["explanation"].strip():
        return parsed["explanation"].strip()
    text = raw.strip()
    prefix = "<think>\n\n</think>\n\n"
    if text.startswith(prefix):
        text = text[len(prefix):].strip()
    first_line = text.split("\n", 1)[0]
    marker = "Explanation: "
    return first_line[len(marker):].strip() if first_line.startswith(marker) else ""


def reference_target(hits: list, absolute) -> tuple[tuple | None, str]:
    """Do not invent a textual reference when several visible events are ambiguous."""
    if not hits:
        return None, "no_gt_in_proposal"
    if len(hits) == 1:
        return hits[0], "single_gt_in_proposal"
    if absolute is None:
        return None, "ambiguous_multiple_gt_without_valid_interval"
    values = [temporal_iou(absolute, hit[1:3]) for hit in hits]
    best = max(values)
    winners = [hit for hit, value in zip(hits, values)
               if math.isclose(value, best, rel_tol=0, abs_tol=1e-12)]
    if best <= 0 or len(winners) != 1:
        return None, "ambiguous_multiple_gt"
    return winners[0], "unique_max_prediction_iou"


def prepare_video(prediction: dict, ground_truth: dict, min_iou: float) -> dict:
    """Retain wrong decisions and low-IoU predictions; GT may be reused."""
    video = prediction["video_path"]
    source_type = ground_truth["type"]
    if source_type not in {"fake", "real"}:
        raise ValueError(f"{video}: unknown GT type {source_type!r}")
    if not isinstance(ground_truth.get("annotations"), list):
        raise ValueError(f"{video}: missing annotations list")
    targets = gt_segments(ground_truth)
    if (source_type == "real" and targets) or (source_type == "fake" and not targets):
        raise ValueError(f"{video}: GT type and anomaly intervals disagree")
    default_domain, default_domain_field, video_domains = video_domain(ground_truth, targets)
    proposals = prediction.get("proposal_results")
    if not isinstance(proposals, list):
        raise ValueError(f"{video}: expected Qwen/evaluate.py proposal_results")
    tasks, audit, seen = [], [], set()
    for row in proposals:
        index = row["proposal_index"]
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError(f"{video}: invalid or duplicate proposal index {index}")
        seen.add(index)
        task_id = f"{video}::{index}"
        if row["id"] != task_id or row["video_path"] != video:
            raise ValueError(f"{video}: inconsistent proposal identity")
        proposal = interval(row["proposal"], task_id)
        if not isinstance(row["raw_response"], str):
            raise ValueError(f"{task_id}: raw_response must be text")
        parsed = parse_answer(row["raw_response"], proposal[1] - proposal[0])
        explanation = extract_explanation(row["raw_response"], parsed)
        prediction_kind = (
            "fake" if parsed["status"] in {"fake", "range_error"}
            else "real" if parsed["status"] == "real" else None
        )
        absolute = ([proposal[0] + x for x in parsed["relative_segment"]]
                    if parsed["status"] == "fake" else None)
        hits = [(i, start, end, ann) for i, (start, end, ann) in enumerate(targets)
                if overlap(proposal, (start, end)) > 0]
        kind = "fake" if hits else "real"
        # Localization diagnostics use all GT, including an explicit zero overlap.
        iou = (max(temporal_iou(absolute, (start, end)) for start, end, _ in targets)
               if absolute is not None and targets else None)
        matched, reference_policy = reference_target(hits, absolute)
        domain, domain_field = default_domain, default_domain_field
        reference, matched_index, reference_iou = None, None, None
        if matched is not None:
            matched_index, start, end, annotation = matched
            reference = reference_text(annotation, proposal, (start, end))
            domain, domain_field = annotation_domain(annotation)
            reference_iou = temporal_iou(absolute, (start, end)) if absolute is not None else None
        valid = parsed["status"] in {"fake", "real"}
        outcome = ("TP" if kind == "fake" else "FP") if prediction_kind == "fake" else (
            ("FN" if kind == "fake" else "TN") if prediction_kind == "real" else "invalid"
        )
        if not valid:
            outcome = "invalid"
        info = {
            "id": task_id, "video_path": video, "proposal_index": index,
            "source_video_type": source_type, "kind": kind,
            "prediction_status": parsed["status"], "proposal": list(proposal),
            "prediction_kind": prediction_kind, "outcome": outcome,
            "classification_correct": prediction_kind == kind if prediction_kind else None,
            "segment": absolute, "iou": iou, "reference_iou": reference_iou,
            "matched_gt_index": matched_index,
            "gt_segment": list(matched[1:3]) if matched is not None else None,
            "overlapping_gt_count": len(hits), "reference_policy": reference_policy,
            "reference_available": reference is not None,
            "tool_domain": domain, "domain_field": domain_field,
            "video_domains": video_domains,
            "high_iou_fake": bool(valid and outcome == "TP" and iou is not None
                                  and iou > min_iou),
            "correct_real": bool(valid and outcome == "TN"),
            "selection": "eligible" if explanation else "missing_explanation",
        }
        audit.append(info)
        if explanation:
            tasks.append({
                **info, "explanation": explanation,
                "reference": reference, "raw_response": row["raw_response"],
            })
    return {"video_path": video, "tasks": tasks, "audit": audit,
            "no_proposal": not proposals}


def select_tasks(tasks: list[dict], count: int) -> list[dict]:
    """Freeze a seed-42 sample before looking at completed judge results."""
    tasks = sorted(tasks, key=lambda task: task["id"])
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate TextRQ task IDs")
    if count == -1:
        return tasks
    if not 1 <= count <= len(tasks):
        raise ValueError(f"--sample-count must be -1 or 1..{len(tasks)}")
    rng = random.Random(42)
    groups = {kind: [task for task in tasks if task["kind"] == kind]
              for kind in ("fake", "real")}
    if count >= 2 and all(groups.values()):
        fake_count = round(count * len(groups["fake"]) / len(tasks))
        fake_count = max(max(1, count - len(groups["real"])),
                         min(fake_count, len(groups["fake"]), count - 1))
        chosen = rng.sample(groups["fake"], fake_count)
        chosen += rng.sample(groups["real"], count - fake_count)
    else:
        chosen = rng.sample(tasks, count)
    return sorted(chosen, key=lambda task: task["id"])


def required_stages(task: dict) -> list[str]:
    stages = ["fluency"]
    if task["prediction_kind"] is not None:
        stages.append("consistency")
    if task["reference"] is not None:
        stages.append("factual")
    return stages


def score_schema(stage: str) -> dict:
    metric = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "reason": {"type": "string"},
        },
        "required": ["score", "reason"], "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {name: metric for name in STAGE_FIELDS[stage]},
        "required": list(STAGE_FIELDS[stage]), "additionalProperties": False,
    }


def validate_stage(value: dict, stage: str) -> None:
    if not isinstance(value, dict) or set(value) != set(STAGE_FIELDS[stage]):
        raise ValueError(f"Judge must return exactly the fields for {stage}")
    for name in STAGE_FIELDS[stage]:
        item = value[name]
        if not isinstance(item, dict) or set(item) != {"score", "reason"}:
            raise ValueError(f"{name}: expected score and reason")
        if type(item["score"]) is not int or not 1 <= item["score"] <= 5:
            raise ValueError(f"{name}: score must be an integer in 1..5")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError(f"{name}: missing scoring reason")


def validate_scores(value: dict, task: dict) -> float | None:
    if not isinstance(value, dict) or set(value) != set(ALL_DIMENSIONS):
        raise ValueError("Expected all four original dimensions and conclusion_consistency")
    stages = required_stages(task)
    for stage, names in STAGE_FIELDS.items():
        if stage in stages:
            validate_stage({name: value[name] for name in names}, stage)
        elif any(value[name] is not None for name in names):
            raise ValueError(f"Inapplicable {stage} scores must be null")
    # Consistency never enters the original four-dimension TextRQ mean.
    return (sum(value[name]["score"] for name in DIMENSIONS) / 4
            if task["reference"] is not None else None)


def format_reference(reference: dict) -> str:
    """Render every CoT entry as labelled text, following the OPSD teacher input style."""
    phases = ("start", "main", "end") if reference["boundary_reference_available"] else ("main",)
    scope = "Start, Main, End" if reference["boundary_reference_available"] else "Main only"
    sections = [f"Reference scope: {scope}."]
    for phase in phases:
        lines = [f"{phase.title()}:"]
        for index, entry in enumerate(reference[phase], 1):
            lines.extend([
                f"{index}. Subject: {entry['subject']}",
                f"   Category: {entry['category']}",
                f"   Subcategory: {entry['subcategory']}",
                f"   Description: {entry['description']}",
            ])
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def judge_messages(task: dict, prompts: dict, stage: str) -> list[dict]:
    if stage not in required_stages(task):
        raise ValueError(f"{task['id']}: inapplicable stage {stage}")
    payload = "Text A (Model Explanation):\n" + task["explanation"]
    if stage == "consistency":
        conclusion = "Real" if task["prediction_kind"] == "real" else "Fake (Interval)"
        payload += "\n\nPrediction conclusion:\n" + conclusion
    elif stage == "factual":
        payload += "\n\nText B (GT Reference Descriptions):\n" + format_reference(task["reference"])
    # All stages share the exact same common prefix. Metadata never reaches the judge.
    prompt = prompts["common"]["text"] + "\n\n" + prompts[stage]["text"]
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": payload}]
