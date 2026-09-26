"""按 Video-OPSD 的教师正确性准入思想筛选训练 proposal。"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from check_teacher import interval_iou


def teacher_results(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    results = {}
    for video in json.loads(path.read_text(encoding="utf-8")):
        for row in video["proposal_results"]:
            item_id = row["id"]
            if item_id in results:
                raise ValueError(f"教师评测重复 proposal：{item_id}")
            results[item_id] = row
    return results


def filter_samples(samples: list[str], audit: list[dict], teacher: dict[str, dict],
                   selected_ids: set[str], expected_teacher_ids: set[str],
                   min_iou: float) -> tuple[str, str, dict]:
    """教师从头回答正确才保留；fake 定位用 IoU 阈值适配连续区间答案。"""
    if not math.isfinite(min_iou) or not 0 < min_iou <= 1:
        raise ValueError("教师 fake 定位 IoU 阈值必须在 (0, 1] 内")
    if len(samples) != len(audit):
        raise ValueError("OPSD 训练数据与目标审计行数不一致")
    audit_ids = [row["id"] for row in audit]
    if len(audit_ids) != len(set(audit_ids)):
        raise ValueError("OPSD 目标审计有重复 proposal ID")
    if set(teacher) != expected_teacher_ids:
        raise ValueError("教师完整生成评测与预期 proposal 集合不一致")
    if not selected_ids or not selected_ids.issubset(set(audit_ids)):
        raise ValueError("训练抽样 ID 与 OPSD 目标审计不一致")

    kept_samples = []
    decisions = []
    totals = Counter()
    retained = Counter()
    reasons = Counter()
    for sample, target in zip(samples, audit):
        item_id = target["id"]
        if json.loads(sample)["videos"] != [target["clip"]]:
            raise ValueError(f"OPSD 训练数据与目标审计顺序不一致：{item_id}")
        if item_id not in selected_ids:
            continue
        result = teacher[item_id]
        if any(result[key] != target[key] for key in ("proposal", "target_kind", "target_relative")):
            raise ValueError(f"教师评测与 OPSD 训练目标不一致：{item_id}")
        kind, status = target["target_kind"], result["status"]
        if kind not in ("fake", "real") or status not in ("fake", "real", "format_error", "range_error"):
            raise ValueError(f"未知教师或真值类别：{item_id} {kind} {status}")
        if (status == "fake") != (result["segment"] is not None):
            raise ValueError(f"教师结果的状态与区间不一致：{item_id}")
        iou = None
        if kind == "fake":
            if target["target_relative"] is None:
                raise ValueError(f"fake proposal 缺少目标区间：{item_id}")
            if status == "fake":
                start = target["proposal"][0]
                absolute_target = [start + value for value in target["target_relative"]]
                iou = interval_iou(result["segment"], absolute_target)
                keep = iou >= min_iou
                reason = "correct_fake" if keep else "low_iou"
            else:
                keep = False
                reason = "predicted_real" if status == "real" else status
        else:
            if target["target_relative"] is not None:
                raise ValueError(f"real proposal 不应有目标区间：{item_id}")
            keep = status == "real"
            reason = "correct_real" if keep else "predicted_fake" if status == "fake" else status
        totals[kind] += 1
        retained[kind] += keep
        reasons[reason] += 1
        decisions.append({"id": item_id, "proposal": target["proposal"],
                          "target_kind": kind, "target_relative": target["target_relative"],
                          "teacher_status": status, "teacher_segment": result["segment"],
                          "teacher_iou": iou, "keep": keep, "reason": reason})
        if keep:
            kept_samples.append(sample)
    if not retained["fake"] or not retained["real"]:
        raise ValueError(f"教师筛选后必须仍有 fake 和 real proposal：{dict(retained)}")
    summary = {"min_fake_iou": min_iou, "selected": dict(totals), "retained": dict(retained),
               "excluded": {kind: totals[kind] - retained[kind] for kind in ("fake", "real")},
               "reasons": dict(reasons)}
    return ("".join(row + "\n" for row in kept_samples),
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions), summary)
