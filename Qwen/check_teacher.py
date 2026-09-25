"""比较同批 proposal 的 SFT 学生与特权教师完整生成评测，给 OPSD 准入结论。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def proposal_rows(directory: Path) -> dict[str, dict]:
    rows = {}
    for video in load(directory / "predictions.json"):
        for row in video["proposal_results"]:
            if row["id"] in rows:
                raise ValueError(f"重复 proposal：{row['id']}")
            rows[row["id"]] = row
    return rows


def interval_iou(predicted: list[float] | None, target: list[float] | None) -> float:
    if predicted is None or target is None:
        return 0.0
    intersection = max(0.0, min(predicted[1], target[1]) - max(predicted[0], target[0]))
    union = max(predicted[1], target[1]) - min(predicted[0], target[0])
    return intersection / union if union > 0 else 0.0


def paired_stats(student_rows: dict[str, dict], teacher_rows: dict[str, dict]) -> tuple[dict, list[dict]]:
    if student_rows.keys() != teacher_rows.keys() or not student_rows:
        raise ValueError("师生预检的 proposal 集合不一致或为空")
    counts = {kind: 0 for kind in ("fake", "real")}
    correct = {model: {kind: 0 for kind in counts} for model in ("student", "teacher")}
    fake_iou = {model: 0.0 for model in correct}
    invalid = {model: 0 for model in correct}
    better = 0
    details = []
    for item_id in sorted(student_rows):
        student, teacher = student_rows[item_id], teacher_rows[item_id]
        if any(student[key] != teacher[key] for key in ("proposal", "target_kind", "target_relative")):
            raise ValueError(f"师生 proposal 真值不一致：{item_id}")
        kind = student["target_kind"]
        counts[kind] += 1
        scores = {}
        for name, row in (("student", student), ("teacher", teacher)):
            status = row["status"]
            correct[name][kind] += status == kind
            invalid[name] += status in ("format_error", "range_error")
            target_absolute = None
            if kind == "fake":
                target_absolute = [student["proposal"][0] + value for value in student["target_relative"]]
            scores[name] = interval_iou(row["segment"] if status == "fake" else None, target_absolute)
            if kind == "fake":
                fake_iou[name] += scores[name]
        if kind == "fake":
            teacher_better = teacher["status"] == "fake" and (
                student["status"] != "fake" or scores["teacher"] > scores["student"])
        else:
            teacher_better = teacher["status"] == "real" and student["status"] != "real"
        better += teacher_better
        details.append({"id": item_id, "target_kind": kind,
                        "student_status": student["status"], "teacher_status": teacher["status"],
                        "student_iou": scores["student"], "teacher_iou": scores["teacher"],
                        "teacher_strictly_better": teacher_better})
    if not counts["fake"] or not counts["real"]:
        raise ValueError("教师预检必须同时覆盖 fake 和 real proposal")
    metrics = {name: {
        "fake_detection_accuracy": correct[name]["fake"] / counts["fake"],
        "real_detection_accuracy": correct[name]["real"] / counts["real"],
        "fake_mean_iou": fake_iou[name] / counts["fake"],
        "invalid_count": invalid[name],
    } for name in correct}
    return {"counts": counts, "metrics": metrics, "teacher_strictly_better_count": better}, details


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-eval", required=True)
    parser.add_argument("--teacher-eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    student_dir, teacher_dir = Path(args.student_eval), Path(args.teacher_eval)
    output = Path(args.output)
    if not output.resolve().is_relative_to(Path("../MSLoc_data/Qwen").resolve()):
        raise ValueError("教师准入结果必须写入 ../MSLoc_data/Qwen/ 下")
    if output.exists() and not args.clean:
        raise FileExistsError(f"结果已存在：{output}；重算时指定 --clean")
    student_config, teacher_config = load(student_dir / "eval_config.json"), load(teacher_dir / "eval_config.json")
    keys = ("model", "adapter", "adapter_config_sha256", "adapter_weights_sha256",
            "proposals", "proposals_sha256", "annotation", "annotation_sha256",
            "video_root", "prompt_text", "frames", "max_new_tokens",
            "temperature", "top_p", "top_k", "repetition_penalty",
            "max_proposals", "selected_proposals", "selected_ids_sha256")
    differences = [key for key in keys if student_config[key] != teacher_config[key]]
    if differences or student_config["teacher_prompt_file"] is not None or teacher_config["teacher_prompt_file"] is None:
        raise ValueError(f"学生与教师评测配置不匹配：{differences}")
    paired, details = paired_stats(proposal_rows(student_dir), proposal_rows(teacher_dir))
    student_paired, teacher_paired = paired["metrics"]["student"], paired["metrics"]["teacher"]
    if student_config["max_proposals"] == -1:
        student_total = load(student_dir / "metrics.json")["Total"]
        teacher_total = load(teacher_dir / "metrics.json")["Total"]
        passed = (teacher_total["Loc_F1"] > student_total["Loc_F1"]
                  and teacher_total["Loc_IoU"] > student_total["Loc_IoU"]
                  and teacher_total["Det_Acc"] >= student_total["Det_Acc"]
                  and teacher_paired["fake_detection_accuracy"] >= student_paired["fake_detection_accuracy"]
                  and teacher_paired["real_detection_accuracy"] >= student_paired["real_detection_accuracy"]
                  and teacher_paired["invalid_count"] <= student_paired["invalid_count"])
        criterion = "全量整视频 Loc_F1/Loc_IoU 更高、Det_Acc 不低；逐 proposal fake/real 判定及无效回答不退化"
    else:
        student_total = teacher_total = None
        passed = (teacher_paired["fake_mean_iou"] > student_paired["fake_mean_iou"]
                  and teacher_paired["fake_detection_accuracy"] >= student_paired["fake_detection_accuracy"]
                  and teacher_paired["real_detection_accuracy"] >= student_paired["real_detection_accuracy"]
                  and teacher_paired["invalid_count"] <= student_paired["invalid_count"])
        criterion = "抽样逐 proposal：fake 平均 IoU 更高、fake/real 判定不退化、无效回答不增加"
    result = {
        "passed": passed,
        "criterion": criterion,
        "max_proposals": student_config["max_proposals"],
        "selected_proposals": student_config["selected_proposals"],
        "selected_ids_sha256": student_config["selected_ids_sha256"],
        "paired": paired,
        "student": {"Total": student_total, "proposal": student_paired},
        "teacher": {"Total": teacher_total, "proposal": teacher_paired},
        "student_eval": str(student_dir),
        "teacher_eval": str(teacher_dir),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    detail_path = output.with_name(output.stem + "_samples.jsonl")
    with detail_path.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("教师预检未通过；不能使用本次结果启动 OPSD")


if __name__ == "__main__":
    main()
