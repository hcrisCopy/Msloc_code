"""Compare full-generation prechecks and select teacher-correct OPSD samples."""

import argparse
from pathlib import Path

from demo_utils import (data_identity, read_json, read_rows, require_root, sha256, write_json, write_rows)
from check_teacher import interval_iou


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-run", required=True)
    p.add_argument("--teacher-run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--min-iou", type=float, default=0.5)
    p.add_argument("--output", required=True, help="An unused JSON file under ../MSLoc_data/Qwen/demo/")
    args = p.parse_args()
    require_root()
    from demo_utils import OUTPUT_ROOT
    output = Path(args.output)
    if not output.resolve().is_relative_to(OUTPUT_ROOT.resolve()) or output.exists():
        raise ValueError("Gate output must be a new file under the demo output root.")
    if not 0 < args.min_iou <= 1:
        raise ValueError("--min-iou must be in (0,1].")
    source = data_identity(args.data)
    configs = [read_json(Path(folder) / "run.json") for folder in (args.student_run, args.teacher_run)]
    if [c["mode"] for c in configs] != ["student", "teacher"]:
        raise ValueError("Expected student and teacher inference runs.")
    for key in ("data", "source", "max_new_tokens", "versions"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Precheck mismatch: {key}")
    if configs[0]["data"] != source:
        raise ValueError("Precheck data changed.")
    records = [read_rows(Path(folder) / "predictions.jsonl") for folder in (args.student_run, args.teacher_run)]
    expected = {r["sample_id"] for r in read_rows(Path(args.data) / "opsd.jsonl")}
    for rows in records:
        if len(rows) != len(expected) or {r["sample_id"] for r in rows} != expected:
            raise ValueError("Precheck sample mismatch.")
    metrics, retained = [], []
    for role, rows in enumerate(records):
        fake = [r for r in rows if r["target_kind"] == "fake"]
        real = [r for r in rows if r["target_kind"] == "real"]
        if not fake or not real:
            raise ValueError("Both fake and real must be present.")
        scores = {r["sample_id"]: interval_iou(r["relative_segment"], r["target_relative"])
                  if r["status"] == "fake" else 0.0 for r in fake}
        metrics.append({"fake_iou": sum(scores.values()) / len(fake),
                        "fake_accuracy": sum(r["status"] == "fake" for r in fake) / len(fake),
                        "real_accuracy": sum(r["status"] == "real" for r in real) / len(real),
                        "invalid": sum(r["status"] not in ("fake", "real") for r in rows)})
        if role == 1:
            retained = [r["sample_id"] for r in rows if
                        (r["target_kind"] == "real" and r["status"] == "real") or
                        (r["target_kind"] == "fake" and scores[r["sample_id"]] >= args.min_iou)]
    student, teacher = metrics
    passed = (teacher["fake_iou"] > student["fake_iou"] and teacher["invalid"] <= student["invalid"]
              and all(teacher[k] >= student[k] for k in ("fake_accuracy", "real_accuracy")))
    selected = [r for r in read_rows(Path(args.data) / "opsd.jsonl") if r["sample_id"] in set(retained)]
    kinds = {r["sample_id"]: r["target_kind"] for r in records[1]}
    passed = passed and {kinds[r["sample_id"]] for r in selected} == {"fake", "real"}
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = output.with_suffix(".jsonl")
    if dataset.exists():
        raise FileExistsError(dataset)
    write_rows(dataset, selected)
    write_json(output, {"passed": passed, "student": configs[0], "teacher": configs[1],
               "metrics": {"student": student, "teacher": teacher}, "min_iou": args.min_iou,
               "retained": len(selected), "dataset": str(dataset), "dataset_sha256": sha256(dataset)})
    print(f"Teacher gate: passed={passed}; metrics={metrics}; retained={len(selected)}")
    if not passed:
        raise RuntimeError("Teacher gate failed. Review predictions; do not start OPSD with these weights/prompts.")


if __name__ == "__main__":
    main()
