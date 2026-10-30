"""Collect all three held-out evaluation reports into one CSV."""
import argparse
import csv
from pathlib import Path

from common import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    rows = []
    signatures = set()
    annotations = set()
    for variant in ("xclip", "mamba", "fusion"):
        report = read_json(root / variant / "test/metrics.json")
        if report["variant"] != variant:
            raise ValueError(f"Wrong variant in {variant}/test/metrics.json")
        signatures.add(report["feature_signature"])
        annotations.add(report["annotation_sha256"])
        for domain, metrics in report["metrics"].items():
            rows.append(dict(variant=variant, domain=domain, epoch=report["epoch"],
                selection=report.get("selection","validation_best_ema"),
                threshold=report["score_threshold"], parameters=report["parameters"],
                F1_Det=metrics["Loc_F1"], F1_Loc=metrics["Loc_IoU"], Det_Acc=metrics["Det_Acc"]))
    if len(signatures) != 1 or len(annotations) != 1:
        raise ValueError("The three evaluations use different features or annotation sets")
    path = root / "comparison.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["domain"] == "Total":
            print(f"{row['variant']:8s} F1_Det={100*row['F1_Det']:.2f} "
                  f"F1_Loc={100*row['F1_Loc']:.2f} Det_Acc={100*row['Det_Acc']:.2f}")
    print(path)


if __name__ == "__main__":
    main()
