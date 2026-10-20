"""Render saved TextRQ scores without inference.

Reuse text_rq_metrics aggregates and the legacy TextRQ per-domain layout.
Keep report settings separate from the immutable judge checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from text_rq_data import ALL_DIMENSIONS
from text_rq_metrics import score_views, selection_stats
from text_rq_storage import atomic_json, checked_result, fingerprint, read_json, result_path


GROUP_FIELDS = (
    "tool_domain", "outcome", "prediction_kind", "kind",
    "source_video_type", "reference_policy",
)
METRICS = ("RQ", *ALL_DIMENSIONS)


def metric_values(stats: dict) -> dict:
    return {
        "RQ": {"mean": stats["rq_mean"], "count": stats["rq_count"]},
        **stats["dimensions"],
    }


def build_report(results: list[dict], audit: list[dict], no_proposal: int,
                 min_iou: float, group_by: str = "tool_domain") -> dict:
    if group_by not in GROUP_FIELDS or not 0 <= min_iou < 1:
        raise ValueError("Invalid report grouping or IoU threshold")
    result_ids = [row["id"] for row in results]
    audit_ids = [row["id"] for row in audit]
    if (len(set(result_ids)) != len(result_ids) or len(set(audit_ids)) != len(audit_ids)
            or not set(result_ids) <= set(audit_ids)):
        raise ValueError("Duplicate IDs or scored proposals missing from selection audit")

    def with_threshold(row):
        return {**row, "high_iou_fake": bool(
            row["outcome"] == "TP" and row["prediction_status"] == "fake"
            and row["iou"] is not None and row["iou"] > min_iou
        )}

    def group(row):
        value = row[group_by]
        if value is None and group_by == "prediction_kind":
            return "unknown"
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid {group_by} in {row['id']}")
        return value

    # Derived report flags must never mutate saved tasks or judge results.
    results = [with_threshold(row) for row in results]
    audit = [with_threshold(row) for row in audit]
    groups = {}
    for name in sorted({group(row) for row in audit}):
        group_audit = [row for row in audit if group(row) == name]
        group_results = [row for row in results if group(row) == name]
        groups[name] = {"input": selection_stats(group_audit), **score_views(group_results)}
    macro = {}
    for metric in METRICS:
        values = {name: metric_values(views["all"])[metric]["mean"]
                  for name, views in groups.items()}
        available = {name: value for name, value in values.items() if value is not None}
        macro[metric] = {
            "mean": sum(available.values()) / len(available) if available else None,
            "group_count": len(available), "included_groups": list(available),
            "excluded_groups": [name for name, value in values.items() if value is None],
        }
    return {
        "schema": "text_rq_report_v1", "group_by": group_by, "min_iou": min_iou,
        "unit": "proposal", "positive_class": "fake",
        "videos_without_proposals": no_proposal,
        "input": selection_stats(audit),
        "overall": score_views(results), "groups": groups,
        "macro_group": macro,
    }


def format_report(report: dict) -> str:
    lines = [
        "TextRQ Results",
        f"Grouping: {report['group_by']}; unit: proposal; positive class: Fake",
        "TP/FP/FN/TN describe classification, not one-to-one localization matches.",
        "Scores: 1-5; RQ: mean of the four original dimensions.",
        "Conclusion consistency is separate. N/A means no applicable scores.",
        f"Videos without proposals: {report['videos_without_proposals']}",
    ]
    heading = "Domain" if report["group_by"] == "tool_domain" else "Group"

    def number(value):
        return "N/A" if value is None else f"{value:.4f}"

    def block(name, views, inputs):
        stats = views["all"]
        lines.extend([
            "", f"--- {heading}: {name} ---",
            f"Input proposals: {inputs['input_proposals']}; "
            f"scored proposals: {stats['count']}; scored videos: {stats['video_count']}",
            f"Input selection: {json.dumps(inputs['selection_counts'], sort_keys=True)}",
            f"{'Metric':<28} {'Mean':>10} {'Valid N':>10}",
        ])
        for key, value in metric_values(stats).items():
            lines.append(f"{key:<28} {number(value['mean']):>10} {value['count']:>10}")
        lines.append("Classification groups (input count includes unscored proposals):")
        lines.append(f"{'Group':<12} {'Input N':>8} {'Scored N':>9} "
                     f"{'RQ':>9} {'RQ N':>6} {'Fluency':>9} {'Consist':>9}")
        for key, subset in views["by_outcome"].items():
            values = metric_values(subset)
            lines.append(
                f"{key:<12} {inputs['outcome_counts'].get(key, 0):>8} {subset['count']:>9} "
                f"{number(subset['rq_mean']):>9} {subset['rq_count']:>6} "
                f"{number(values['fluency']['mean']):>9} "
                f"{number(values['conclusion_consistency']['mean']):>9}"
            )
        subset = views["high_iou_fake"]
        lines.append(f"TP & IoU > {report['min_iou']:g}: "
                     f"scored N={subset['count']}, RQ={number(subset['rq_mean'])}, "
                     f"RQ N={subset['rq_count']}")

    for name, views in report["groups"].items():
        block(name, views, views["input"])
    block("Overall_proposal (equal weight per scored proposal)",
          report["overall"], report["input"])
    macro_name = "Macro_domain" if report["group_by"] == "tool_domain" else "Macro_group"
    lines.extend(["", f"--- {macro_name} (equal weight per applicable group) ---",
                  f"{'Metric':<28} {'Mean':>10} {'Groups N':>10}"])
    for key, value in report["macro_group"].items():
        lines.append(f"{key:<28} {number(value['mean']):>10} {value['group_count']:>10}")
    # Preserve full precision in JSON; terminal output is rounded for readability.
    def compact(stats):
        return {"scored_proposals": stats["count"], "scored_videos": stats["video_count"],
                **{key: {"mean": round(value["mean"], 4) if value["mean"] is not None else None,
                         "valid_n": value["count"]}
                   for key, value in metric_values(stats).items()}}

    final = {"groups": {name: compact(views["all"]) for name, views in report["groups"].items()},
             "Overall_proposal": compact(report["overall"]["all"]),
             macro_name: {key: {"mean": round(value["mean"], 4) if value["mean"] is not None else None,
                               "groups_n": value["group_count"]}
                          for key, value in report["macro_group"].items()}}
    lines.extend(["", "Final Results Summary (All Groups):",
                  json.dumps(final, ensure_ascii=False, indent=2)])
    return "\n".join(lines) + "\n"


def report_csv(report: dict) -> str:
    buffer = io.StringIO(newline="")
    fields = ["scope", "group", "view", "proposals", "videos", "metric", "mean", "valid_n", "unit"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    sections = [("overall", "Overall_proposal", report["overall"])]
    sections += [("group", name, views) for name, views in report["groups"].items()]
    for scope, group, views in sections:
        for view, stats in views.items():
            if view == "input":
                continue
            subsets = {view: stats} if "dimensions" in stats else {
                f"{view}/{name}": subset for name, subset in stats.items()
            }
            for label, subset in subsets.items():
                for metric, value in metric_values(subset).items():
                    writer.writerow({
                        "scope": scope, "group": group, "view": label,
                        "proposals": subset["count"], "videos": subset["video_count"],
                        "metric": metric, "mean": value["mean"],
                        "valid_n": value["count"], "unit": "proposal",
                    })
    for metric, value in report["macro_group"].items():
        writer.writerow({"scope": "macro_group", "group": "", "view": "all",
                         "metric": metric, "mean": value["mean"],
                         "valid_n": value["group_count"], "unit": "group"})
    return buffer.getvalue()


def write_report(output: Path, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report_summary.json", report)
    text = format_report(report)
    for name, content in (("report.txt", text), ("report.csv", report_csv(report))):
        path = output / name
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    print(text, end="", flush=True)
    print(f"Saved report: {output / 'report_summary.json'}", flush=True)
    print(f"Saved table: {output / 'report.csv'}; terminal copy: {output / 'report.txt'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Completed TextRQ run directory")
    parser.add_argument("--output", required=True, type=Path, help="Report directory under RUN/reports/")
    parser.add_argument("--group-by", choices=GROUP_FIELDS, default="tool_domain")
    parser.add_argument("--min-iou", type=float, default=0.7)
    args = parser.parse_args()
    if Path.cwd().resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("Run from MSLoc_code/")
    if args.run.is_absolute() or args.output.is_absolute():
        raise ValueError("Use relative paths")
    reports_root = (args.run / "reports").resolve()
    if not args.output.resolve().is_relative_to(reports_root):
        raise ValueError("--output must be inside RUN/reports/")
    manifest = read_json(args.run / "run_config.json")
    if (manifest["config"]["protocol"] != "text_rq_v2"
            or manifest["run_id"] != fingerprint(manifest["config"])):
        raise ValueError("Invalid TextRQ v2 run manifest")
    tasks = read_json(args.run / "tasks.json")
    from tqdm import tqdm

    results = [checked_result(result_path(args.run, task), task, manifest["run_id"])
               for task in tqdm(tasks, desc="Read saved scores", unit="proposal")]
    with (args.run / "selection.jsonl").open(encoding="utf-8") as handle:
        audit = [json.loads(line) for line in handle]
    selection = read_json(args.run / "selection_summary.json")
    if len(tasks) != selection["selected_count"] or len(audit) != selection["input_proposals"]:
        raise ValueError("Saved task/audit counts differ from selection summary")
    report = build_report(results, audit, selection["videos_without_proposals"],
                          args.min_iou, args.group_by)
    report.update(run_id=manifest["run_id"], source_run=args.run.as_posix())
    write_report(args.output, report)


if __name__ == "__main__":
    main()
