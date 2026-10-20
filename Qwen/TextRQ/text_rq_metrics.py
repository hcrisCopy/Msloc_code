"""Proposal-weighted summaries and IoU associations, retaining tool domains.

Spearman correlation follows scipy.stats.spearmanr:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from text_rq_data import ALL_DIMENSIONS


def aggregate(rows: list[dict]) -> dict:
    rq = [row["score"] for row in rows if row["score"] is not None]
    dimensions = {}
    for name in ALL_DIMENSIONS:
        values = [row["scores"][name]["score"] for row in rows
                  if row["scores"][name] is not None]
        dimensions[name] = {
            "count": len(values), "mean": sum(values) / len(values) if values else None,
        }
    return {
        "count": len(rows), "video_count": len({row["video_path"] for row in rows}),
        "rq_count": len(rq), "rq_mean": sum(rq) / len(rq) if rq else None,
        "dimensions": dimensions,
    }


def score_views(rows: list[dict]) -> dict:
    return {
        "all": aggregate(rows),
        "reference_scored": aggregate([row for row in rows if row["reference_available"]]),
        "high_iou_fake": aggregate([row for row in rows if row["high_iou_fake"]]),
        "correct_real": aggregate([row for row in rows if row["correct_real"]]),
        "by_prediction_class": {
            key: aggregate([row for row in rows if (row["prediction_kind"] or "unknown") == key])
            for key in ("fake", "real", "unknown")
        },
        "by_gt_clip_class": {
            key: aggregate([row for row in rows if row["kind"] == key])
            for key in ("fake", "real")
        },
        "by_outcome": {
            key: aggregate([row for row in rows if row["outcome"] == key])
            for key in ("TP", "FP", "FN", "TN", "invalid")
        },
        "correct_real_by_source_video_type": {
            key: aggregate([row for row in rows
                            if row["correct_real"] and row["source_video_type"] == key])
            for key in ("real", "fake")
        },
    }


def selection_stats(audit: list[dict]) -> dict:
    return {
        "input_proposals": len(audit),
        "input_videos_with_proposals": len({row["video_path"] for row in audit}),
        "selection_counts": dict(Counter(row["selection"] for row in audit)),
        "parse_status_counts": dict(Counter(row["prediction_status"] for row in audit)),
        "outcome_counts": dict(Counter(row["outcome"] for row in audit)),
        "reference_policy_counts": dict(Counter(row["reference_policy"] for row in audit)),
        "reference_available_count": sum(row["reference_available"] for row in audit),
        "high_iou_fake_count": sum(row["high_iou_fake"] for row in audit),
        "correct_real_count": sum(row["correct_real"] for row in audit),
    }


def summarize(results: list[dict], audit: list[dict], no_proposal: int) -> dict:
    domains = sorted({row["tool_domain"] for row in audit})
    return {
        "unit": "proposal", "aggregation": "equal_weight_per_scored_proposal",
        "videos_without_proposals": no_proposal,
        **selection_stats(audit), **score_views(results),
        "by_tool_domain": {
            domain: {
                "input": selection_stats([row for row in audit if row["tool_domain"] == domain]),
                **score_views([row for row in results if row["tool_domain"] == domain]),
            } for domain in domains
        },
    }


def iou_pairs(results: list[dict]) -> list[dict]:
    """Real decisions have no interval; real GT has no anomaly IoU."""
    pairs = []
    for row in results:
        if row["prediction_status"] != "fake" or row["iou"] is None:
            continue
        pairs.append({
            "id": row["id"], "video_path": row["video_path"], "tool_domain": row["tool_domain"],
            "iou": row["iou"], "reference_policy": row["reference_policy"],
            "matched_gt_index": row["matched_gt_index"], "outcome": row["outcome"],
            "rq": row["score"],
            **{name: row["scores"][name]["score"] if row["scores"][name] else None
               for name in ALL_DIMENSIONS},
        })
    return pairs


def spearman(values: list[tuple[float, float]]) -> float | None:
    if len(values) < 2:
        return None
    x, y = zip(*values)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    from scipy.stats import spearmanr

    result = float(spearmanr(x, y).statistic)
    if not math.isfinite(result):
        raise ValueError("Non-finite Spearman correlation")
    return result


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def correlation(pairs: list[dict], metric: str, repeats: int) -> dict:
    rows = [row for row in pairs if row[metric] is not None]
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["video_path"]].append((row["iou"], row[metric]))
    rho = spearman([(row["iou"], row[metric]) for row in rows])
    result = {
        "count": len(rows), "video_count": len(clusters), "spearman": rho,
        "ci95_video_bootstrap": None, "bootstrap_valid_repeats": 0,
        "undefined_reason": None if rho is not None else "insufficient_pairs_or_constant_values",
        "ci_unavailable_reason": None,
    }
    if rho is None or len(clusters) < 2 or repeats == 0:
        result["ci_unavailable_reason"] = (
            "undefined_correlation" if rho is None else
            "fewer_than_two_videos" if len(clusters) < 2 else "bootstrap_disabled"
        )
        return result
    # Resample whole videos, preserving all dependent proposals within each draw.
    rng = random.Random(42)
    names = sorted(clusters)
    estimates = []
    for _ in range(repeats):
        sampled = [pair for name in rng.choices(names, k=len(names)) for pair in clusters[name]]
        estimate = spearman(sampled)
        if estimate is not None:
            estimates.append(estimate)
    result["bootstrap_valid_repeats"] = len(estimates)
    # A mostly degenerate bootstrap must not be presented as a reliable interval.
    if len(estimates) >= max(20, math.ceil(0.8 * repeats)):
        result["ci95_video_bootstrap"] = [percentile(estimates, 0.025),
                                        percentile(estimates, 0.975)]
    else:
        result["ci_unavailable_reason"] = "insufficient_non_degenerate_bootstrap_draws"
    return result


def association_group(rows: list[dict], pairs: list[dict], repeats: int, progress=None) -> dict:
    bins = {}
    for label, lower, upper in (
        ("[0,0.3]", None, 0.3), ("(0.3,0.5]", 0.3, 0.5),
        ("(0.5,0.7]", 0.5, 0.7), ("(0.7,1]", 0.7, 1.0),
    ):
        bins[label] = aggregate([row for row in rows
                                 if row["prediction_status"] == "fake" and row["iou"] is not None
                                 and (lower is None or row["iou"] > lower) and row["iou"] <= upper])
    correlations = {}
    for name in ("rq", *ALL_DIMENSIONS):
        correlations[name] = correlation(pairs, name, repeats)
        if progress is not None:
            progress.update(1)
    return {
        "pair_count": len(pairs), "iou_bins": bins,
        "reference_policy_counts": dict(Counter(row["reference_policy"] for row in pairs)),
        "correlations": correlations,
    }


def association_analysis(results: list[dict], repeats: int, show_progress: bool = False) -> dict:
    if repeats < 0:
        raise ValueError("bootstrap-repeats must be nonnegative")
    pairs = iou_pairs(results)
    domains = sorted({row["tool_domain"] for row in results})
    from tqdm import tqdm

    with tqdm(total=(len(domains) + 1) * (len(ALL_DIMENSIONS) + 1),
              desc="IoU analysis", unit="metric", disable=not show_progress) as progress:
        all_stats = association_group(results, pairs, repeats, progress)
        by_domain = {}
        for domain in domains:
            by_domain[domain] = association_group(
                [row for row in results if row["tool_domain"] == domain],
                [row for row in pairs if row["tool_domain"] == domain], repeats, progress,
            )
    return {
        "unit": "proposal", "seed": 42, "bootstrap_repeats": repeats,
        "scope": "valid_fake_predictions_in_videos_with_anomaly_gt",
        "note": ("IoU is hidden from judges. Missing factual references stay null. "
                 "Associations do not establish that explanations cause better localization."),
        "all": all_stats, "by_tool_domain": by_domain,
    }
