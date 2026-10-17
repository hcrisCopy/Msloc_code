"""Build standalone HTML/PNG reports and compare only compatible demo runs."""

import argparse
import base64
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

from demo_utils import OUTPUT_ROOT, read_json, read_rows, require_root, write_json


def load_run(path):
    config = read_json(path / "run.json")
    if config.get("kind") != "training":
        raise ValueError(f"Not a training run: {path}")
    ranks = []
    for file in sorted(path.glob("events_rank_*.jsonl")):
        steps, metrics, phases, syncs, sessions = {}, {}, defaultdict(dict), [], []
        for row in read_rows(file):
            step = row["step"]
            if row["event"] == "session":
                sessions.append(step)
                # Discard observations from updates rolled back to an earlier checkpoint.
                steps = {k: v for k, v in steps.items() if k <= step}
                metrics = {k: v for k, v in metrics.items() if k <= step}
                phases = defaultdict(dict, {k: v for k, v in phases.items() if k <= step})
                syncs = [r for r in syncs if r["step"] < step]
            elif row["event"] == "step":
                steps[step] = row
            elif row["event"] == "metrics":
                metrics.setdefault(step, {}).update(row["values"])
            elif row["event"] == "phase":
                phases[step][row["phase"]] = phases[step].get(row["phase"], 0.0) + row["seconds"]
            elif row["event"] == "weight_sync":
                syncs.append(row)
        ranks.append({"steps": steps, "metrics": metrics, "phases": phases, "syncs": syncs, "sessions": sessions})
    if len(ranks) != len(config["parameters"]["devices"].split(",")):
        raise ValueError(f"Need every GPU rank's event file: {path}")
    common = sorted(set.intersection(*(set(r["steps"]) for r in ranks)))
    if not common:
        raise ValueError(f"No optimizer step completed on every rank: {path}")
    rows = []
    for step in common:
        rows.append({"step": step,
                     "seconds": max(r["steps"][step]["seconds"] for r in ranks),
                     "allocated_gib": max(r["steps"][step]["allocated_gib"] for r in ranks),
                     "device_used_gib": max(r["steps"][step]["device_used_gib"] for r in ranks),
                     "probe_changed": any(r["steps"][step]["parameter_probe_changed"] for r in ranks),
                     "metrics": ranks[0]["metrics"].get(step, {}),
                     "phases": {name: max(r["phases"].get(step, {}).get(name, 0) for r in ranks)
                                for name in set().union(*(r["phases"].get(step, {}) for r in ranks))}})
    return config, rows, ranks


def comparison_identity(config):
    ignored = {"backend", "output", "devices", "alignment", "vllm_memory"}
    return {"parameters": {k: v for k, v in config["parameters"].items() if k not in ignored},
            "weights": config["weights"], "dataset_sha256": config["dataset_sha256"],
            "data": config["data"], "sources": config["sources"],
            "versions": {k: v for k, v in config["versions"].items() if k != "vllm"},
            "extra": {k: v for k, v in config["extra_inputs"].items() if k != "alignment_sha256"}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--warmup-steps", type=int, default=4)
    args = p.parse_args()
    require_root()
    output = Path(args.output)
    if output.resolve() == OUTPUT_ROOT.resolve() or not output.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Report output must be a subdirectory of the demo output root.")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be nonnegative.")
    loaded = [(Path(path), *load_run(Path(path))) for path in args.runs]
    if any(comparison_identity(item[1]) != comparison_identity(loaded[0][1]) for item in loaded[1:]):
        raise ValueError("Runs differ in method, weights, data, batch, steps or source; do not report a speedup ratio.")
    if len(loaded) > 1 and any(r["sessions"] != [0] for _, _, _, ranks in loaded for r in ranks):
        raise ValueError("Speed comparisons require uninterrupted runs from step zero; plot resumed runs individually.")
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    summaries, cards, sample_html = [], [], []
    for path, config, rows, ranks in loaded:
        params = config["parameters"]
        label = f"{params['stage']} / {params['backend']} / {len(ranks)} GPU"
        x = [r["step"] for r in rows]
        axes[0, 0].plot(x, [r["metrics"].get("loss", float("nan")) for r in rows], label=label)
        axes[1, 0].plot(x, [r["seconds"] for r in rows], label=label)
        axes[1, 1].plot(x, [r["device_used_gib"] for r in rows], label=label + " device used")
        if params["stage"] == "grpo":
            for key in ("reward", "rewards/LocalizationReward/mean", "rewards/FormatReward/mean",
                        "rewards/ExplanationReward/mean", "frac_reward_zero_std"):
                axes[0, 1].plot(x, [r["metrics"].get(key, float("nan")) for r in rows], label=label + " " + key)
        else:
            axes[0, 1].plot(x, [r["metrics"].get("grad_norm", float("nan")) for r in rows], label=label + " grad norm")
        timed = [r for r in rows if r["step"] > args.warmup_steps]
        summary = {"run": str(path), "label": label, "completed_steps": len(rows),
                   "timed_steps": len(timed), "parameter_probe_changed_steps": sum(r["probe_changed"] for r in rows),
                   "sync_calls_per_rank": [len(r["syncs"]) for r in ranks],
                   "session_start_steps": [r["sessions"] for r in ranks],
                   "peak_pytorch_allocated_gib": max(r["allocated_gib"] for r in rows),
                   "peak_observed_device_used_gib": max(r["device_used_gib"] for r in rows)}
        if timed:
            durations = [r["seconds"] for r in timed]
            summary.update(median_step_seconds=statistics.median(durations),
                           mean_step_seconds=statistics.mean(durations),
                           sequences_per_second=params["global_batch_size"] * len(timed) / sum(durations),
                           unique_proposals_per_second=config["unique_inputs_per_batch"] * len(timed) / sum(durations))
        summaries.append(summary)
        metric_cards = {"完成更新": str(len(rows)),
                        "平均 step / s": f"{summary['mean_step_seconds']:.3f}" if timed else "预热后数据不足",
                        "训练序列 / s": f"{summary['sequences_per_second']:.3f}" if timed else "预热后数据不足",
                        "观察到的设备占用 / GiB": f"{summary['peak_observed_device_used_gib']:.2f}"}
        cards.append(f"<h2>{html.escape(label)}</h2><div class='stats'>"
                     + "".join(f"<div><small>{html.escape(k)}</small><strong>{html.escape(v)}</strong></div>" for k, v in metric_cards.items())
                     + f"</div><details><summary>监测摘要</summary><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></details>"
                     f"<details><summary>运行参数</summary><pre>{html.escape(json.dumps(params, ensure_ascii=False, indent=2))}</pre></details>")
        completion_file = path / "completions.jsonl"
        if completion_file.exists():
            # Show original framework records: no fabricated answers or scores.
            examples = read_rows(completion_file)[-4:]
            sample_html.append(f"<h2>{html.escape(label)}：最近生成记录</h2>"
                               + "".join(f"<details><summary>回答与训练信号</summary><pre>{html.escape(json.dumps(e, ensure_ascii=False, indent=2))}</pre></details>" for e in examples))
    for axis, title, ylabel in zip(axes.flat,
            ("Training loss", "Reward / gradient diagnostics", "Observed step cycle (includes monitoring)", "GPU memory"),
            ("Loss", "Value", "Seconds", "GiB")):
        axis.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output / "training.png", dpi=160)
    plt.close(figure)
    phase_figure, phase_ax = plt.subplots(figsize=(12, 5))
    names = sorted(set().union(*(set(r["phases"]) for _, _, rows, _ in loaded for r in rows)))
    for index, (_, config, rows, ranks) in enumerate(loaded):
        values = [statistics.mean([r["phases"].get(name, 0) for r in rows if r["step"] > args.warmup_steps])
                  if any(r["step"] > args.warmup_steps for r in rows) else 0 for name in names]
        width = 0.8 / len(loaded)
        phase_ax.bar([i + index * width for i in range(len(names))], values, width=width,
                     label=f"{config['parameters']['backend']} / {len(ranks)} GPU")
    phase_ax.set(xticks=list(range(len(names))), xticklabels=names, ylabel="Seconds / step",
                 title="Inclusive phase timers: overlapping bars must NOT be added")
    phase_ax.tick_params(axis="x", labelrotation=22, labelsize=8)
    phase_ax.legend()
    phase_figure.tight_layout()
    phase_figure.savefig(output / "phases.png", dpi=160)
    plt.close(phase_figure)
    comparison = {}
    if len(summaries) > 1:
        if any(s["timed_steps"] == 0 for s in summaries) or len({s["completed_steps"] for s in summaries}) != 1:
            raise ValueError("Comparison requires equal completed steps and post-warmup observations.")
        comparison = {s["run"]: summaries[0]["mean_step_seconds"] / s["mean_step_seconds"] for s in summaries}
    write_json(output / "summary.json", {"runs": summaries, "speedup_vs_first_run": comparison,
               "warmup_steps": args.warmup_steps,
               "notes": "Synchronized instrumentation overhead included. Memory is sampled, device-wide. Timing alone does not establish equal model quality."})
    images = "".join('<img alt="Training diagnostics" src="data:image/png;base64,'
                     + base64.b64encode((output / name).read_bytes()).decode() + '">' for name in ("training.png", "phases.png"))
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MSLoc training demo</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f4f6f8;color:#182b3a;max-width:1150px;margin:40px auto;padding:0 24px}h1,h2{color:#123b50}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px;border:1px solid #dbe2e7;border-radius:8px}img{width:100%;background:white;margin:16px 0}details{margin:12px 0}summary{cursor:pointer}p{max-width:950px}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}.stats div{background:white;padding:18px;border-radius:8px;border-top:3px solid #17806b}.stats strong{display:block;font-size:28px}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}table{width:100%;background:white;border-collapse:collapse}</style>
<h1>MSLoc 训练 Demo</h1><p>图表来自真实训练日志。Step 是优化器更新次数。耗时包含同步监测开销；阶段计时存在嵌套，不能相加。设备占用包含同卡其他进程，且不是连续采样的显存峰值。参数探针只观察部分可训练参数，权重同步记录证明接口被调用，不等于对生成引擎全部权重逐项验真。</p>"""
    if comparison:
        page += "<h2>相对第一个运行的速度</h2><p>平均 step 耗时之比，大于 1 表示更快；包含监测开销。</p><table><tr><th>运行</th><th>倍数</th></tr>"
        page += "".join(f"<tr><td>{html.escape(name)}</td><td>{ratio:.3f}×</td></tr>" for name, ratio in comparison.items()) + "</table>"
    page += "".join(cards) + images + "".join(sample_html) + "</html>"
    (output / "report.html").write_text(page, encoding="utf-8")
    print(f"Report: {output}/report.html; charts: training.png, phases.png; summary: summary.json")


if __name__ == "__main__":
    main()
