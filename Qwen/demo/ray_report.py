"""Validate version barriers and render comparable Ray training runs and role timelines."""

import argparse
import base64
import html
import json
import math
import statistics
from pathlib import Path

from demo_utils import OUTPUT_ROOT, read_json, require_root, sha256, write_json


def check_sync(sync, version, replicas, tolerance):
    if (sync["version"] != version or sync["chunks"] != len(sync["checksums"])
            or sync["chunks"] < 1 or sync["bytes_per_receiver"] <= 0):
        raise ValueError("Invalid weight transaction record.")
    for key in ("acknowledgments", "probes"):
        records = sync[key]
        if (sorted(r["rank"] for r in records) != list(range(replicas))
                or any(r["version"] != version for r in records)):
            raise ValueError("Missing or wrong-version rollout replica at sync barrier.")
    if any(a["chunks"] != sync["chunks"] for a in sync["acknowledgments"]):
        raise ValueError("Incomplete weight acknowledgment.")
    if any(not p["verified"] or not math.isfinite(p["max_logp_error"])
           or not 0 <= p["max_logp_error"] <= tolerance for p in sync["probes"]):
        raise ValueError("Missing or failed forward verification for the current policy.")


def comparable(config):
    ignored = {"output", "schedule", "backend", "alignment", "vllm_memory"}
    return {"parameters": {k: v for k, v in config["parameters"].items() if k not in ignored},
            "weights": config["weights"], "data": config["data"], "dataset": config["dataset_sha256"],
            "method": config["method"], "layout": config["layout"], "sources": config["sources"],
            "versions": {k: v for k, v in config["versions"].items() if k != "vllm"},
            "extra": {k: v for k, v in config["extra"].items() if k != "alignment"}}


def read_run(path):
    config, status = read_json(path / "run.json"), read_json(path / "status.json")
    if config["kind"] != "ray_training" or status["state"] != "complete":
        raise ValueError(f"Only completed Ray runs can produce a final report: {path}")
    steps = [read_json(path / "steps" / f"step-{i:06d}.json") for i in range(1, status["version"] + 1)]
    sessions = read_json(path / "sessions.json")
    rollout_count = sum(r["role"] == "rollout" for r in config["layout"])
    trainer_count = sum(r["role"] == "trainer" for r in config["layout"])
    for session in sessions:
        initial = path / f"initial_sync-{session['id']}.json"
        # A failed initialization can precede a later successful recovery.
        if initial.exists():
            check_sync(read_json(initial), session["start_version"], rollout_count, config["parameters"]["sync_atol"])
        elif any(s["session"] == session["id"] for s in steps):
            raise ValueError("Training records have no initial weight verification.")
    for i, step in enumerate(steps):
        version = step["step"]
        if version != i+1 or step["input_version"] != version-1 or step["output_version"] != version:
            raise ValueError("Nonconsecutive policy versions.")
        check_sync(step["sync"], version, rollout_count, config["parameters"]["sync_atol"])
        if (sorted(u["rank"] for u in step["updates"]) != list(range(trainer_count))
                or any(u["version"] != version for u in step["updates"])
                or len({u["probe"] for u in step["updates"]}) != 1 or not math.isfinite(step["loss"])):
            raise ValueError("Invalid or inconsistent trainer update record.")
        if (len(step["jobs"]) != config["parameters"]["global_batch_size"]
                or [j["index"] for j in step["jobs"]] != list(range(len(step["jobs"])))
                or any(j["version"] != version-1 or not j["tokens"] for j in step["jobs"])):
            raise ValueError("Stale on-policy rollout found.")
        # Across a resumed session, elapsed wall time between sessions is not a step duration.
        if i+1 < len(steps) and steps[i+1]["session"] == step["session"]:
            end = steps[i+1]["start"]
        elif i == len(steps)-1 and status["session"] == step["session"]:
            end = status["loop_end"]
        else:
            end = step["end"]
        step["seconds"] = end-step["start"]
        if not math.isfinite(step["seconds"]) or step["seconds"] <= 0:
            raise ValueError("Invalid measured step duration.")
    checkpoint = read_json(Path(status["checkpoint"]) / "COMPLETE.json")
    if (checkpoint["version"] != status["version"] or checkpoint["run_sha256"] != sha256(path / "run.json")
            or checkpoint["step_sha256"] != sha256(path / "steps" / f"step-{status['version']:06d}.json")
            or len({r["trainable_digest"] for r in checkpoint["trainers"]}) != 1):
        raise ValueError("Final checkpoint does not match the completed run.")
    return config, status, sessions, steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--warmup-steps", type=int, default=4)
    args = p.parse_args()
    require_root()
    output = Path(args.output)
    if output.resolve() == OUTPUT_ROOT.resolve() or not output.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Report output must be inside the demo data output root.")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be nonnegative.")
    runs = [(Path(path), *read_run(Path(path))) for path in args.runs]
    labels = [(c["parameters"]["schedule"], c["parameters"]["backend"]) for _, c, _, _, _ in runs]
    if len(labels) != len(set(labels)):
        raise ValueError("Use one run per schedule/backend variant in this comparison.")
    if len(runs) > 1:
        baseline = comparable(runs[0][1])
        hardware = [{k: v for k, v in r.items() if k != "pid"} for r in read_json(runs[0][0] / "hardware.json")]
        for path, config, status, sessions, steps in runs:
            current_hw = [{k: v for k, v in r.items() if k != "pid"} for r in read_json(path / "hardware.json")]
            if comparable(config) != baseline or current_hw != hardware:
                raise ValueError("Runs differ in topology/hardware/method/inputs/versions; speedup comparison rejected.")
            if len(sessions) != 1 or sessions[0]["start_version"] != 0:
                raise ValueError("Compare fresh uninterrupted runs; resumed runs may be plotted individually.")
            if len(steps) <= args.warmup_steps:
                raise ValueError("No post-warmup observations.")
            if [s["batch_indices"] for s in steps] != [s["batch_indices"] for s in runs[0][4]]:
                raise ValueError("The compared runs sampled different proposal batches.")
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 2, figsize=(12, 11))
    summaries, details = [], []
    for path, config, status, sessions, steps in runs:
        params = config["parameters"]
        label = f"{params['schedule']} / {params['backend']}"
        timed = [s for s in steps if s["step"] > args.warmup_steps]
        summary = {"run": str(path), "label": label, "completed_steps": len(steps),
                   "startup_seconds": status["startup_seconds"], "sessions": len(sessions),
                   "all_version_barriers_passed": True,
                   "probe_changed_updates": sum(any(u["probe_changed"] for u in s["updates"]) for s in steps),
                   "last_loss": steps[-1]["loss"], "last_reward": steps[-1]["reward_metrics"].get("mean"),
                   "final_checkpoint": status["checkpoint"],
                   "max_sync_logp_error": max(p["max_logp_error"] for s in steps for p in s["sync"]["probes"])}
        if timed:
            total = sum(s["seconds"] for s in timed)
            summary.update(mean_step_seconds=total/len(timed), median_step_seconds=statistics.median(s["seconds"] for s in timed),
                           sequences_per_second=len(timed)*params["global_batch_size"]/total,
                           generated_tokens_per_second=sum(len(j["tokens"]) for s in timed for j in s["jobs"])/total,
                           mean_response_tokens=statistics.mean(len(j["tokens"]) for s in timed for j in s["jobs"]),
                           mean_sync_bytes_per_receiver=statistics.mean(s["sync"]["bytes_per_receiver"] for s in timed),
                           mean_phases={name: statistics.mean(s["phases"][name] for s in timed)
                                        for name in ("rollout", "score", "train", "sync", "checkpoint")})
        summaries.append(summary)
        x = [s["step"] for s in steps]
        axes[0, 0].plot(x, [s["loss"] for s in steps], label=label)
        axes[0, 1].plot(x, [s["reward_metrics"].get("mean", float("nan")) for s in steps], label=label)
        axes[1, 0].plot(x, [s["seconds"] for s in steps], label=label)
        axes[1, 1].plot(x, [s["sync"]["seconds"] for s in steps], label=label)
        axes[2, 0].plot(x, [max(u["grad_norm"] for u in s["updates"]) for s in steps], label=label)
        axes[2, 1].plot(x, [max(t["memory"]["device_used_gib"] for t in s["traces"] if t["memory"])
                            for s in steps], label=label)
        details.append(f"<h2>{html.escape(label)}</h2><details><summary>运行参数、角色分配与验收数据</summary><pre>"
                       + html.escape(json.dumps({"parameters": params, "layout": config["layout"], "summary": summary}, ensure_ascii=False, indent=2)) + "</pre></details>")
        examples = [{k: v for k, v in j.items() if k != "tokens"} for j in steps[-1]["jobs"][:4]]
        details.append("<details><summary>最后一轮真实回答与奖励</summary><pre>"
                       + html.escape(json.dumps(examples, ensure_ascii=False, indent=2)) + "</pre></details>")
    for axis, title in zip(axes.flat, ("Training loss", "GRPO reward (OPSD has no scalar reward)",
                                       "End-to-end step cycle / s", "Weight transfer + forward probe / s",
                                       "Gradient norm before clipping", "Largest sampled GPU usage / GiB")):
        axis.set(title=title, xlabel="Optimizer update")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "training.png", dpi=160)
    plt.close(figure)
    timeline, chart_axes = plt.subplots(len(runs), 1, figsize=(13, max(4, len(runs)*3.5)), squeeze=False)
    colors = {"rollout": "#1d8391", "teacher": "#dc8b35", "reference": "#dc8b35", "reward": "#9a69b6", "trainer": "#42629b"}
    for ax, (_, config, _, _, steps) in zip(chart_axes[:, 0], runs):
        step = steps[min(args.warmup_steps, len(steps)-1)]
        lanes = sorted({(t["role"], t["rank"]) for t in step["traces"]})
        for trace in step["traces"]:
            lane = lanes.index((trace["role"], trace["rank"]))
            for event in trace["events"]:
                ax.broken_barh([(event["start"]-step["start"], event["end"]-event["start"])],
                               (lane-0.35, 0.7), facecolors=colors[trace["role"]])
        ax.set(yticks=list(range(len(lanes))), yticklabels=[f"{role}-{rank}" for role, rank in lanes], xlabel="Seconds from step start",
               title=f"{config['parameters']['schedule']} / {config['parameters']['backend']} — update {step['step']}")
        ax.grid(axis="x", alpha=0.2)
    timeline.tight_layout()
    timeline.savefig(output / "roles.png", dpi=160)
    plt.close(timeline)
    ratios = {s["label"]: summaries[0]["mean_step_seconds"]/s["mean_step_seconds"] for s in summaries} if len(runs)>1 else {}
    adjacent = {summaries[i]["label"]: summaries[i-1]["mean_step_seconds"]/summaries[i]["mean_step_seconds"]
                for i in range(1, len(summaries))}
    write_json(output / "summary.json", {"runs": summaries, "speedup_vs_first": ratios,
                                        "speedup_vs_previous": adjacent, "warmup_steps": args.warmup_steps})
    heading = "<h1>MSLoc · Ray 多角色同步训练</h1><p>计时来自实际运行。每轮包含生成、监督信号、DDP 更新、权重传输和前向校验；保存步也包含 checkpoint 写盘。角色图显示同轮内并行，没有跨轮异步生成。</p>"
    table = "<table><tr><th>运行</th><th>平均 step / s</th><th>回答 / s</th><th>生成 token / s</th><th>相对首组</th><th>相对上一组</th></tr>"
    for s in summaries:
        table += f"<tr><td>{html.escape(s['label'])}</td>"
        for key in ("mean_step_seconds", "sequences_per_second", "generated_tokens_per_second"):
            table += f"<td>{s[key]:.3f}</td>" if key in s else "<td>预热后数据不足</td>"
        table += f"<td>{ratios[s['label']]:.3f}×</td>" if ratios else "<td>单次运行</td>"
        table += f"<td>{adjacent[s['label']]:.3f}×</td></tr>" if s["label"] in adjacent else "<td>—</td></tr>"
    images = "".join('<img src="data:image/png;base64,' + base64.b64encode((output/name).read_bytes()).decode() + '">' for name in ("training.png", "roles.png"))
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Ray training report</title><style>body{font:16px/1.6 system-ui;background:#f3f6f8;color:#183343;max-width:1200px;margin:40px auto;padding:0 24px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:14px;border-bottom:1px solid #ddd;text-align:left}img{width:100%;margin:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px}details{margin:16px 0}summary{cursor:pointer}</style>'
    document += heading + table + "</table>" + images + "".join(details)
    document += "<p>同资源的 serial→parallel 对比测同轮并发调度收益；parallel Transformers→vLLM 对比测生成后端收益。生成长度可能不同，因此同时查看 token 吞吐与输出质量。显存图是每步结束采样的最忙 GPU 总占用，包含同卡其他进程，不是连续峰值。单输入前向探针不是所有输入上的数值等价证明。没有实测收益时应如实报告。</p></html>"
    (output / "report.html").write_text(document, encoding="utf-8")
    print(f"Ray report: {output}/report.html")


if __name__ == "__main__":
    main()
