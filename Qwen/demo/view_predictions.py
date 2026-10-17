"""Render saved inference results with videos and relative-time interval bars."""

import argparse
import html
import os
from pathlib import Path
from urllib.parse import quote

from demo_utils import OUTPUT_ROOT, data_identity, read_json, read_rows, require_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="inspect_model.py student/teacher output directories")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    require_root()
    output = Path(args.output)
    if output.suffix != ".html" or not output.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Output must be an HTML file under ../MSLoc_data/Qwen/demo/.")
    targets = read_rows(Path(args.data) / "targets.jsonl")
    data_hash = data_identity(args.data)
    runs = []
    for path in args.runs:
        config = read_json(Path(path) / "run.json")
        if config["mode"] not in ("student", "teacher") or config["data"] != data_hash:
            raise ValueError("Prediction runs must use the same demo data.")
        rows = read_rows(Path(path) / "predictions.jsonl")
        by_id = {r["sample_id"]: r for r in rows}
        if len(by_id) != len(rows) or set(by_id) != {t["id"] for t in targets}:
            raise ValueError("Missing or duplicate prediction IDs.")
        runs.append((path, config, by_id))

    def escape(value):
        return html.escape(str(value), quote=True)

    def interval_bar(segment, duration, label, color):
        if segment is None:
            return f"<p>{escape(label)}: 无区间</p>"
        start, end = map(float, segment)
        if not 0 <= start < end <= duration + 0.001:
            raise ValueError("Saved interval lies outside its proposal.")
        return (f"<p>{escape(label)}: [{start:g}, {end:g}] s</p><div class='track'>"
                f"<div class='bar' style='left:{100*start/duration:.4f}%;width:{100*(end-start)/duration:.4f}%;"
                f"background:{color}'></div></div>")

    cards = []
    for target in targets:
        duration = target["proposal"][1] - target["proposal"][0]
        video = quote(Path(os.path.relpath(Path(target["clip"]).resolve(), output.parent.resolve())).as_posix(), safe="/:")
        card = (f"<article><h2>{escape(target['id'])}</h2><p>proposal={escape(target['proposal'])} s · "
                f"clip duration={duration:g} s · GT={escape(target['target_kind'])}</p>"
                f"<video controls preload='none' src='{escape(video)}'></video>"
                + interval_bar(target["target_relative"], duration, "GT", "#17806b"))
        for name, config, records in runs:
            row = records[target["id"]]
            card += (f"<section><h3>{escape(name)} · {escape(config['mode'])}</h3>"
                     f"<p>status={escape(row['status'])}</p>"
                     + interval_bar(row.get("relative_segment"), duration, "Prediction", "#c27b30")
                     + f"<pre>{escape(row['raw'])}</pre></section>")
        cards.append(card + "</article>")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MSLoc prediction examples</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f4f6f8;color:#183442;max-width:1100px;margin:40px auto;padding:0 24px}article{background:white;border:1px solid #dbe2e7;border-radius:12px;padding:24px;margin:24px 0}h2,h3{overflow-wrap:anywhere}video{width:100%;max-height:340px;background:#172631}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#f3f6f7}section{border-top:1px solid #ddd;margin-top:20px}.track{position:relative;background:#e2e8eb;height:16px;border-radius:4px;overflow:hidden}.bar{position:absolute;height:100%}</style>
<h1>MSLoc 片段与回答</h1><p>时间条以原 proposal 的相对秒数为准。预览 MP4 将非均匀抽取的 40 帧重新编码，播放器时间不是原 proposal 时间；不要用播放进度读取定位秒数。教师结果使用了标签与区间特权信息，仅作训练前检查，不是测试集性能。</p>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page + "".join(cards) + "</html>", encoding="utf-8")
    print(f"Prediction report: {output}; keep the referenced video files available.")


if __name__ == "__main__":
    main()
