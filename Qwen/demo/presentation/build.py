"""Build an offline presentation from existing demo artifacts; never train a model.

Reuses view_predictions.py's ID join and interval rules, check_lengths.py's
official swift template, and the saved training/report schemas. Run at repo root.
"""

import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def indexed(records, key):
    result = {r[key]: r for r in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate {key}")
    return result


def encoded_file(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def iou(target, result):
    a, b = target["target_relative"], result.get("relative_segment")
    if target["target_kind"] == "real":
        return None
    if result["status"] != "fake" or b is None:
        return 0.0
    duration = target["proposal"][1] - target["proposal"][0]
    if not 0 <= b[0] < b[1] <= duration + 0.001:
        raise ValueError("Parsed interval outside proposal")
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return intersection / (a[1] - a[0] + b[1] - b[0] - intersection)


def snippet(path, start, count):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    matches = [i for i, line in enumerate(lines) if start in line]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous source anchor: {path}: {start}")
    i = matches[0]
    return {"file": str(path).replace(str(Path(sys.prefix)), "<conda-env>"),
            "line": i + 1, "text": "\n".join(f"{n+1:4d}  {lines[n]}" for n in range(i, min(i+count, len(lines))))}


def token_example(model, sample, cases):
    # Use exactly the official processor/template path used by check_lengths.py.
    sys.path[:0] = [str(Path("Qwen/demo").resolve()), str(Path("Qwen").resolve())]
    os.environ.update(FORCE_QWENVL_VIDEO_READER="decord", FPS_MAX_FRAMES="40",
                      VIDEO_MAX_TOKEN_NUM="128", TOKENIZERS_PARALLELISM="false")
    import video_plugin  # noqa: F401
    from swift.model import get_processor
    from swift import get_template
    processor = get_processor(model, model_type="qwen3_5", download_model=False)
    template = get_template(processor, enable_thinking=False, add_non_thinking_prefix=True)
    template.set_mode("train")
    data = template.encode(copy.deepcopy(sample))
    ids, labels = data["input_ids"], data["labels"]
    if len(ids) != len(labels) or not any(x != -100 for x in labels):
        raise ValueError("Invalid SFT loss mask")
    tokenizer = processor.tokenizer
    def compact_decode(ids):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        return re.sub(r"(?P<token><\|[^<>]+\|>)(?P=token)+",
                      lambda m: m.group("token") + f" [repeated {len(m.group(0))//len(m.group('token'))} times]", text)
    template.set_mode("transformers")
    for case in cases:
        case["encoded_prompts"] = {}
        for role in ["student", "teacher"]:
            view = copy.deepcopy(case["opsd"])
            if role == "teacher":
                view["messages"] = [{"role":"user", "content":view["teacher_prompt"]}]
            encoded = template.encode(view)
            decoded = compact_decode(encoded["input_ids"])
            case["encoded_prompts"][role] = {"tokens":len(encoded["input_ids"]),
                                             "text":decoded,
                                             "times":re.findall(r"<([\d.]+) seconds>",decoded)}
            if len(case["encoded_prompts"][role]["times"]) != 20:
                raise ValueError("Expected 20 temporal patches from 40 input frames")
    selected = [i for i, x in enumerate(labels) if x != -100]
    first = selected[0]
    show = list(range(max(0, first - 8), min(len(ids), first + 16)))
    show += list(range(max(first + 16, len(ids) - 18), len(ids)))
    return {"total": len(ids), "supervised": len(selected), "ignored": len(ids)-len(selected),
            "first": first, "rows": [{"position": i, "id": int(ids[i]), "label": int(labels[i]),
                                      "text": tokenizer.decode([ids[i]], skip_special_tokens=False)} for i in show],
            "supervised_text": tokenizer.decode([ids[i] for i in selected], skip_special_tokens=False)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="../MSLoc_data/Qwen/demo")
    p.add_argument("--model", default="../Qwen/Qwen3.5-4B")
    p.add_argument("--video-root", default="../MSLoc_data/data/Tasle-CoT-10K/videos")
    p.add_argument("--output", default="../MSLoc_data/Qwen/demo/presentation/index.html")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not Path("Qwen/demo/train.py").is_file():
        raise ValueError("Run from Msloc_code/")
    root, output = Path(args.root), Path(args.output)
    if output.suffix != ".html" or not output.resolve().is_relative_to(Path("../MSLoc_data/Qwen/demo").resolve()):
        raise ValueError("Output must be an HTML file under ../MSLoc_data/Qwen/demo")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output}; pass --overwrite to rebuild only this report")
    inputs = []

    def load_rows(relative):
        path = root / relative
        inputs.append(path)
        return rows(path)

    targets = indexed(load_rows("data/targets.jsonl"), "id")
    stages = ["student_before", "student_after", "student_after_grpo_hf", "student_after_grpo_vllm", "teacher_before"]
    predictions = {}
    inference_runs = {}
    for name in stages:
        predictions[name] = indexed(load_rows(name + "/predictions.jsonl"), "sample_id")
        if set(predictions[name]) != set(targets):
            raise ValueError(f"Incomplete predictions: {name}")
        run_path = root / name / "run.json"
        inputs.append(run_path)
        inference_runs[name] = read(run_path)
        expected_mode = "teacher" if name == "teacher_before" else "student"
        if inference_runs[name]["mode"] != expected_mode:
            raise ValueError(f"Unexpected inference mode: {name}")
    sft = indexed(load_rows("data/sft.jsonl"), "sample_id")
    opsd = indexed(load_rows("data/opsd.jsonl"), "sample_id")
    grpo = indexed(load_rows("data/grpo.jsonl"), "sample_id")
    completions = load_rows("grpo_hf/completions.jsonl")

    def scores(id):
        return [iou(targets[id], predictions[n][id]) for n in stages[:4]]

    # Audit every saved checkpoint prediction, separately from training rollouts.
    inference = [{"id": id, "kind": t["target_kind"], "proposal": t["proposal"],
                  "target_relative": t["target_relative"],
                  "predictions": {n: dict(predictions[n][id], iou=iou(t, predictions[n][id]))
                                  for n in stages[:4]}}
                 for id, t in targets.items()]
    trajectory = {}
    for branch in stages[2:4]:
        fake = [r for r in inference if r["kind"] == "fake"]
        real = [r for r in inference if r["kind"] == "real"]
        def improved(row, before, after):
            return row["predictions"][after]["iou"] > row["predictions"][before]["iou"] + 1e-9
        trajectory[branch] = {
            "fake_count": len(fake), "real_count": len(real),
            "continuous": [r["id"] for r in fake if improved(r, stages[0], stages[1])
                           and improved(r, stages[1], branch)],
            "grpo_improved": [r["id"] for r in fake if improved(r, stages[1], branch)],
            "real_corrected": [r["id"] for r in real
                               if r["predictions"][stages[0]]["status"] == "fake"
                               and r["predictions"][branch]["status"] == "real"]}

    # Require a valid initial interval for the main localization example.
    candidates = [id for id,t in targets.items() if t["target_kind"] == "fake"
                  and predictions[stages[0]][id]["status"] == "fake"
                  and scores(id)[1] > scores(id)[0] and scores(id)[2] >= scores(id)[1]]
    if not candidates:
        raise ValueError("No qualifying OPSD improvement example; revise the presentation selection explicitly")
    main_id = max(candidates, key=lambda id: (scores(id)[1]-scores(id)[0], id))
    grpo_id = max((id for id,t in targets.items() if t["target_kind"] == "fake"),
                  key=lambda id: (scores(id)[2]-scores(id)[1], id))
    real_id = next(id for id,t in targets.items() if t["target_kind"] == "real"
                   and predictions[stages[2]][id]["status"] != "real")
    case_ids = [main_id, grpo_id, real_id]
    cases = []
    for index,id in enumerate(case_ids):
        t = targets[id]
        clip = Path(t["clip"])
        inputs.extend([clip, Path(t["timestamps"])])
        groups = []
        for record in completions:
            indices = [i for i,v in enumerate(record["sample_id"]) if v == id]
            if not indices:
                continue
            if len(indices) != 4 or len({record["step"][i] for i in indices}) != 1:
                raise ValueError(f"Expected one complete generation group for {id}")
            answers = []
            for i in indices:
                r = [float(record[k][i]) for k in ["LocalizationReward", "FormatReward", "ExplanationReward"]]
                answers.append({"raw": record["completion"][i], "loc": r[0], "format": r[1], "explanation": r[2],
                                "reward": r[0]+0.1*r[1]+0.3*r[2], "advantage": float(record["advantages"][i])})
            mean = sum(a["reward"] for a in answers)/4
            std = math.sqrt(sum((a["reward"]-mean)**2 for a in answers)/3)
            for a in answers:
                if abs(a["advantage"]-(a["reward"]-mean)/(std+1e-4)) > 2e-4:
                    raise ValueError("Saved advantage disagrees with group normalization")
            groups.append({"step": int(record["step"][indices[0]]), "answers": answers, "mean": mean, "std": std})
        if not groups:
            raise ValueError(f"No saved GRPO group for selected example {id}")
        group = max(groups, key=lambda g: g["std"])
        case = {"id": id, "target": t, "video": encoded_file(clip, "video/mp4"),
                "timestamps": read(t["timestamps"])["relative_milliseconds"],
                "predictions": {n: dict(predictions[n][id], iou=iou(t,predictions[n][id])) for n in stages},
                "sft": sft.get(id), "opsd": opsd[id], "grpo": grpo[id], "group": group}
        if index == 0:
            original = Path(args.video_root) / id.rsplit("::",1)[0]
            inputs.append(original)
            case["original_video"] = encoded_file(original,"video/mp4")
        cases.append(case)
    print("Selected real examples:", *case_ids, sep="\n", flush=True)
    print("Encoding one SFT sample with the official CPU processor (no model weights)", flush=True)
    tokens = token_example(args.model, sft[main_id], cases)
    metrics = []
    for name in stages:
        fake = [iou(t, predictions[name][id]) for id,t in targets.items() if t["target_kind"] == "fake"]
        real = [predictions[name][id]["status"] == "real" for id,t in targets.items() if t["target_kind"] == "real"]
        invalid = sum(r["status"] not in ("fake","real") for r in predictions[name].values())
        metrics.append({"stage": name, "iou": sum(fake)/len(fake), "real_correct": sum(real),
                        "real_count": len(real), "invalid": invalid})
    training = {}
    for name in ["sft", "opsd_hf", "grpo_hf", "grpo_vllm"]:
        training[name] = {"config": read(root/name/"run.json")["parameters"],
                          "logs": load_rows(name+"/logging.jsonl"),
                          "chart": encoded_file(root/name/"report/training.png", "image/png")}
        # The final train_runtime summary repeats step 32; only loss rows are updates.
        progress = [r for r in training[name]["logs"] if "loss" in r]
        steps = [int(r["global_step/max_steps"].split("/")[0]) for r in progress]
        if steps != list(range(1, training[name]["config"]["steps"] + 1)):
            raise ValueError(f"Incomplete optimizer-step records: {name}")
        size = len(sft) if name == "sft" else len(opsd) if name == "opsd_hf" else len(grpo)
        visits = float(progress[-1]["epoch"]) * size
        if abs(visits - round(visits)) > 1e-4:
            raise ValueError(f"Epoch progress does not map to whole samples: {name}")
        training[name]["data_visits"] = round(visits)
        inputs.extend([root/name/"run.json",root/name/"report/training.png"])
    swift_root = Path(importlib.util.find_spec("swift").origin).parent
    sources = {
        "jsd": snippet(swift_root/"rlhf_trainers/gkd_loss.py", "def jsd_loss(", 55),
        "grpo": snippet(swift_root/"rlhf_trainers/grpo_trainer.py", "            coef_2 = torch.clamp(coef_1,", 14),
        "kl": snippet(swift_root/"rlhf_trainers/grpo_trainer.py", "            safe_ratio = torch.clamp(ref_per_token_logps", 2),
        "reward": snippet(Path("Qwen/grpo_rewards.py"), "class LocalizationReward(ORM):", 20),
        "proposal": snippet(Path("DeMamba/util.py"), "        binary_preds = [1 if score > threshold", 23)}
    sources["template"] = snippet(Path("Qwen/trace_video_template.py"), '        metadata = {', 9)
    sources["inference"] = snippet(Path("Qwen/demo/inspect_model.py"), '        engine = TransformersEngine(', 14)
    sources["ray_sync"] = snippet(Path("Qwen/demo/ray_driver.py"), '    def sync_weights(version, full=False):', 23)
    inputs.extend([Path("Qwen/demo/inspect_model.py"), Path("Qwen/demo/ray_driver.py"),
                   Path("Qwen/demo/ray_protocol.py")])
    speed = read(root/"grpo_compare/summary.json")
    inputs.append(root/"grpo_compare/summary.json")
    gate = read(root/"teacher_gate.json")
    inputs.append(root/"teacher_gate.json")
    payload = {"cases": cases, "tokens": tokens, "metrics": metrics, "training": training,
               "inference": inference, "inference_runs": inference_runs, "trajectory": trajectory,
               "speed": speed, "sources": sources, "sample_count":len(targets), "gate_passed":gate["passed"],
               "selection": "主例按有效初始区间的 OPSD IoU 增量选取，要求 GRPO HF 不退步；补例按 GRPO HF 增量选取；反例取首个 GRPO HF 未正确判断的 real。不是随机样本，也不是平均效果。"}
    payload["provenance"] = [{"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in dict.fromkeys(inputs)]
    here = Path(__file__).parent
    html = (here/"page.html").read_text(encoding="utf-8")
    for tag,file in [("__CSS__","style.css"),("__JS__","app.js")]:
        html = html.replace(tag,(here/file).read_text(encoding="utf-8"))
    html = html.replace("__DATA__", json.dumps(payload,ensure_ascii=False).replace("<","\\u003c"))
    if any(t in html for t in ["__CSS__","__JS__","__DATA__"]):
        raise ValueError("Unresolved template placeholder")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(html,encoding="utf-8")
    temporary.replace(output)
    audit = {"output":str(output), "bytes":output.stat().st_size, "cases":case_ids,
             "sft_mask":{k:v for k,v in tokens.items() if k in ["total","supervised","ignored","first"]},
             "metrics":metrics, "trajectory":trajectory, "provenance":payload["provenance"]}
    output.with_suffix(".audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Offline report: {output} ({output.stat().st_size/1024/1024:.2f} MiB)")


if __name__ == "__main__":
    main()
