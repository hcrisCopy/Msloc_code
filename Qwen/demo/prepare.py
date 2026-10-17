"""Build a small deterministic dataset from existing MSLoc proposal clips."""

import argparse
import copy
import random
from pathlib import Path

from demo_utils import (data_identity, prepare_output, read_json, read_rows, relative,
                        require_root, sha256, write_json, write_rows)
from common import explanation, gt_segments, overlap, read_records, target_text
from prepare_grpo import evidence_for
from opsd_common import student_message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opsd-dataset", required=True)
    parser.add_argument("--opsd-targets", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--samples", type=int, default=64,
                        help="Total proposal clips across fake and real, shared by all demo stages (default: 64).")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                        help="Explicit clip path prefix replacement; repeat for multiple roots.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", choices=["none", "auto"], default="none")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    require_root()
    if args.samples < 1:
        raise ValueError("--samples must be positive.")
    mappings = []
    for entry in args.path_map:
        old, sep, new = entry.partition("=")
        if not sep or not old or not new:
            raise ValueError("Use --path-map OLD=NEW")
        mappings.append((old.replace("\\", "/").rstrip("/"), new.rstrip("/")))

    def relocate(value):
        value = value.replace("\\", "/")
        matches = [(old, new) for old, new in mappings if value == old or value.startswith(old + "/")]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous path mapping: {value}")
        if matches:
            old, new = matches[0]
            value = new + value[len(old):]
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"{value}; use --path-map for paths from another server.")
        return relative(path)

    samples, targets = read_rows(args.opsd_dataset), read_rows(args.opsd_targets)
    if len(samples) != len(targets) or len({r["id"] for r in targets}) != len(targets):
        raise ValueError("OPSD samples/targets are not a one-to-one sequence.")
    annotations = {r["video_path"]: r for r in read_records(args.annotation)}
    groups = {kind: [i for i, r in enumerate(targets) if r["target_kind"] == kind] for kind in ("fake", "real")}
    source_counts = {kind: len(indices) for kind, indices in groups.items()}
    if sum(source_counts.values()) != len(targets):
        raise ValueError("Every source target_kind must be fake or real.")
    if args.samples > len(samples):
        raise ValueError("--samples exceeds the number of source proposals.")
    # Preserve source proposal proportions using integer largest-remainder quotas.
    # Equal remainders use the stable fake/real order; no minority oversampling.
    quotas = {kind: args.samples * count // len(samples) for kind, count in source_counts.items()}
    remainders = {kind: args.samples * count % len(samples) for kind, count in source_counts.items()}
    remaining = args.samples - sum(quotas.values())
    for kind in sorted(groups, key=lambda kind: remainders[kind], reverse=True)[:remaining]:
        quotas[kind] += 1
    rng = random.Random(42)
    chosen = sorted(index for kind, indices in groups.items() for index in rng.sample(indices, quotas[kind]))
    print(f"Source proposals: {source_counts}; proportional subset: {quotas}; seed=42")
    config = {"source": {name: sha256(getattr(args, name)) for name in
                         ("opsd_dataset", "opsd_targets", "annotation")},
              "samples": args.samples, "path_map": args.path_map, "seed": 42,
              "sampling": "source_proportional_largest_remainder",
              "source_counts": source_counts, "sample_counts": quotas,
              "prepare_sources": {name: sha256(name) for name in
                  ("Qwen/demo/prepare.py", "Qwen/demo/demo_utils.py", "Qwen/common.py",
                   "Qwen/opsd_common.py", "Qwen/prepare_grpo.py", "Qwen/prompts/student_sft.txt")}}
    protected = [args.opsd_dataset, args.opsd_targets, args.annotation]
    protected += [relocate(samples[i]["videos"][0]) for i in chosen]
    output = prepare_output(args.output, config, args.resume, args.clean, protected)
    if args.resume == "auto" and (output / "manifest.json").exists():
        data_identity(output)
        print(f"Prepared data already complete: {output}")
        return
    from tqdm import tqdm
    opsd_rows, grpo_rows, sft_rows, audit, media = [], [], [], [], []
    sft_prompt = Path("Qwen/prompts/student_sft.txt").read_text(encoding="utf-8").strip()
    for index in tqdm(chosen, desc="Prepare demo", unit="proposal"):
        sample, target = copy.deepcopy(samples[index]), copy.deepcopy(targets[index])
        if len(sample["videos"]) != 1 or len(sample["messages"]) != 1 or not sample.get("teacher_prompt"):
            raise ValueError(f"Invalid OPSD row: {target['id']}")
        if samples[index]["videos"] != [targets[index]["clip"]]:
            raise ValueError(f"OPSD sample/target order mismatch: {target['id']}")
        clip = Path(relocate(sample["videos"][0]))
        sidecar = clip.with_suffix(".timestamps.json")
        timing = read_json(sidecar)
        if timing["proposal"] != target["proposal"] or timing["sampling"] != "trace16_8_16":
            raise ValueError(f"Wrong timestamps: {clip}")
        duration = target["proposal"][1] - target["proposal"][0]
        name = target["id"].rsplit("::", 1)[0]
        evidence = evidence_for(target["proposal"], annotations[name], target)
        sample.update(videos=[relative(clip)], sample_id=target["id"])
        target.update(clip=relative(clip), timestamps=relative(sidecar))
        opsd_rows.append(sample)
        grpo_rows.append({key: sample[key] for key in ("messages", "videos", "chat_template_kwargs", "sample_id")}
                         | {"target_kind": target["target_kind"], "target_relative": target["target_relative"],
                            "clip_duration": duration, "evidence": evidence})
        if target["target_kind"] == "fake":
            hits = [(overlap(tuple(target["proposal"]), (s, e)), s, e, ann)
                    for s, e, ann in gt_segments(annotations[name])]
            _, start, end, ann = max(hits, key=lambda row: row[0])
            captions = explanation(ann)
            if target["explanation_sentence_count"] == 1 and len(captions) == 3:
                captions = [captions[1]]
            sft_rows.append({"messages": [
                {"role": "user", "content": student_message(sft_prompt, duration)},
                {"role": "assistant", "content": target_text(captions, tuple(target["target_relative"]))}],
                "videos": sample["videos"], "chat_template_kwargs": {"nframes": 40},
                "sample_id": target["id"]})
        audit.append(target)
        media.append({"clip": relative(clip), "size": clip.stat().st_size,
                      "mtime_ns": clip.stat().st_mtime_ns, "timestamps_sha256": sha256(sidecar)})
    files = {"sft.jsonl": sft_rows, "opsd.jsonl": opsd_rows, "grpo.jsonl": grpo_rows, "targets.jsonl": audit}
    for name, rows in files.items():
        write_rows(output / name, rows)
    write_json(output / "manifest.json", {"complete": True, "config": config, "media": media,
               "files": {name: sha256(output / name) for name in files},
               "counts": {"fake": len(sft_rows), "real": len(opsd_rows) - len(sft_rows)}})
    print(f"Data: {output}; fake={len(sft_rows)}, real={len(opsd_rows)-len(sft_rows)}")


if __name__ == "__main__":
    main()
