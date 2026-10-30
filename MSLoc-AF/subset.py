"""Persist a deterministic, disjoint small-video protocol for every pipeline stage."""
import random
from pathlib import Path

from common import load_records, read_json, sha256, write_json


def select_records(records, count, seed):
    if len(records) < count:
        raise ValueError(f"Subset needs {count} videos, but this split only has {len(records)}")
    rng = random.Random(seed)
    groups = {kind: [r for r in records if r["type"] == kind] for kind in ("real", "fake")}
    selected = []
    # Keep both authenticity types whenever the source split contains them.
    for group in groups.values():
        if group:
            selected.append(rng.choice(group))
    ids = {r["video_path"] for r in selected}
    remaining = [r for r in records if r["video_path"] not in ids]
    selected += rng.sample(remaining, count-len(selected))
    return sorted(selected, key=lambda r: r["video_path"])


def prepare_subset(paths, output, max_videos, seed):
    """10 total = 8/2 train/test, or 6/2/2 if validation is explicitly provided."""
    names = ("train", "val", "test") if "val" in paths else ("train", "test")
    if set(paths) != set(names):
        raise ValueError("Subset requires train/test, with an optional val split")
    if max_videos < 2*len(names):
        raise ValueError(f"Small-data mode requires at least {2*len(names)} videos (2 per split)")
    data = {name: load_records(path) for name, path in paths.items()}
    for i, left in enumerate(names):
        for right in names[i+1:]:
            overlap = {r["video_path"] for r in data[left]} & {r["video_path"] for r in data[right]}
            if overlap:
                raise ValueError(f"Source {left}/{right} overlap: {sorted(overlap)[:3]}")
    test_count = max(2, round(max_videos*0.2))
    val_count = test_count if "val" in paths else 0
    counts = dict(train=max_videos-val_count-test_count, test=test_count)
    if val_count:
        counts["val"] = val_count
    selected = {name: select_records(data[name], counts[name], seed+i)
                for i, name in enumerate(names)}
    if not any(r["type"] == "fake" for r in selected["train"]):
        raise ValueError("The selected training split has no fake videos")
    spec = dict(max_videos=max_videos, seed=seed, counts=counts,
                source_sha256={name:sha256(path) for name,path in paths.items()},
                video_ids={name:[r["video_path"] for r in values] for name,values in selected.items()})
    output = Path(output)
    manifest = output / "subset.json"
    if manifest.exists() and read_json(manifest) != spec:
        raise ValueError("Existing subset uses different source data/seed/count; use a new directory")
    result = {name: output / f"{name}.json" for name in names}
    for name, path in result.items():
        if path.exists() and read_json(path) != selected[name]:
            raise ValueError(f"Existing subset annotations differ: {path}")
        if not path.exists():
            write_json(path, selected[name])
    if not manifest.exists():
        write_json(manifest, spec)
    print(f"Small-data protocol: {counts} = {max_videos} videos; seed={seed}; {manifest}", flush=True)
    return result
