"""从第一阶段 proposal 和 TASLE 训练标注生成 Qwen3.5 视频 LoRA SFT 数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from common import clip_name, clip_timestamps_path, explanation, gt_segments, make_clip, number, overlap, proposal_segments, read_records, target_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.frames != 40:
        raise ValueError("Qwen 输入固定为 40 帧")
    output_dir = Path(args.output_dir)
    data_root = Path("../MSLoc_data/Qwen").resolve()
    if output_dir.resolve() == data_root or not output_dir.resolve().is_relative_to(data_root):
        raise ValueError("训练数据输出必须位于 ../MSLoc_data/Qwen/ 下")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "train.jsonl"
    audit_path = output_dir / "targets.jsonl"
    if (dataset_path.exists() or audit_path.exists()) and not args.clean:
        raise FileExistsError("输出已存在；重建时显式指定 --clean，已有视频片段仍可复用")
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    gt = {row["video_path"]: row for row in read_records(args.annotation)}
    proposals = read_records(args.proposals)
    if set(gt) != {row["video_path"] for row in proposals}:
        raise ValueError("proposal 与 GT 的 video_path 集合不同")
    dataset_tmp = dataset_path.with_suffix(".jsonl.tmp")
    audit_tmp = audit_path.with_suffix(".jsonl.tmp")
    positive = 0
    negative = 0
    multiple = 0
    with dataset_tmp.open("w", encoding="utf-8") as data_file, audit_tmp.open("w", encoding="utf-8") as audit_file:
        for row in tqdm(proposals, desc="SFT proposals", unit="video"):
            video_name = row["video_path"]
            relative_video = Path(video_name)
            if relative_video.is_absolute() or ".." in relative_video.parts:
                raise ValueError(f"video_path 必须是视频根目录下的相对路径：{video_name}")
            source = Path(args.video_root) / relative_video
            targets = gt_segments(gt[video_name])
            for index, proposal in enumerate(proposal_segments(row)):
                hits = [(overlap(proposal, (start, end)), start, end, ann) for start, end, ann in targets]
                hits = [hit for hit in hits if hit[0] > 0]
                if not hits:
                    negative += 1
                    continue
                if len(hits) > 1:
                    multiple += 1
                _, start, end, ann = max(hits, key=lambda hit: hit[0])
                clipped = (max(start, proposal[0]), min(end, proposal[1]))
                relative = (clipped[0] - proposal[0], clipped[1] - proposal[0])
                clip = output_dir / "clips" / relative_video.with_suffix("") / clip_name(index, proposal)
                make_clip(source, proposal, clip, args.frames)
                caption = explanation(ann)
                if clipped != (start, end) and len(caption) == 3:
                    caption = [caption[1]]
                user = f"{prompt}\n\nClip duration: {number(proposal[1] - proposal[0])} seconds."
                sample = {
                    "messages": [
                        {"role": "user", "content": f"<video>{user}"},
                        {"role": "assistant", "content": target_text(caption, relative)},
                    ],
                    "videos": [str(clip)],
                    "chat_template_kwargs": {"nframes": 40},
                }
                audit = {
                    "video_path": video_name,
                    "proposal_index": index,
                    "proposal": list(proposal),
                    "target_absolute": list(clipped),
                    "target_relative": list(relative),
                    "clip": str(clip),
                    "timestamps": str(clip_timestamps_path(clip)),
                    "explanation": caption,
                    "overlapping_gt_count": len(hits),
                }
                data_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
                positive += 1
    if positive == 0:
        raise ValueError("没有与 GT 相交的训练 proposal")
    dataset_tmp.replace(dataset_path)
    audit_tmp.replace(audit_path)
    print(f"SFT samples={positive} negative_excluded={negative} multiple_gt={multiple}")
    print(f"Dataset: {dataset_path}\nAudit: {audit_path}\nClips: {output_dir / 'clips'}")


if __name__ == "__main__":
    main()
