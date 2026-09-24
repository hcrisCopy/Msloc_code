"""从第一阶段训练 proposal 构造 Qwen3.5 视频 OPSD 数据，含异常和正常片段。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from common import clip_name, make_clip, proposal_segments, read_records
from opsd_common import proposal_target, student_message, teacher_message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--student-prompt-file", required=True)
    parser.add_argument("--teacher-prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.frames <= 0:
        raise ValueError("--frames 必须大于 0")
    output = Path(args.output_dir)
    if not output.resolve().is_relative_to(Path("../MSLoc_data/Qwen").resolve()):
        raise ValueError("OPSD 数据必须写入 ../MSLoc_data/Qwen/ 下")
    dataset = output / "train.jsonl"
    audit = output / "targets.jsonl"
    config_path = output / "data_config.json"
    if (dataset.exists() or audit.exists() or config_path.exists()) and not args.clean:
        raise FileExistsError("OPSD 数据已存在；重建时指定 --clean，片段文件可复用")
    student_prompt = Path(args.student_prompt_file).read_text(encoding="utf-8").strip()
    teacher_prompt = Path(args.teacher_prompt_file).read_text(encoding="utf-8").strip()
    annotations = {row["video_path"]: row for row in read_records(args.annotation)}
    proposals = read_records(args.proposals)
    if set(annotations) != {row["video_path"] for row in proposals}:
        raise ValueError("训练 proposal 与 GT 的 video_path 集合不同")
    output.mkdir(parents=True, exist_ok=True)
    counts = {"fake": 0, "real": 0, "multiple_gt": 0}
    dataset_tmp = dataset.with_suffix(".jsonl.tmp")
    audit_tmp = audit.with_suffix(".jsonl.tmp")
    with dataset_tmp.open("w", encoding="utf-8") as data_file, audit_tmp.open("w", encoding="utf-8") as audit_file:
        for row in tqdm(proposals, desc="OPSD proposals", unit="video"):
            name = row["video_path"]
            relative_video = Path(name)
            if relative_video.is_absolute() or ".." in relative_video.parts:
                raise ValueError(f"video_path 必须是视频根目录下的相对路径：{name}")
            for index, proposal in enumerate(proposal_segments(row)):
                target = proposal_target(proposal, annotations[name])
                counts[target["kind"]] += 1
                counts["multiple_gt"] += target["overlapping_gt_count"] > 1
                clip = output / "clips" / relative_video.with_suffix("") / clip_name(index, proposal)
                make_clip(Path(args.video_root) / relative_video, proposal, clip, args.frames)
                student = student_message(student_prompt, proposal[1] - proposal[0])
                teacher = teacher_message(student, teacher_prompt, target, proposal[1] - proposal[0])
                sample = {
                    "messages": [{"role": "user", "content": student}],
                    "videos": [str(clip)],
                    "teacher_prompt": teacher,
                }
                audit_row = {
                    "id": f"{name}::{index}",
                    "proposal": list(proposal),
                    "target_kind": target["kind"],
                    "target_relative": target["relative_segment"],
                    "overlapping_gt_count": target["overlapping_gt_count"],
                    "manipulation_class": target.get("manipulation_class"),
                    "annotation_hints": target.get("annotation_hints"),
                    "clip": str(clip),
                }
                data_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                audit_file.write(json.dumps(audit_row, ensure_ascii=False) + "\n")
    if counts["fake"] == 0 or counts["real"] == 0:
        raise ValueError(f"OPSD 训练数据必须包含异常和正常 proposal：{counts}")
    dataset_tmp.replace(dataset)
    audit_tmp.replace(audit)
    config = {
        "proposals": args.proposals,
        "annotation": args.annotation,
        "student_prompt_text": student_prompt,
        "teacher_prompt_text": teacher_prompt,
        "frames": args.frames,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OPSD samples: {counts}")
    print(f"Dataset: {dataset}\nAudit: {audit}\nClips: {output / 'clips'}")


if __name__ == "__main__":
    main()
