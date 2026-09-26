"""从第一阶段训练 proposal 构造 Qwen3.5 视频 OPSD 数据，含异常和正常片段。"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from common import clip_name, clip_timestamps_path, make_clip, proposal_segments, read_records
from opsd_common import proposal_target, student_message, teacher_message
from prepare_progress import PrepareProgress, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--student-prompt-file", required=True)
    parser.add_argument("--teacher-opsd-prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--resume", choices=["none", "auto"], required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.frames != 40:
        raise ValueError("Qwen 输入固定为 40 帧")
    output = Path(args.output_dir)
    data_root = Path("../MSLoc_data/Qwen").resolve()
    if output.resolve() == data_root or not output.resolve().is_relative_to(data_root):
        raise ValueError("OPSD 数据必须写入 ../MSLoc_data/Qwen/ 下")
    student_prompt = Path(args.student_prompt_file).read_text(encoding="utf-8").strip()
    teacher_opsd_prompt = Path(args.teacher_opsd_prompt_file).read_text(encoding="utf-8").strip()
    annotations = {row["video_path"]: row for row in read_records(args.annotation)}
    proposals = read_records(args.proposals)
    if set(annotations) != {row["video_path"] for row in proposals}:
        raise ValueError("训练 proposal 与 GT 的 video_path 集合不同")
    inputs = {
        "proposals": str(Path(args.proposals).resolve()),
        "proposals_sha256": sha256(Path(args.proposals)),
        "annotation": str(Path(args.annotation).resolve()),
        "annotation_sha256": sha256(Path(args.annotation)),
        "video_root": str(Path(args.video_root).resolve()),
        "student_prompt_text": student_prompt,
        "teacher_opsd_prompt_text": teacher_opsd_prompt,
        "frames": args.frames,
        "sampling": "trace16_8_16",
    }
    progress = PrepareProgress(output, inputs, args.resume, args.clean)
    if len(progress.completed) > len(proposals):
        raise ValueError("OPSD 续建进度超过输入视频数")
    for index, record in enumerate(progress.completed):
        if record["video_path"] != proposals[index]["video_path"]:
            raise ValueError(f"OPSD 续建顺序与输入 proposal 不符：第 {index} 条视频")
    counts = {"fake": 0, "real": 0, "multiple_gt": 0}
    buckets = {name: 0 for name in ("positive", "hard_positive", "near_hard_negative", "real_false_positive")}
    for record in progress.completed:
        for key in counts:
            counts[key] += record["counts"][key]
        for key in buckets:
            buckets[key] += record["buckets"][key]
    for row in tqdm(proposals[len(progress.completed):], desc="OPSD proposals", unit="video",
                    initial=len(progress.completed), total=len(proposals)):
        name = row["video_path"]
        relative_video = Path(name)
        if relative_video.is_absolute() or ".." in relative_video.parts:
            raise ValueError(f"video_path 必须是视频根目录下的相对路径：{name}")
        samples = []
        audit_rows = []
        video_counts = {key: 0 for key in counts}
        video_buckets = {key: 0 for key in buckets}
        for index, proposal in enumerate(proposal_segments(row)):
            target = proposal_target(proposal, annotations[name])
            counts[target["kind"]] += 1
            video_counts[target["kind"]] += 1
            counts["multiple_gt"] += target["overlapping_gt_count"] > 1
            video_counts["multiple_gt"] += target["overlapping_gt_count"] > 1
            buckets[target["replay_bucket"]] += 1
            video_buckets[target["replay_bucket"]] += 1
            clip = output / "clips" / relative_video.with_suffix("") / clip_name(index, proposal)
            make_clip(Path(args.video_root) / relative_video, proposal, clip, args.frames)
            student = student_message(student_prompt, proposal[1] - proposal[0])
            teacher = teacher_message(student, teacher_opsd_prompt, target, mode="opsd")
            sample = {
                "messages": [{"role": "user", "content": student}],
                "videos": [str(clip)],
                "chat_template_kwargs": {"nframes": 40},
                "teacher_prompt": teacher,
            }
            audit_row = {
                "id": f"{name}::{index}",
                "proposal": list(proposal),
                "target_kind": target["kind"],
                "target_relative": target["relative_segment"],
                "overlapping_gt_count": target["overlapping_gt_count"],
                "source_video_type": target["source_video_type"],
                "replay_bucket": target["replay_bucket"],
                "max_gt_iou": target["max_gt_iou"],
                "nearest_gt_gap_seconds": target["nearest_gt_gap_seconds"],
                "object_name": target.get("object_name"),
                "object_class": target.get("object_class"),
                "manipulation_type": target.get("manipulation_type"),
                "object_subclass": target.get("object_subclass"),
                "start_class": target.get("start_class"),
                "end_class": target.get("end_class"),
                "explanation_sentence_count": target.get("explanation_sentence_count"),
                "clip": str(clip),
                "timestamps": str(clip_timestamps_path(clip)),
            }
            samples.append(sample)
            audit_rows.append(audit_row)
        progress.append({"video_path": name, "samples": samples, "targets": audit_rows,
                         "counts": video_counts, "buckets": video_buckets})
    if counts["fake"] == 0 or counts["real"] == 0:
        raise ValueError(f"OPSD 训练数据必须包含异常和正常 proposal：{counts}")
    config = {
        "proposals": args.proposals,
        "annotation": args.annotation,
        "student_prompt_text": student_prompt,
        "teacher_opsd_prompt_text": teacher_opsd_prompt,
        "frames": args.frames,
        "sampling": "trace16_8_16",
    }
    progress.finish({**config, "counts": counts, "buckets": buckets})
    print(f"OPSD samples: {counts}; Trace replay buckets: {buckets}")
    print(f"Dataset: {progress.dataset}\nAudit: {progress.audit}\nClips: {output / 'clips'}")


if __name__ == "__main__":
    main()
