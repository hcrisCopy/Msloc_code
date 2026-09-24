"""复用 OPSD 的 40 帧片段，构造不向学生泄漏真值的 GRPO 数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from common import gt_segments, overlap, read_records
from opsd_common import student_message


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_for(proposal: list[float], annotation: dict, target: dict) -> dict:
    """与 proposal_target 同样选最长交集，只保留片段内可见的标注事实。"""
    hits = [(overlap(tuple(proposal), (start, end)), start, end, ann)
            for start, end, ann in gt_segments(annotation)]
    hits = [hit for hit in hits if hit[0] > 0]
    if target["target_kind"] == "real":
        if hits:
            raise ValueError(f"OPSD 负例与 GT 相交：{target['id']}")
        return {}
    if not hits:
        raise ValueError(f"OPSD 正例没有 GT 交集：{target['id']}")
    _, start, end, ann = max(hits, key=lambda hit: hit[0])
    clipped = [max(start, proposal[0]), min(end, proposal[1])]
    relative = [clipped[0] - proposal[0], clipped[1] - proposal[0]]
    if any(abs(a - b) > 1e-5 for a, b in zip(relative, target["target_relative"])):
        raise ValueError(f"OPSD 与 GRPO 的目标区间不同：{target['id']}")
    obj = (ann.get("obj_cot") or [])[0]
    if target["explanation_sentence_count"] not in (1, 3):
        raise ValueError(f"解释句数必须是 1 或 3：{target['id']}")
    one_sentence = target["explanation_sentence_count"] == 1
    start_ann = (ann.get("bnd_cot_st") or [{}])[0]
    end_ann = (ann.get("bnd_cot_ed") or [{}])[0]
    evidence = {
        "object_caption": str(obj.get("obj_caption", "")).strip(),
        "object_class": str(obj.get("bnd_sub_class", "")).strip(),
        "start_caption": "" if one_sentence else str(start_ann.get("bnd_caption", "")).strip(),
        "start_class": "" if one_sentence else str(start_ann.get("bnd_class", "")).strip(),
        "end_caption": "" if one_sentence else str(end_ann.get("bnd_caption", "")).strip(),
        "end_class": "" if one_sentence else str(end_ann.get("bnd_class", "")).strip(),
    }
    if not evidence["object_caption"]:
        raise ValueError(f"缺少对象异常标注：{target['id']}")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opsd-dataset", required=True)
    parser.add_argument("--opsd-targets", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", choices=["none", "auto"], required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    data_root = Path("../MSLoc_data/Qwen").resolve()
    if output.resolve() == data_root or not output.resolve().is_relative_to(data_root):
        raise ValueError("GRPO 数据必须写入 ../MSLoc_data/Qwen/ 下")
    source_config = json.loads((Path(args.opsd_dataset).parent / "data_config.json").read_text(encoding="utf-8"))
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if source_config["student_prompt_text"] != prompt or source_config["sampling"] != "trace16_8_16":
        raise ValueError("GRPO 必须复用 OPSD 的 student.txt 和 16/8/16 视频")
    annotations = {row["video_path"]: row for row in read_records(args.annotation)}
    if source_config["annotation"] != args.annotation:
        raise ValueError("GRPO 与 OPSD 的标注文件必须相同")
    opsd_root = Path(args.opsd_dataset).parent.resolve()
    if output.resolve() == opsd_root or opsd_root.is_relative_to(output.resolve()):
        raise ValueError("GRPO 输出目录不能覆盖 OPSD 数据")
    if args.clean and args.resume != "none":
        raise ValueError("--clean 与 --resume auto 不能同时使用")
    if args.clean and output.exists():
        shutil.rmtree(output)
    if args.resume == "none" and output.exists():
        raise FileExistsError(f"GRPO 数据目录已存在：{output}；继续使用 --resume auto 或重建使用 --clean")
    if args.resume == "auto" and not output.is_dir():
        raise FileNotFoundError(f"没有可续建的 GRPO 目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "train.jsonl"
    dataset_tmp = output / "train.jsonl.tmp"
    config_path = output / "data_config.json"
    source_paths = (Path(args.opsd_dataset), Path(args.opsd_targets), Path(args.annotation), Path(args.prompt_file))
    source_hashes = {str(path.resolve()): sha256(path) for path in source_paths}
    config = {
        "opsd_dataset": args.opsd_dataset,
        "opsd_targets": args.opsd_targets,
        "annotation": args.annotation,
        "prompt_text": prompt,
        "sampling": "trace16_8_16",
        "frames": 40,
        "source_hashes": source_hashes,
    }
    count = {"fake": 0, "real": 0}
    completed = 0
    if args.resume == "auto":
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in config.items()) or saved.get("complete") is not False:
            raise ValueError("GRPO 续建的输入文件或配置与原运行不符，或数据已经完成")
        if dataset.exists() or not dataset_tmp.is_file():
            raise ValueError("GRPO 续建文件状态不一致")
        with dataset_tmp.open(encoding="utf-8") as existing:
            for line in existing:
                row = json.loads(line)
                count[row["target_kind"]] += 1
                completed += 1
    else:
        config_path.write_text(json.dumps({**config, "complete": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    total = 0
    with Path(args.opsd_dataset).open(encoding="utf-8") as samples, \
         Path(args.opsd_targets).open(encoding="utf-8") as targets, \
         dataset_tmp.open("a" if args.resume == "auto" else "w", encoding="utf-8") as writer:
        from itertools import zip_longest
        for index, (sample_line, target_line) in enumerate(tqdm(zip_longest(samples, targets), desc="GRPO proposals", unit="clip")):
            total = index + 1
            if sample_line is None or target_line is None:
                raise ValueError("OPSD 训练数据与目标审计行数不一致")
            if index < completed:
                continue
            sample, target = json.loads(sample_line), json.loads(target_line)
            video_name, _, proposal_index = target["id"].rpartition("::")
            if not video_name or not proposal_index.isdigit() or video_name not in annotations:
                raise ValueError(f"无效的 proposal ID：{target['id']}")
            if len(sample["videos"]) != 1:
                raise ValueError(f"OPSD 每条数据必须恰有一个片段：{target['id']}")
            clip = Path(sample["videos"][0])
            if sample["videos"] != [target["clip"]] or not clip.is_file() or not Path(target["timestamps"]).is_file():
                raise ValueError(f"OPSD 片段或时间戳缺失：{target['id']}")
            sidecar = json.loads(Path(target["timestamps"]).read_text(encoding="utf-8"))
            if sidecar["sampling"] != "trace16_8_16" or sidecar["proposal"] != target["proposal"]:
                raise ValueError(f"片段时间戳与目标 proposal 不一致：{target['id']}")
            if sample["chat_template_kwargs"] != {"nframes": 40} or len(sample["messages"]) != 1:
                raise ValueError(f"OPSD 视频输入格式不符：{target['id']}")
            duration = target["proposal"][1] - target["proposal"][0]
            if sample["messages"][0] != {"role": "user", "content": student_message(prompt, duration)}:
                raise ValueError(f"OPSD 学生提示词或时长不符：{target['id']}")
            if target["target_kind"] not in count:
                raise ValueError(f"未知片段类别：{target['id']}")
            if target["target_kind"] == "real" and target["target_relative"] is not None:
                raise ValueError(f"真实片段不能有 GT 区间：{target['id']}")
            evidence = evidence_for(target["proposal"], annotations[video_name], target)
            row = {
                "messages": sample["messages"],
                "videos": sample["videos"],
                "chat_template_kwargs": sample["chat_template_kwargs"],
                "target_kind": target["target_kind"],
                "target_relative": target["target_relative"],
                "clip_duration": duration,
                "evidence": evidence,
            }
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            writer.flush()
            count[target["target_kind"]] += 1
        if completed > total:
            raise ValueError("GRPO 续建行数超过原始 OPSD 数据")
    if not all(count.values()):
        raise ValueError(f"GRPO 必须有异常和正常 proposal：{count}")
    dataset_tmp.replace(dataset)
    config_path.write_text(json.dumps({**config, "counts": count, "complete": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GRPO samples: {count}\nDataset: {dataset}")


if __name__ == "__main__":
    main()
