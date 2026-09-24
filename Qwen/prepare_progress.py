"""SFT/OPSD 数据准备的逐视频进度记录；中断时只重做未提交的视频。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PrepareProgress:
    def __init__(self, output: Path, inputs: dict, resume: str, clean: bool):
        if clean and resume != "none":
            raise ValueError("--clean 与 --resume auto 不能同时使用")
        self.output = output
        self.config_path = output / "data_config.json"
        self.progress_path = output / "progress.jsonl.tmp"
        self.dataset = output / "train.jsonl"
        self.audit = output / "targets.jsonl"
        self.dataset_tmp = output / "train.jsonl.tmp"
        self.audit_tmp = output / "targets.jsonl.tmp"
        self.inputs = inputs
        self.completed = []
        if resume == "auto" and not output.is_dir():
            raise FileNotFoundError(f"没有可续建的数据目录：{output}")
        output.mkdir(parents=True, exist_ok=True)
        if clean:
            # 已解码的视频和时间戳可复用，只清理本步骤生成的数据文件。
            for path in (self.config_path, self.progress_path, self.dataset,
                         self.audit, self.dataset_tmp, self.audit_tmp):
                path.unlink(missing_ok=True)
        if resume == "auto":
            if not self.config_path.is_file() or not self.progress_path.is_file():
                raise FileNotFoundError("没有可续建的数据记录；旧版本的临时 JSONL 请用 --clean 重建，视频片段会复用")
            saved = json.loads(self.config_path.read_text(encoding="utf-8"))
            if saved.get("inputs") != inputs or saved.get("complete") is not False:
                raise ValueError("续建的输入文件或配置与原运行不符，或数据已经完成")
            with self.progress_path.open("rb+") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    # 中断落在写入一条记录的中途时，只丢弃该条未提交记录。
                    if not line.endswith(b"\n"):
                        handle.truncate(offset)
                        break
                    record = json.loads(line)
                    if len(record["samples"]) != len(record["targets"]):
                        raise ValueError("进度记录的训练样本和审计行数不同")
                    self.completed.append({key: record[key] for key in ("video_path", "counts", "buckets")
                                           if key in record})
        else:
            if any(path.exists() for path in (self.config_path, self.progress_path,
                                               self.dataset, self.audit,
                                               self.dataset_tmp, self.audit_tmp)):
                raise FileExistsError("输出已存在；续建使用 --resume auto，重建使用 --clean")
            self.config_path.write_text(
                json.dumps({"inputs": inputs, "complete": False}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.progress_path.touch()

    def append(self, record: dict) -> None:
        if len(record["samples"]) != len(record["targets"]):
            raise ValueError("训练样本和审计行数不同")
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def finish(self, extra_config: dict) -> None:
        # 一次只读一条视频记录，避免把全部样本重新装入内存。
        with self.progress_path.open(encoding="utf-8") as source, \
             self.dataset_tmp.open("w", encoding="utf-8") as samples, \
             self.audit_tmp.open("w", encoding="utf-8") as targets:
            for line in source:
                record = json.loads(line)
                for sample, target in zip(record["samples"], record["targets"]):
                    samples.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    targets.write(json.dumps(target, ensure_ascii=False) + "\n")
        self.dataset_tmp.replace(self.dataset)
        self.audit_tmp.replace(self.audit)
        self.config_path.write_text(
            json.dumps({"inputs": self.inputs, **extra_config, "complete": True},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
