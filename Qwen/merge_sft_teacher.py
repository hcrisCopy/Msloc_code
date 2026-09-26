"""将已预检的 SFT LoRA 合并为 OPSD 使用的固定教师模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    model, adapter, output = (Path(value).resolve() for value in
                              (args.model, args.adapter, args.output))
    if not model.is_dir() or not adapter.is_dir():
        raise FileNotFoundError("底座模型或 SFT LoRA 不存在")
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise FileNotFoundError(f"SFT LoRA 缺少配置或权重：{adapter}")
    sft_config = json.loads((adapter.parent / "sft_config.json").read_text(encoding="utf-8"))
    if Path(sft_config["model"]).resolve() != model:
        raise ValueError("SFT LoRA 与底座模型不匹配")
    data_root = Path("../MSLoc_data/Qwen").resolve()
    if (output == data_root or not output.is_relative_to(data_root)
            or not output.name.startswith("sft_teacher")):
        raise ValueError("教师模型必须写入 ../MSLoc_data/Qwen/sft_teacher* 独立目录")
    if (model.is_relative_to(output) or adapter.is_relative_to(output)
            or output.is_relative_to(model) or output.is_relative_to(adapter)):
        raise ValueError("教师输出目录不能覆盖底座或 SFT LoRA")
    devices = [value.strip() for value in args.devices.split(",")]
    if not devices or any(not value.isdigit() for value in devices) or len(set(devices)) != len(devices):
        raise ValueError("--devices 应为不重复的 GPU 编号")
    source = {
        "model": str(model),
        "adapter": str(adapter),
        "adapter_config_sha256": sha256(config),
        "adapter_weights_sha256": sha256(weights),
    }
    source_path = output / "teacher_source.json"
    if output.exists() and not args.clean:
        if (source_path.is_file() and (output / "config.json").is_file()
                and list(output.glob("*.safetensors"))
                and json.loads(source_path.read_text(encoding="utf-8")) == source):
            print(f"复用固定教师模型：{output}")
            return
        raise FileExistsError(f"教师模型目录不完整或来源不符：{output}；重建时传 --clean")
    if args.clean and output.exists():
        shutil.rmtree(output)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    command = ["swift", "export", "--model", str(model), "--adapters", str(adapter),
               "--merge_lora", "true", "--output_dir", str(output),
               "--torch_dtype", "bfloat16"]
    subprocess.run(command, env=env, check=True)
    if not (output / "config.json").is_file() or not list(output.glob("*.safetensors")):
        raise RuntimeError(f"LoRA 合并结束，但教师模型权重或配置缺失：{output}")
    source_path.write_text(
        json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"固定教师模型：{output}")


if __name__ == "__main__":
    main()
