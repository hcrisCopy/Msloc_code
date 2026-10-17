"""Run matched serial-HF, parallel-HF and parallel-vLLM experiments on the server."""

import argparse
import subprocess
import sys
from pathlib import Path

from demo_utils import require_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True)
    p.add_argument("--variants", nargs="+", choices=["serial_hf", "parallel_hf", "parallel_vllm"],
                   default=["serial_hf", "parallel_hf", "parallel_vllm"])
    p.add_argument("--clean", action="store_true")
    p.add_argument("training_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    require_root()
    common = args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args
    banned = ("--backend", "--schedule", "--output", "--resume", "--clean", "--plan-only")
    if not common or any(value.split("=", 1)[0] in banned for value in common):
        raise ValueError("Pass common training arguments after --; benchmark controls backend/schedule/output/resume.")
    if len(args.variants) != len(set(args.variants)):
        raise ValueError("Duplicate benchmark variants.")
    outputs = []
    for variant in args.variants:
        path = Path(args.output_root) / variant
        command = [sys.executable, "Qwen/demo/ray_launch.py", *common,
                   "--backend", "vllm" if variant.endswith("vllm") else "transformers",
                   "--schedule", "serial" if variant.startswith("serial") else "parallel",
                   "--output", path.as_posix(), "--resume", "none"]
        if args.clean:
            command.append("--clean")
        print(f"\n=== Benchmark: {variant} ===", flush=True)
        subprocess.run(command, check=True)
        outputs.append(path.as_posix())
    subprocess.run([sys.executable, "Qwen/demo/ray_report.py", "--runs", *outputs,
                    "--warmup-steps", "4", "--output", (Path(args.output_root) / "comparison").as_posix()], check=True)


if __name__ == "__main__":
    main()
