"""在服务器检查第二阶段依赖、GPU 及 Decord/PyAV 视频读写。"""

from __future__ import annotations

import importlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu

import trl_compat  # noqa: F401
import trace_video_template  # noqa: F401
from common import clip_timestamps_path, make_clip, write_clip_pixels


def check_cuda_extensions() -> None:
    """按官方接口调用真实 CUDA 算子，并与 PyTorch 参考运算比较。"""
    from causal_conv1d import causal_conv1d_fn
    from flash_attn import flash_attn_func
    import torch.nn.functional as F

    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 16, 2, 32, device="cuda", dtype=torch.bfloat16,
                          requires_grad=True) for _ in range(3)]
    result = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    with torch.no_grad():
        scores = q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-2, -1)
        expected = ((scores / 32 ** 0.5).softmax(-1) @ v.float().transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(result.float(), expected, atol=0.02, rtol=0.02)
    result.float().square().mean().backward()
    if any(tensor.grad is None or not torch.isfinite(tensor.grad).all().item() for tensor in (q, k, v)):
        raise RuntimeError("Flash Attention 反向传播异常")

    x = torch.randn(2, 4, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4, 3, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    result = causal_conv1d_fn(x, weight, activation=None)
    with torch.no_grad():
        expected = F.conv1d(x.float(), weight.float().unsqueeze(1), padding=2, groups=4)[..., :16]
    torch.testing.assert_close(result.float(), expected, atol=0.04, rtol=0.02)
    result.float().square().mean().backward()
    if any(tensor.grad is None or not torch.isfinite(tensor.grad).all().item() for tensor in (x, weight)):
        raise RuntimeError("causal-conv1d 反向传播异常")
    torch.cuda.synchronize()
    print("Flash Attention / causal-conv1d forward + backward: OK", flush=True)


def main() -> None:
    # 包括用户保留的三个训练后端，仅检查可导入，不改变训练开关。
    for module_name, package_name in (
        ("swift", "ms-swift"), ("transformers", "transformers"),
        ("trl", "trl"), ("peft", "peft"), ("vllm", "vllm"),
        ("deepspeed", "deepspeed"), ("liger_kernel", "liger-kernel"),
        ("flash_attn", "flash-attn"), ("causal_conv1d", "causal-conv1d"),
        ("fla", "flash-linear-attention"), ("decord", "decord"), ("av", "av"),
    ):
        importlib.import_module(module_name)
        print(f"{package_name}: {version(package_name)}", flush=True)

    if not torch.__version__.startswith("2.10.0") or torch.version.cuda != "12.8":
        raise RuntimeError(f"需要 PyTorch 2.10.0 cu128：{torch.__version__}, {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 不可用，请检查服务器驱动")
    result = (torch.ones(1, device="cuda") + 1).item()
    if result != 2:
        raise RuntimeError("GPU 运算结果异常")
    print(f"GPU: {torch.cuda.get_device_name(0)} / CUDA {torch.version.cuda}", flush=True)
    check_cuda_extensions()

    import grpo_logging  # noqa: F401
    from swift.rlhf_trainers.gkd_trainer import GKDTrainer  # noqa: F401

    # 通过真实写文件和读取帧验证预编译包，而非只检查 import。
    output = Path("../MSLoc_data/Qwen/env_check")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source.mp4"
    pixels = np.zeros((80, 64, 64, 3), dtype=np.uint8)
    pixels[:, :, :, 0] = np.arange(80, dtype=np.uint8)[:, None, None] * 3
    write_clip_pixels(pixels, source, 20.0)
    reader = VideoReader(str(source), ctx=cpu(0), fault_tol=1e-12)
    if len(reader) != 80 or not np.isclose(reader.get_avg_fps(), 20.0):
        raise RuntimeError("源视频帧数或 FPS 与编码设置不一致")
    decoded = reader.get_batch([0, 79]).asnumpy()
    if decoded.shape != (2, 64, 64, 3) or decoded[1, :, :, 0].mean() <= decoded[0, :, :, 0].mean():
        raise RuntimeError("视频 RGB 帧读取异常")

    clip = output / "proposal_trace16_8_16.mp4"
    make_clip(source, (0.0, 3.95), clip, 40)
    timing = json.loads(clip_timestamps_path(clip).read_text(encoding="utf-8"))
    expected_indices = (
        np.linspace(0, 16, 16, dtype=int).tolist()
        + np.linspace(16, 63, 8, dtype=int).tolist()
        + np.linspace(63, 79, 16, dtype=int).tolist()
    )
    if timing["source_frame_indices"] != expected_indices:
        raise RuntimeError("16/8/16 采样索引不符合预期")
    if timing["relative_milliseconds"] != [round(index / 20.0 * 1000) for index in expected_indices]:
        raise RuntimeError("片段内真实时间戳不符合预期")
    if len(VideoReader(str(clip), ctx=cpu(0), fault_tol=1e-12)) != 40:
        raise RuntimeError("输出片段不是 40 帧")
    print(f"Environment check: OK / Video artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
