"""ms-swift 4.5.3 的 Qwen3.5 视频模板：保留 Trace 16/8/16 采样的真实相对时间。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from qwen_vl_utils import fetch_video
from swift.template.register import TEMPLATE_MAPPING, register_template
from swift.template.templates.qwen import Qwen3_5Template


class TraceVideoQwen3_5Template(Qwen3_5Template):
    def replace_tag(self, media_type, index, inputs):
        if media_type != "video":
            return super().replace_tag(media_type, index, inputs)

        clip = inputs.videos[index]
        if not isinstance(clip, str) or not clip.endswith("_trace16_8_16.mp4"):
            raise ValueError(f"Qwen 视频必须是 16/8/16 片段 MP4：{clip!r}")
        sidecar = Path(clip).with_suffix(".timestamps.json")
        timing = json.loads(sidecar.read_text(encoding="utf-8"))
        milliseconds = timing["relative_milliseconds"]
        if (timing["sampling"] != "trace16_8_16" or timing["timebase_hz"] != 1000
                or len(milliseconds) != 40 or any(type(value) is not int or value < 0 for value in milliseconds)
                or milliseconds != sorted(milliseconds)
                or milliseconds[-1] > round((timing["proposal"][1] - timing["proposal"][0]) * 1000) + 1):
            raise ValueError(f"16/8/16 片段时间戳无效：{sidecar}")

        # 沿用 ms-swift/Qwen 官方的图像缩放与视频读取，只替换它按均匀间隔生成的元数据。
        video, decoded_metadata = fetch_video(
            {"video": clip, **inputs.chat_template_kwargs},
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_metadata=True,
        )
        if (video.shape[0] != 40 or decoded_metadata["total_num_frames"] != 40
                or decoded_metadata["video_backend"] != "torchcodec"):
            raise ValueError(f"16/8/16 片段必须由 torchcodec 完整读取 40 帧：{clip}")
        metadata = {
            "fps": 1000.0,
            "frames_indices": milliseconds,
            "total_num_frames": max(40, round((timing["proposal"][1] - timing["proposal"][0]) * 1000) + 1),
        }
        inputs.mm_processor_kwargs.setdefault("video_metadata", []).append(metadata)
        inputs.mm_processor_kwargs["do_sample_frames"] = False
        inputs.videos[index] = video.to(torch.uint8)
        return ["<|video_pad|>"]


def register_trace_video_template() -> None:
    original = TEMPLATE_MAPPING["qwen3_5"]
    if original.template_cls is TraceVideoQwen3_5Template:
        return
    if original.template_cls is not Qwen3_5Template:
        raise RuntimeError("ms-swift Qwen3.5 模板与预期不符，不能替换视频时间戳")
    template = copy.copy(original)
    template.template_cls = TraceVideoQwen3_5Template
    register_template(template, exist_ok=True)


# swift --external_plugins 导入文件时自动注册；直接评测也显式调用此函数。
register_trace_video_template()
