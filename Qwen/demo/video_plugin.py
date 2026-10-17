"""Demo-only Qwen3.5 video bridge, based on swift 4.5.3 / vLLM 0.19.1.

HF expands a bare video pad; vLLM replaces the complete vision wrapper.
Both receive the same decoded frames and original nonuniform timestamps.
"""

import copy

from demo_utils import check_versions
check_versions()
from trace_video_template import TraceVideoQwen3_5Template
from swift.template.register import TEMPLATE_MAPPING, register_template


class DemoVideoTemplate(TraceVideoQwen3_5Template):
    def replace_tag(self, media_type, index, inputs):
        tokens = super().replace_tag(media_type, index, inputs)
        if media_type != "video" or self.mode != "vllm":
            return tokens
        metadata = inputs.mm_processor_kwargs.pop("video_metadata")
        if len(metadata) != 1:
            raise ValueError("The demo supports exactly one proposal video per request.")
        pixels = inputs.videos[index]
        if pixels.ndim != 4 or pixels.shape[0] != 40 or pixels.shape[1] != 3:
            raise ValueError(f"Expected TCHW video, got {pixels.shape}")
        # vLLM's Qwen3VL processor consumes (THWC numpy array, metadata).
        inputs.videos[index] = (pixels.permute(0, 2, 3, 1).cpu().numpy(), metadata[0])
        inputs.mm_processor_kwargs.update(do_resize=False, do_sample_frames=False)
        return ["<|vision_start|><|video_pad|><|vision_end|>"]


def register():
    existing = TEMPLATE_MAPPING["qwen3_5"]
    if existing.template_cls is DemoVideoTemplate:
        return
    if existing.template_cls is not TraceVideoQwen3_5Template:
        raise RuntimeError("Unexpected qwen3_5 template registration.")
    template = copy.copy(existing)
    template.template_cls = DemoVideoTemplate
    register_template(template, exist_ok=True)


register()
