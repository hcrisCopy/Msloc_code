"""MSLoc-Qwen3.5：在官方 Qwen3.5 主干上保留 DAM、EAM 与 LAA。

视觉编码严格调用 Transformers 官方 ``Qwen3_5ForConditionalGeneration`` 的
``get_image_features``。DAM/EAM 的 slot pooling 结构改编自本仓库现有
``Trace/trace/model/multimodal_projector/builder.py``，只把输入/输出维度适配为
Qwen3.5 的视觉输出维度和文本隐藏维度。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast


IGNORE_INDEX = -100


def require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        finite = tensor[torch.isfinite(tensor)]
        finite_range = (
            f"min={finite.min().item():.6g}, max={finite.max().item():.6g}"
            if finite.numel()
            else "没有有限值"
        )
        raise FloatingPointError(
            f"{name} 出现 NaN/Inf：shape={tuple(tensor.shape)}, dtype={tensor.dtype}, {finite_range}"
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class SlotRotaryEmbedding(nn.Module):
    """与原 RefProjector 一致的轻量 RoPE，用于 slot pooling。"""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("slot RoPE 维度必须是正偶数")
        self.dim = dim
        self.base = float(base)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # 不能把 inv_freq 作为非持久 buffer 依赖：Transformers 的
        # from_pretrained 低内存加载会在 meta 设备构造模型，而该 buffer
        # 不在 checkpoint 中，可能无法得到有效实体值。它是固定公式，直接
        # 在当前设备确定性重建最可靠，也与原 TRACE 的 RoPE 数值定义一致。
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=positions.device, dtype=torch.float32) / self.dim)
        )
        angles = torch.outer(positions.float(), inv_freq)
        embeddings = torch.cat((angles, angles), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, rotary: SlotRotaryEmbedding) -> torch.Tensor:
    cos, sin = rotary(positions, x.dtype)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    rotary_dim = cos.shape[-1]
    rotated = x[..., :rotary_dim] * cos + rotate_half(x[..., :rotary_dim]) * sin
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


class EventAggregationModule(nn.Module):
    """EAM：在时间与空间维度联合聚合事件区域 token。"""

    def __init__(self, vision_dim: int, text_dim: int, num_slots: int = 32, rope_dim: int = 128):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(vision_dim, num_slots) * (vision_dim ** -0.5))
        self.norm = nn.LayerNorm(vision_dim)
        self.readout = nn.Linear(vision_dim, text_dim, bias=False)
        actual_rope_dim = min(rope_dim, vision_dim)
        actual_rope_dim -= actual_rope_dim % 2
        self.rotary = SlotRotaryEmbedding(actual_rope_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """显式初始化后加模块，不能依赖 ``from_pretrained`` 的缺失权重初始化。"""
        nn.init.normal_(self.slots, mean=0.0, std=self.slots.shape[0] ** -0.5)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.xavier_uniform_(self.readout.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, T, N, D]
        batch, frames, tokens, dim = features.shape
        flattened = features.reshape(batch, frames * tokens, dim)
        normalized = F.layer_norm(
            flattened.float(), self.norm.normalized_shape,
            self.norm.weight.float(), self.norm.bias.float(),
        ).to(flattened.dtype)
        require_finite("EAM归一化特征", normalized)
        positions = torch.arange(frames, device=features.device).repeat_interleave(tokens)
        normalized = apply_rope(normalized, positions, self.rotary)
        # 注意力 logits 与加权累加必须使用 float32。40 帧视觉特征在 bf16
        # 下直接做长维度归约会放大舍入误差，并可能产生非有限池化结果。
        normalized_unit = F.normalize(normalized.float(), p=2, dim=-1, eps=1e-6)
        slots_unit = F.normalize(self.slots.float(), p=2, dim=0, eps=1e-6)
        similarity = torch.matmul(normalized_unit, slots_unit)
        require_finite("EAM相似度", similarity)
        attention = torch.softmax(similarity, dim=1)
        require_finite("EAM注意力", attention)
        pooled = torch.matmul(normalized.float().transpose(1, 2), attention).transpose(1, 2)
        require_finite("EAM池化特征", pooled)
        output = self.readout(pooled.to(self.readout.weight.dtype))
        require_finite("EAM输出", output)
        return output


class DifferenceAwareModule(nn.Module):
    """DAM：逐帧压缩边界 token，并显式加入跨帧变化/稳定 token。"""

    def __init__(self, vision_dim: int, text_dim: int, num_slots: int = 8, rope_dim: int = 128):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(vision_dim, num_slots) * (vision_dim ** -0.5))
        self.norm = nn.LayerNorm(vision_dim)
        self.readout = nn.Linear(vision_dim, text_dim, bias=False)
        actual_rope_dim = min(rope_dim, vision_dim)
        actual_rope_dim -= actual_rope_dim % 2
        self.rotary = SlotRotaryEmbedding(actual_rope_dim)
        self.scale = vision_dim ** -0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """显式初始化后加模块，不能依赖 ``from_pretrained`` 的缺失权重初始化。"""
        nn.init.normal_(self.slots, mean=0.0, std=self.slots.shape[0] ** -0.5)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.xavier_uniform_(self.readout.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, T, N, D]
        batch, frames, tokens, dim = features.shape
        flattened = features.reshape(batch * frames, tokens, dim)
        normalized = F.layer_norm(
            flattened.float(), self.norm.normalized_shape,
            self.norm.weight.float(), self.norm.bias.float(),
        ).to(flattened.dtype)
        require_finite("DAM归一化特征", normalized)
        per_frame = normalized.reshape(batch, frames, tokens, dim)

        # 原 TRACE 将 LayerNorm 后的点积视为相似度。这里显式归一化为
        # 余弦相似度，保持空间 token 排序，同时让随机初始化阶段的 logits
        # 严格有界，避免高维原始点积在混合精度下产生非有限注意力。
        per_frame_unit = F.normalize(per_frame.float(), p=2, dim=-1, eps=1e-6)
        previous = torch.roll(per_frame_unit, shifts=1, dims=1)
        difference = -(per_frame_unit * previous).sum(dim=-1)
        require_finite("DAM跨帧差异", difference)
        motion_attention = torch.softmax(difference, dim=-1)
        stable_attention = torch.softmax(-difference, dim=-1)
        extra_attention = torch.stack((motion_attention, stable_attention), dim=-1)
        extra_attention = extra_attention.reshape(batch * frames, tokens, 2)

        positions = torch.arange(tokens, device=features.device)
        normalized = apply_rope(normalized, positions, self.rotary)
        normalized_unit = F.normalize(normalized.float(), p=2, dim=-1, eps=1e-6)
        slots_unit = F.normalize(self.slots.float(), p=2, dim=0, eps=1e-6)
        slot_similarity = torch.matmul(normalized_unit, slots_unit)
        require_finite("DAM slot相似度", slot_similarity)
        slot_attention = torch.softmax(slot_similarity, dim=1)
        all_attention = torch.cat((slot_attention, extra_attention), dim=-1)
        require_finite("DAM注意力", all_attention)
        pooled = torch.matmul(normalized.float().transpose(1, 2), all_attention).transpose(1, 2)
        pooled = pooled.reshape(batch, frames, pooled.shape[1], dim)
        require_finite("DAM池化特征", pooled)
        output = self.readout(pooled.to(self.readout.weight.dtype))
        require_finite("DAM输出", output)
        return output


class MSLocProjector(nn.Module):
    """按左边界、事件、右边界切分逐帧特征，并分别调用 DAM/EAM。"""

    def __init__(self, vision_dim: int, text_dim: int, bnd_frames: int, seg_frames: int):
        super().__init__()
        self.bnd_frames = bnd_frames
        self.seg_frames = seg_frames
        self.dam = DifferenceAwareModule(vision_dim, text_dim, num_slots=8)
        self.eam = EventAggregationModule(vision_dim, text_dim, num_slots=32)

    @property
    def output_tokens(self) -> int:
        # DAM 每帧输出 8 个 slot + motion/static 两个 token；EAM 固定输出 32 个。
        return 2 * self.bnd_frames * 10 + 32

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected = 2 * self.bnd_frames + self.seg_frames
        if features.ndim != 4 or features.shape[1] != expected:
            raise ValueError(f"MSLocProjector 需要 [B,{expected},N,D]，实际为 {tuple(features.shape)}")
        event_start = self.bnd_frames
        event_end = event_start + self.seg_frames
        left = self.dam(features[:, :event_start]).flatten(1, 2)
        event = self.eam(features[:, event_start:event_end])
        right = self.dam(features[:, event_end:]).flatten(1, 2)
        return torch.cat((left, right, event), dim=1)


class MSLocQwen35ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 + DAM/EAM + 三个 anomaly-aware token + LAA/CLoss。"""

    def __init__(self, config):
        class_names = list(getattr(config, "msloc_class_names", []) or [])
        if not class_names:
            raise ValueError("config.msloc_class_names 不能为空；请从 class_features_bge.pt 读取类别")
        super().__init__(config)
        text_dim = int(config.text_config.hidden_size)
        vision_dim = int(config.vision_config.out_hidden_size)
        bnd_frames = int(getattr(config, "msloc_bnd_frames", 16))
        seg_frames = int(getattr(config, "msloc_seg_frames", 8))
        self.msloc_projector = MSLocProjector(vision_dim, text_dim, bnd_frames, seg_frames)
        self.anomaly_tokens = nn.Parameter(torch.randn(3, text_dim) * 1e-4)
        self.closs_head = nn.Linear(text_dim, len(class_names), bias=False)
        nn.init.normal_(self.closs_head.weight, std=0.02)
        class_feature_dim = int(getattr(config, "msloc_class_feature_dim", 0))
        if class_feature_dim <= 0:
            raise ValueError("config.msloc_class_feature_dim 必须为正整数")
        # 与原 TRACE 一致保存固定类别特征库，保证 SFT/OPD/GRPO 类别语义和顺序一致。
        # 当前 LAA 的优化目标与原 forward 相同，使用可训练线性分类 head 计算交叉熵。
        self.register_buffer(
            "class_feature_bank",
            torch.zeros(len(class_names), class_feature_dim, dtype=torch.float32),
            persistent=True,
        )
        self.closs_weight = float(getattr(config, "msloc_closs_weight", 1.0))
        self.config.architectures = [self.__class__.__name__]

    def reset_msloc_parameters(self) -> None:
        """在基础 Qwen 权重加载完成后初始化 checkpoint 中不存在的 MSLoc 参数。"""
        self.msloc_projector.dam.reset_parameters()
        self.msloc_projector.eam.reset_parameters()
        nn.init.normal_(self.anomaly_tokens, mean=0.0, std=1e-4)
        nn.init.normal_(self.closs_head.weight, mean=0.0, std=0.02)

    def validate_msloc_parameters(self) -> None:
        """训练或推理前检查全部 MSLoc 参数，禁止非有限权重进入前向。"""
        for name, parameter in self.named_parameters():
            if name.startswith("msloc_projector.") or name in {"anomaly_tokens", "closs_head.weight"}:
                require_finite(f"MSLoc参数[{name}]", parameter)

    def _encode_frames(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        frame_counts: torch.Tensor,
    ) -> list[torch.Tensor]:
        outputs = self.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        per_frame = list(outputs.pooler_output)
        counts = [int(value) for value in frame_counts.tolist()]
        if sum(counts) != len(per_frame):
            raise ValueError(f"视觉帧数量不一致：{sum(counts)} != {len(per_frame)}")
        grouped: list[torch.Tensor] = []
        offset = 0
        for count in counts:
            current = per_frame[offset : offset + count]
            offset += count
            token_counts = {int(frame.shape[0]) for frame in current}
            if len(token_counts) != 1:
                raise ValueError(f"同一视频的逐帧视觉 token 数不同：{sorted(token_counts)}")
            grouped.append(torch.stack(current, dim=0))
        return grouped

    def _prepare_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        frame_counts: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        frame_features = self._encode_frames(pixel_values, image_grid_thw, frame_counts)
        for index, features in enumerate(frame_features):
            require_finite(f"Qwen视觉特征[{index}]", features)
        # 不同视频可具有不同空间 token 数；DAM/EAM 分别压缩后输出固定长度。
        visual_tokens = torch.cat(
            [self.msloc_projector(features.unsqueeze(0)) for features in frame_features], dim=0
        )
        require_finite("DAM/EAM输出", visual_tokens)
        anomaly_tokens = self.anomaly_tokens.to(visual_tokens.dtype).unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        text_embeddings = self.get_input_embeddings()(input_ids)
        require_finite("文本embedding", text_embeddings)
        inputs_embeds = torch.cat((visual_tokens, anomaly_tokens, text_embeddings), dim=1)
        require_finite("视觉文本混合输入", inputs_embeds)
        prefix_length = visual_tokens.shape[1] + anomaly_tokens.shape[1]
        prefix_mask = torch.ones(
            (attention_mask.shape[0], prefix_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        combined_mask = torch.cat((prefix_mask, attention_mask), dim=1)
        combined_labels = None
        if labels is not None:
            ignored = torch.full(
                (labels.shape[0], prefix_length), IGNORE_INDEX, dtype=labels.dtype, device=labels.device
            )
            combined_labels = torch.cat((ignored, labels), dim=1)
        return inputs_embeds, combined_mask, combined_labels, visual_tokens.shape[1]

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        frame_counts: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        closs_labels: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        # Trainer 可能传入仅供 loss 归一化使用的参数，不能继续传给 Qwen 文本骨干。
        kwargs.pop("num_items_in_batch", None)
        # GenerationMixin 会把 return_dict 放入 model_inputs；本层始终需要
        # 结构化输出，因此先移除调用方值，再统一向文本骨干传 True。
        kwargs.pop("return_dict", None)
        # GenerationMixin 的预填充和解码步直接走官方 Qwen3.5 文本骨干与缓存参数。
        # DAM/EAM 输出是抽象语义 slot，因此作为一维语言前缀，不伪装成原生图像 patch token。
        if pixel_values is None:
            if inputs_embeds is None and input_ids is None:
                raise ValueError("生成步必须提供 input_ids 或 inputs_embeds")
            # 必须经过 Qwen3_5Model.forward，由官方实现计算四路 M-RoPE
            # position_ids 和混合全注意力/线性注意力 mask。
            outputs = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=use_cache,
                return_dict=True,
                **kwargs,
            )
            return CausalLMOutputWithPast(
                logits=self.lm_head(outputs.last_hidden_state).float(),
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        if input_ids is None or attention_mask is None or image_grid_thw is None or frame_counts is None:
            raise ValueError("MSLoc SFT 必须同时提供文本、视觉和 frame_counts")
        inputs_embeds, combined_mask, combined_labels, visual_length = self._prepare_prefix(
            input_ids, attention_mask, pixel_values, image_grid_thw, frame_counts, labels
        )
        # 不直接调用 language_model；Qwen3.5 多模态外层负责为整段
        # 语义前缀 + 文本生成正确的四路 M-RoPE position_ids。
        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            use_cache=False if use_cache is None else use_cache,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        require_finite("Qwen语言骨干隐藏状态", hidden_states)
        logits = self.lm_head(hidden_states).float()
        require_finite("文本logits", logits)
        loss = None
        if combined_labels is not None:
            supervised_tokens = int(combined_labels[:, 1:].ne(IGNORE_INDEX).sum().item())
            if supervised_tokens <= 0:
                raise ValueError("当前 batch 没有任何 assistant 监督 token")
            text_loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                combined_labels[:, 1:].contiguous().view(-1).to(logits.device),
                ignore_index=IGNORE_INDEX,
            )
            if closs_labels is None or closs_labels.shape != (input_ids.shape[0], 3):
                raise ValueError("启用 MSLoc SFT 时 closs_labels 必须为 [batch, 3]")
            anomaly_hidden = hidden_states[:, visual_length : visual_length + 3]
            class_logits = self.closs_head(anomaly_hidden).float()
            require_finite("CLoss logits", class_logits)
            class_loss = F.cross_entropy(
                class_logits.reshape(-1, class_logits.shape[-1]),
                closs_labels.reshape(-1).to(class_logits.device),
                ignore_index=IGNORE_INDEX,
            )
            loss = text_loss + self.closs_weight * class_loss
            if not torch.isfinite(text_loss) or not torch.isfinite(class_loss) or not torch.isfinite(loss):
                raise FloatingPointError(
                    f"SFT loss 出现 NaN/Inf：text_loss={text_loss.item()}，"
                    f"closs={class_loss.item()}，loss={loss.item()}"
                )
            self.last_text_loss = text_loss.detach()
            self.last_closs = class_loss.detach()
            self.last_total_loss = loss.detach()
            self.last_supervised_tokens = supervised_tokens
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.inference_mode()
    def generate_msloc(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        frame_counts: torch.Tensor,
        max_new_tokens: int,
        eos_token_ids: set[int],
        pad_token_id: int,
    ) -> torch.Tensor:
        if input_ids.shape[0] != 1:
            raise ValueError("generate_msloc 当前严格要求 batch-size=1")
        inputs_embeds, combined_mask, _, _ = self._prepare_prefix(
            input_ids, attention_mask, pixel_values, image_grid_thw, frame_counts
        )
        if not eos_token_ids:
            raise ValueError("eos_token_ids 不能为空")
        if pad_token_id < 0:
            raise ValueError("pad_token_id 必须是非负整数")
        # 使用 Transformers 官方 GenerationMixin，由它统一管理 Qwen3.5
        # 混合线性注意力缓存、cache_position 和停止条件。
        return self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=sorted(eos_token_ids),
            pad_token_id=pad_token_id,
            use_cache=True,
        )
