"""UniFormerV2-B/16 video and patch features with the existing DeMamba head."""
import json
from pathlib import Path

import torch

from .DeMamba import _DINOFullFeatureDeMamba
from .uniformerv2_backbone import VisionTransformer


BACKBONE = "uniformerv2_b16_k400_8x224"
FEATURE_LOCATION = "post_block_pre_global_dpe_patch_tokens_only"
MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)


def load_encoder(weights_path):
    """Load the official K400 B/16 checkpoint without partial weight loading."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"UniFormerV2 weights not found: {path}")
    encoder = VisionTransformer(
        input_resolution=224, patch_size=16, width=768, layers=12,
        heads=12, output_dim=512, t_size=8, temporal_downsample=False,
        no_lmhra=True, double_lmhra=True, return_list=[8, 9, 10, 11],
        n_layers=4, n_dim=768, n_head=12, mlp_factor=4.0,
        backbone_drop_path_rate=0.0, drop_path_rate=0.0,
        mlp_dropout=[0.5] * 4, cls_dropout=0.5, num_classes=400,
        use_checkpoint=False, frozen=False,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a checkpoint dictionary: {path}")
    # SlowFast training saves model_state; released model-zoo weights may
    # contain the bare state dictionary. These are explicit upstream formats.
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Invalid official UniFormerV2 checkpoint: {path}")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if not all(key.startswith("backbone.") for key in state):
        raise ValueError("Expected official checkpoint keys prefixed with 'backbone.'")
    state = {key.removeprefix("backbone."): value for key, value in state.items()}
    encoder.load_state_dict(state, strict=True)
    # Retain DPE, G-MHRA and the balance gate. Only the original action
    # classification head is unused when returning pre-classifier features.
    encoder.transformer.proj.requires_grad_(False)
    return encoder


class UniFormerV2_DeMamba(_DINOFullFeatureDeMamba):
    """Use the native global video feature and all final-block patch channels."""

    def __init__(self, uniformerv2_weights_path, class_num=4):
        super().__init__(input_size=224, class_num=class_num)
        self.encoder = load_encoder(uniformerv2_weights_path)

    def _patch_features(self, states):
        return states[-1]

    def forward(self, x):
        if x.ndim != 5 or tuple(x.shape[1:]) != (8, 3, 224, 224):
            raise ValueError(f"Expected [B,8,3,224,224], got {tuple(x.shape)}")
        batch_size, num_frames = x.shape[:2]
        pooled, states = self.encoder(x.permute(0, 2, 1, 3, 4), return_features=True)
        features = self._patch_features(states).to(dtype=self.fc1.weight.dtype)
        if pooled.shape != (batch_size, 768) or features.shape != (batch_size, num_frames, 196, 768):
            raise RuntimeError(f"Unexpected video/patch shapes: {tuple(pooled.shape)}, {tuple(features.shape)}")

        global_feat = self.fc_norm2(pooled.to(dtype=self.fc1.weight.dtype))
        # Preserve the existing DeMamba local branch and classifier.
        mamba_input = features.reshape(batch_size, num_frames, 1, 14, 1, 14, 768)
        mamba_input = mamba_input.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        mamba_input = mamba_input.reshape(batch_size * 196, num_frames, 768)
        local_feat = self.mamba(mamba_input).mean(dim=1).reshape(batch_size, -1)
        local_feat = self.fc_norm(local_feat)
        return self.dropout(self.fc1(torch.cat((global_feat, local_feat), dim=1)))


class UniFormerV2_NeuronDeMamba(UniFormerV2_DeMamba):
    """Select 64 patch channels per block; keep the same native global branch."""

    def __init__(self, uniformerv2_weights_path, neuron_indices_path, class_num=4):
        super().__init__(uniformerv2_weights_path, class_num)
        selector = json.loads(Path(neuron_indices_path).read_text(encoding="utf-8"))
        expected = {"backbone": BACKBONE, "feature_location": FEATURE_LOCATION,
                    "final_neuron_count": 768, "neurons_per_layer": 64}
        for key, value in expected.items():
            if selector.get(key) != value:
                raise ValueError(f"UniFormerV2 selector {key} must be {value!r}")
        layers = selector["layers"]
        if set(layers) != {str(layer) for layer in range(1, 13)}:
            raise ValueError("UniFormerV2 selector must contain exactly layers 1 through 12")
        for layer in range(1, 13):
            values = layers[str(layer)]
            if (not isinstance(values, list) or len(values) != 64
                    or any(type(value) is not int or not 0 <= value < 768 for value in values)
                    or len(set(values)) != 64):
                raise ValueError(f"Layer {layer} requires 64 unique integer channels in [0,767]")
            self.register_buffer(f"neuron_indices_l{layer}", torch.tensor(values, dtype=torch.long))

    def _patch_features(self, states):
        if len(states) != 12:
            raise RuntimeError(f"Expected 12 UniFormerV2 block outputs, got {len(states)}")
        return torch.cat([
            state.index_select(-1, getattr(self, f"neuron_indices_l{layer}"))
            for layer, state in enumerate(states, 1)
        ], dim=-1)
