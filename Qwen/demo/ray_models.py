"""HF model utilities and losses, following pinned ms-swift 4.5.3 implementations."""

import copy
import math

import torch

from demo_utils import read_json
import video_plugin  # noqa: F401


def build_model(config, path, adapter="none", trainable=False):
    from swift.model import get_model_processor
    from swift import get_template
    from peft import LoraConfig, PeftModel, get_peft_model
    from peft.tuners.lora import LoraLayer
    if read_json(f"{path}/config.json").get("quantization_config"):
        raise ValueError("The Ray weight bridge requires unquantized dense HF weights.")
    model, processor = get_model_processor(path, model_type="qwen3_5", torch_dtype=torch.bfloat16,
                                          attn_impl="flash_attn", device_map={"": "cuda:0"})
    if trainable and config["tuner"] == "lora" and adapter == "none":
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and "language_model" in name and "lm_head" not in name]
        if not targets:
            raise ValueError("No Qwen3.5 language linear layers found.")
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=config["lora_rank"],
                              lora_alpha=config["lora_alpha"], lora_dropout=0.0, target_modules=targets))
    elif adapter != "none":
        cfg = read_json(f"{adapter}/adapter_config.json")
        if cfg.get("modules_to_save") or cfg.get("use_dora") or cfg.get("bias", "none") != "none":
            raise ValueError("The synchronous weight bridge supports ordinary language LoRA without DoRA/modules_to_save/bias.")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    if any(isinstance(module, LoraLayer) and "visual" in name for name, module in model.named_modules()):
        raise ValueError("Visual LoRA is outside this demo's supported weight bridge.")
    if not trainable:
        model.requires_grad_(False)
    else:
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad_(False)
        if any(p.requires_grad for n, p in model.named_parameters() if "visual" in n):
            raise RuntimeError("Visual parameters must remain frozen.")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.config.use_cache = False
    model.eval()
    template = get_template(processor, enable_thinking=False)
    template.set_mode("transformers")
    template.model = model
    template.register_post_encode_hook([model])
    return model, processor, template


def encode_prompt(template, row, teacher=False):
    data = {"messages": copy.deepcopy(row["messages"]), "videos": row["videos"],
            "chat_template_kwargs": row["chat_template_kwargs"]}
    if teacher:
        data["messages"] = [{"role": "user", "content": row["teacher_prompt"]}]
    return template.encode(data)


def forward_inputs(template, row, tokens, max_length, teacher=False):
    """Append actual sampled token IDs; never decode/re-tokenize a training target."""
    encoded = encode_prompt(template, row, teacher)
    length = len(encoded["input_ids"])
    if not tokens or length + len(tokens) > max_length:
        raise ValueError("Empty completion or input longer than max-length.")
    encoded["input_ids"] = list(encoded["input_ids"]) + tokens
    encoded["labels"] = [-100] * length + tokens
    if encoded.get("mm_token_type_ids") is not None:
        encoded["mm_token_type_ids"] = list(encoded["mm_token_type_ids"]) + [0] * len(tokens)
    if encoded.get("attention_mask") is not None:
        encoded["attention_mask"] = [1] * (length + len(tokens))
    encoded.pop("loss_scale", None)
    batch = template.data_collator([encoded])
    batch.pop("labels", None)
    batch.pop("loss_scale", None)
    return {key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}, length


def response_logits(model, template, row, tokens, max_length, teacher=False):
    batch, prefix = forward_inputs(template, row, tokens, max_length, teacher)
    # Qwen3.5 supports an index tensor: materialize only logits predicting response tokens.
    positions = torch.arange(prefix - 1, prefix + len(tokens) - 1, device="cuda:0")
    logits = model(**batch, logits_to_keep=positions, use_cache=False).logits[0]
    if logits.shape[0] != len(tokens):
        raise ValueError("Response logits and sampled token IDs are not aligned.")
    return logits


def selected_logps(logits, tokens, temperature=1.0):
    indices = torch.tensor(tokens, device=logits.device)
    scaled = logits.float() / temperature
    return scaled.gather(-1, indices[:, None]).squeeze(-1) - scaled.logsumexp(-1)


def opsd_loss(student, teacher):
    from swift.rlhf_trainers.gkd_loss import jsd_loss
    if student.shape != teacher.shape:
        raise ValueError("Teacher/student full vocabulary logits have different shapes.")
    return jsd_loss(student, teacher.detach(), beta=0.5) / student.shape[0]


def grpo_loss(logps, reference, advantage, old_logps=None):
    """Token-level clipped GRPO, matching swift's k3 stability clamps and sequence mean."""
    old = logps.detach() if old_logps is None else old_logps.detach()
    ratio = (logps - old).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
    difference = (reference.detach() - logps).clamp(-20, 20)
    kl = (difference.exp() - difference - 1).clamp(-10, 10)
    return (policy + 0.02 * kl).mean(), kl.mean().detach()


def learning_rate_factor(step, total):
    warmup = math.ceil(total * 0.03)
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def effective_weights(model, names=None):
    """Produce HF-format tensors without in-place LoRA merge/unmerge rounding drift."""
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    peft = isinstance(model, PeftModel)
    deltas = {}
    if peft:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                deltas[name.removeprefix("base_model.model.") + ".weight"] = module
    for raw_name, tensor in model.state_dict().items():
        if peft and ("lora_" in raw_name or "original_module" in raw_name):
            continue
        name = raw_name.removeprefix("base_model.model.") if peft else raw_name
        name = name.replace(".base_layer", "")
        if (name == "lm_head.weight"
                and model.get_output_embeddings().weight is model.get_input_embeddings().weight):
            continue
        if names is not None and name not in names:
            continue
        if name in deltas:
            delta = deltas[name].get_delta_weight("default")
            tensor = (tensor.to(delta.dtype) + delta).to(tensor.dtype)
        yield name, tensor.detach()


def trainable_probe(model):
    return [(name, p.detach().reshape(-1)[:32].float().cpu())
            for name, p in model.named_parameters() if p.requires_grad]
