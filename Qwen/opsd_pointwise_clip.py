"""为 ms-swift 4.5.3 GKD 加入 OPSD 作者的逐词表项正向 KL 裁剪。"""

from __future__ import annotations

import os
from importlib.metadata import version

import torch.nn.functional as F


def register_opsd_pointwise_clip() -> None:
    # 外部插件路径会随 LoRA 写入 args.json；非 OPSD 训练进程导入时无需改动损失。
    raw_clip = os.environ.get("MSLOC_OPSD_POINTWISE_CLIP")
    if raw_clip is None:
        return
    if version("ms-swift") != "4.5.3":
        raise RuntimeError("OPSD 逐词表项裁剪仅核对了 ms-swift 4.5.3 的 GKD 接口")
    clip = float(raw_clip)
    if clip < 0:
        raise ValueError("MSLOC_OPSD_POINTWISE_CLIP 不能小于 0")
    if clip == 0:
        print("OPSD forward KL: pointwise clipping disabled", flush=True)
        return

    from swift.rlhf_trainers import gkd_trainer
    from swift.rlhf_trainers.gkd_loss import gkd_loss

    if gkd_trainer.gkd_loss is not gkd_loss:
        raise RuntimeError("ms-swift 的 GKD 损失入口已变化，不能安全加入 OPSD 裁剪")

    def clipped_kl(input_log_probs, target_log_probs):
        # 与 OPSD 作者的 F.kl_div(..., reduction='none', log_target=True).clamp(max=tau)
        # 相同：先裁剪每个生成位置、每个词表项的贡献，再对词表求和。
        contributions = F.kl_div(input_log_probs, target_log_probs, reduction="none", log_target=True)
        return contributions.clamp(max=clip).sum(dim=-1)

    def clipped_gkd_loss(student_logits, teacher_output, labels, beta, temperature):
        if beta != 0:
            raise ValueError("此插件只实现正向 KL；OPSD 的 beta 必须为 0")
        if teacher_output.is_topk_mode:
            raise ValueError("OPSD 逐词表项裁剪需要完整词表教师 logits")
        return gkd_loss(
            student_logits, teacher_output, labels, beta, temperature, kl_div_fn=clipped_kl,
        )

    gkd_trainer.gkd_loss = clipped_gkd_loss
    print(f"OPSD forward KL: full-vocabulary pointwise clip={clip:g}", flush=True)


register_opsd_pointwise_clip()
