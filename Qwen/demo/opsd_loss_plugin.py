"""Bound OPSD JSD intermediates while retaining swift's full-vocabulary loss.

References:
https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/gkd_loss.py
https://docs.pytorch.org/docs/2.10/checkpoint.html

Only the token axis is split. Each block calls the installed swift function;
non-reentrant checkpointing recomputes its intermediates during backward.
"""

import argparse
import functools
import importlib
import inspect
import os

import torch
from torch.utils.checkpoint import checkpoint


def chunked_jsd(original, token_chunk):
    if token_chunk <= 0:
        raise ValueError("JSD token chunk must be positive.")
    expected = {"s_logits", "t_logits", "beta", "log_softmax_fn", "kl_div_fn", "chunk_size"}
    if not expected.issubset(inspect.signature(original).parameters):
        raise RuntimeError("Installed swift jsd_loss signature differs from the supported implementation.")

    @functools.wraps(original)
    def compute(s_logits, t_logits, beta, log_softmax_fn=None, kl_div_fn=None, chunk_size=512):
        if s_logits.ndim != 2 or s_logits.shape != t_logits.shape:
            raise ValueError("Expected aligned student/teacher logits with shape [tokens, vocabulary].")
        block = min(token_chunk, chunk_size)
        if block <= 0:
            raise ValueError("swift JSD chunk_size must be positive.")
        kwargs = {"beta": beta, "chunk_size": block}
        if log_softmax_fn is not None:
            kwargs["log_softmax_fn"] = log_softmax_fn
        if kl_div_fn is not None:
            kwargs["kl_div_fn"] = kl_div_fn
        loss_block = functools.partial(original, **kwargs)
        total = s_logits.new_zeros(())
        for start in range(0, s_logits.shape[0], block):
            student = s_logits[start:start + block]
            teacher = t_logits[start:start + block]
            if torch.is_grad_enabled() and (student.requires_grad or teacher.requires_grad):
                value = checkpoint(loss_block, student, teacher, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                value = loss_block(student, teacher)
            total = total + value
        # Keep swift's summed loss; its trainer divides by the valid token count.
        return total

    compute._msloc_jsd_chunk = token_chunk
    return compute


def loss_module():
    from demo_utils import check_versions
    check_versions()
    import trl_compat  # noqa: F401
    module = importlib.import_module("swift.rlhf_trainers.gkd_loss")
    trainer = importlib.import_module("swift.rlhf_trainers.gkd_trainer")
    if trainer.gkd_loss is not module.gkd_loss or module.gkd_loss.__globals__ is not vars(module):
        raise RuntimeError("Installed GKD trainer does not use the expected shared loss module.")
    return module


def verify_equivalence(module, optimized):
    """Compare the installed full loss and student gradients, including OPSD masks.

    Tiny CPU tensors are intentional: this checks math before loading GPU models.
    Fork the RNG so this check cannot change the training sampling sequence.
    """
    original = module.jsd_loss
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        for dtype in (torch.float32, torch.bfloat16):
            for count in (1, 33, 65, 256):
                # Teacher prefixes are longer; valid response positions still align.
                s_data = torch.randn(1, count + 3, 257, dtype=dtype, device="cpu")
                t_data = torch.randn(1, count + 7, 257, dtype=dtype, device="cpu")
                labels = torch.full((1, count + 3), -100, dtype=torch.long, device="cpu")
                teacher_labels = torch.full((1, count + 7), -100, dtype=torch.long, device="cpu")
                labels[:, 3:] = 1
                teacher_labels[:, 7:] = 1
                for temperature in (1.0, 1.7):
                    results = []
                    try:
                        for implementation in (original, optimized):
                            module.jsd_loss = implementation
                            student = s_data.clone().requires_grad_(True)
                            teacher = module.TeacherOutput(full_logits=t_data, labels=teacher_labels)
                            total, valid = module.gkd_loss(student, teacher, labels, 0.5, temperature)
                            mean = total / valid
                            mean.backward()
                            results.append((mean.detach().float(), student.grad.detach().float(), valid))
                    finally:
                        module.jsd_loss = original
                    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
                    torch.testing.assert_close(results[0][0], results[1][0], rtol=tolerance, atol=1e-6)
                    torch.testing.assert_close(results[0][1], results[1][1], rtol=tolerance, atol=2e-6)
                    if int(results[0][2]) != count or int(results[1][2]) != count:
                        raise AssertionError("JSD normalization changed the valid response token count.")


def install(token_chunk):
    module = loss_module()
    previous = getattr(module.jsd_loss, "_msloc_jsd_chunk", None)
    if previous is not None:
        if previous != token_chunk:
            raise RuntimeError("JSD plugin was already installed with a different chunk size.")
        return
    optimized = chunked_jsd(module.jsd_loss, token_chunk)
    verify_equivalence(module, optimized)
    module.jsd_loss = optimized
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[Demo] JSD: full vocabulary, {token_chunk} response tokens/block, "
              "checkpointed intermediates; CPU loss/gradient comparison passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    install(args.chunk_size)
elif "MSLOC_DEMO_JSD_CHUNK_SIZE" in os.environ:
    install(int(os.environ["MSLOC_DEMO_JSD_CHUNK_SIZE"]))
