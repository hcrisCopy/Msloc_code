"""给 ms-swift 4.5.3 的逐条 GRPO 生成记录附上原 proposal ID。"""

from collections import deque

from swift.rlhf_trainers.grpo_trainer import GRPOTrainer


_original_log_rollout = GRPOTrainer._log_rollout


def _log_rollout_with_sample_id(self, samples):
    # 与 ms-swift 的 prompt/completion 使用同一次多卡汇总、相同的 rank 顺序。
    sample_ids = []
    for sample in samples:
        sample_id = sample.extra.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("GRPO rollout 缺少有效 sample_id")
        sample_ids.append(sample_id)

    _original_log_rollout(self, samples)
    gathered_ids = self._gather_and_flatten(sample_ids, flatten_level=0)
    if "sample_id" not in self._logs:
        self._logs["sample_id"] = deque(maxlen=self.args.generation_batch_size)
    self._logs["sample_id"].extend(gathered_ids)


GRPOTrainer._log_rollout = _log_rollout_with_sample_id
