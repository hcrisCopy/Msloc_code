"""给 ms-swift 4.5.3 的逐条 GRPO 生成记录附上原 proposal ID。"""

from collections import deque

# TRL 0.27 把 Transformers 5.9 的 (available, version) 返回值直接当布尔值，
# 导致未安装的可选包（例如 weave）也被导入。保留查询版本时的二元组接口。
import trl.import_utils as trl_import_utils
from transformers.utils.import_utils import _is_package_available as transformers_package_available


def _trl_package_available(package_name, return_version=False):
    result = transformers_package_available(package_name, return_version=return_version)
    return result if return_version else result[0]


trl_import_utils._is_package_available = _trl_package_available

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
