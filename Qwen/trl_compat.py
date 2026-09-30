"""统一处理 TRL 0.27 与 Transformers 5.9 的可选依赖查询接口差异。"""

import trl.import_utils as trl_import_utils
from transformers.utils.import_utils import _is_package_available as transformers_package_available


def _trl_package_available(package_name, return_version=False):
    # 沿用原 grpo_logging.py 的导入补丁；不安装未启用的可选训练后端。
    result = transformers_package_available(package_name, return_version=return_version)
    return result if return_version else result[0]


trl_import_utils._is_package_available = _trl_package_available
