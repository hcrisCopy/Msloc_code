"""ms-swift 4.5.3 GRPO 奖励插件；参考 Trace，解释分只奖励标注覆盖。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from swift.rewards import ORM, orms

from common import parse_answer


def _load_trace_reward_file(filename: str):
    """只加载奖励实现，避免执行 Trace 包初始化及其旧模型依赖。"""
    module_name = f"msloc_trace_reward_{Path(filename).stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).resolve().parents[1] / "Trace" / "trace" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 Trace 奖励文件：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclass 定义时需要能找到所属模块
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[module_name]
        raise
    return module


_temporal = _load_trace_reward_file("opd_grpo.py")
_text = _load_trace_reward_file("text_explanation_reward.py")
boundary_score = _temporal.boundary_score
temporal_iou = _temporal.temporal_iou
EntailmentExplanationJudge = _text.EntailmentExplanationJudge


def _rows(completions, kwargs):
    fields = ("target_kind", "target_relative", "clip_duration", "evidence")
    for key in fields:
        if key not in kwargs or len(kwargs[key]) != len(completions):
            raise ValueError(f"GRPO 奖励缺少逐 completion 字段：{key}")
    for index, completion in enumerate(completions):
        if not isinstance(completion, str):
            raise TypeError(f"GRPO completion 不是文本：{type(completion).__name__}")
        kind = kwargs["target_kind"][index]
        target = kwargs["target_relative"][index]
        duration = float(kwargs["clip_duration"][index])
        evidence = kwargs["evidence"][index]
        if kind not in ("fake", "real") or duration <= 0:
            raise ValueError("GRPO 目标类型或片段时长无效")
        if (kind == "real" and target is not None) or (kind == "fake" and (not isinstance(target, list) or len(target) != 2)):
            raise ValueError("GRPO 目标区间与真假标签不一致")
        if kind == "fake" and not (0 <= target[0] < target[1] <= duration + 0.001):
            raise ValueError("GRPO 目标区间超出片段")
        if not isinstance(evidence, dict) or (kind == "fake" and not evidence.get("object_caption")):
            raise ValueError("GRPO 解释证据缺失")
        yield parse_answer(completion, duration), kind, target, evidence


class LocalizationReward(ORM):
    def __call__(self, completions, **kwargs):
        scores = []
        for parsed, kind, target, _ in _rows(completions, kwargs):
            status = parsed["status"]
            if kind == "real":
                scores.append(1.0 if status == "real" else -1.0 if status == "fake" else -0.5)
            elif status != "fake":
                scores.append(-1.0)
            else:
                predicted = tuple(parsed["relative_segment"])
                expected = tuple(target)
                iou = temporal_iou(predicted, expected)
                # 单区间相当于 Trace 的一次匹配：无交集时，该预测是 unmatched。
                matched_boundary = boundary_score(predicted, expected, 1.0) if iou > 0 else 0.0
                unmatched = 0 if iou > 0 else 1
                scores.append(0.25 + 0.55 * iou + 0.20 * matched_boundary - 0.25 * unmatched)
        return scores


class FormatReward(ORM):
    def __call__(self, completions, **kwargs):
        return [1.0 if parsed["status"] in ("real", "fake") else -1.0
                for parsed, _, _, _ in _rows(completions, kwargs)]


class ExplanationReward(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.judge = None

    def __call__(self, completions, **kwargs):
        scores = []
        for parsed, kind, target, evidence in _rows(completions, kwargs):
            if kind != "fake" or parsed["status"] != "fake":
                scores.append(0.0)
                continue
            iou = temporal_iou(tuple(parsed["relative_segment"]), tuple(target))
            if iou < 0.30:
                scores.append(0.0)
                continue
            if self.judge is None:
                model = os.environ.get("MSLOC_GRPO_NLI_MODEL")
                if not model:
                    raise RuntimeError("未配置 GRPO 的本地 NLI 模型")
                self.judge = EntailmentExplanationJudge(
                    nli_model_path=model,
                    nli_device="cpu",
                    nli_batch_size=32,
                )
            verdict = self.judge.score(
                caption=parsed["explanation"],
                # 旧版已准备的数据可能含类别字段；NLI 只比较标注原句。
                evidence=SimpleNamespace(
                    object_caption=evidence["object_caption"],
                    start_caption=evidence["start_caption"],
                    end_caption=evidence["end_caption"],
                ),
            )
            # 标注可能不完整：额外句子没有匹配到标注，不等于视频解释错误。
            # 保留标注覆盖和与已匹配事实的矛盾惩罚，不使用按生成句数归一的 precision。
            scores.append(verdict.graph_recall - 0.50 * verdict.contradiction)
        return scores


orms["msloc_localization"] = LocalizationReward
orms["msloc_format"] = FormatReward
orms["msloc_explanation"] = ExplanationReward
