# SFT 重复输出小实验

复用旧版贪心评测 `sft_eval/predictions.json` 中的 proposal 和已生成的 40 帧片段。同一权重、提示词和 256 token 上限下，对 20 条原格式错误、10 条应为伪造且输出有效区间的回答、10 条应为 Real 却输出伪造区间的回答，分别比较原来的贪心解码与 [Qwen3.5 官方非 thinking 采样参数](https://huggingface.co/Qwen/Qwen3.5-4B)。格式错误组包含重复最明显的一条，其余按 seed 42 抽样。当前 ms-swift `TransformersEngine` 不执行 `presence_penalty`，这里只比较它实际支持的 `temperature`、`top_p` 和 `top_k`。正式评测现已统一采用相同采样参数；复跑本实验需要保留旧版贪心评测目录。

在项目代码根目录运行：

```bash
python Qwen/experiments/repetition/compare_decoding.py \
  --source-eval ../MSLoc_data/Qwen/sft_eval \
  --output ../MSLoc_data/Qwen/repetition_experiment \
  --devices 0 \
  --format-errors 20 \
  --positive-controls 10 \
  --negative-controls 10 \
  --max-new-tokens 256 \
  --resume none \
  --clean
```

`summary.json` 比较有效输出、正确真假判定、伪造片段的平均区间 IoU、重复八词片段和因长度停止的数量；`comparisons.jsonl` 保留每条旧输出和两组新输出的完整原文。中断后去掉 `--clean` 并改用 `--resume auto`。这只用于排查生成重复，不能代替完整测试集指标。

## 现象与结论

40 条样本中，贪心解码与原评测输出 40/40 一致。改用采样后，重复输出由 24 条降至 6 条，格式错误由 20 条降至 4 条，生成到 256 token 上限的回答由 20 条降至 4 条。另有 5 条回答的区间终点比片段时长多 0.017–0.032 秒，被判为越界；10 条原本有效的异常回答中有 2 条变成格式错误。15 条目标为 Real 的样本中只有 1 条正确输出 Real，符合 SFT 只用异常片段训练后的预期偏向。

采样能明显缓解解释重复，但不保证每条回答都改善。提示词已明确要求区间终点不超过片段时长；这 5 条是生成数字时轻微超出边界，不能仅据此归因于提示词。当前正式评测统一使用这组采样参数；是否提升完整测试集的定位指标，仍需以完整评测为准。256 token 暂不增加：采样后触及上限的 4 条中，3 条仍在重复解释。
