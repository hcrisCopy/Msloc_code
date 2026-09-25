# SFT 重复输出小实验

复用 `sft_eval/predictions.json` 中的 proposal 和已生成的 40 帧片段。同一权重、提示词和 256 token 上限下，对 20 条原格式错误、10 条应为伪造且输出有效区间的回答、10 条应为 Real 却输出伪造区间的回答，分别比较原来的贪心解码与 [Qwen3.5 官方非 thinking 采样参数](https://huggingface.co/Qwen/Qwen3.5-4B)。格式错误组包含重复最明显的一条，其余按 seed 42 抽样。当前 ms-swift `TransformersEngine` 不执行 `presence_penalty`，这里仅比较它实际支持的 `temperature`、`top_p` 和 `top_k`；正式评测代码不受影响。

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
