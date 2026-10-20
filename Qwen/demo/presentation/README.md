# 从视频到参数更新：讲解网页

[返回主 README](../README.md#navigation)

成品：`../MSLoc_data/Qwen/demo/presentation/index.html`。下载这个文件，用 Chrome 或 Edge 打开即可。视频、原始回答、公式、图表、样式和交互全部内嵌，不依赖网络。打印会保留当前选中的样本和已展开的内容；完整展示使用 HTML。

网页按“任务 → 训练后推理对比 → 视频输入 → SFT → 教师 → OPSD → GRPO → 实测结果 → 加速”展开。完整提示词、原始回答、关键代码和公式都保留；代码和详细记录可以展开查看。

- 推理对比：三个带视频的例子，以及全部 64 条预测的筛选、区间指标和原始回答。
- 输入与训练：16/8/16 取帧、Qwen 时间模板、fake/real 师生提示、SFT label、正向 KL / 反向 KL / JSD、GRPO 四条真实 rollout 与奖励/优势。
- 结果与加速：已有 loss 曲线、单卡速度和显存；可切换 OPSD / GRPO 的 Ray 角色流程及权重同步说明。Ray 多卡 GPU 尚未实测。

主例按 OPSD 定位增量选取，要求起点有有效区间且 GRPO HF 不退步；补例按 GRPO HF 增量选取；反例取仍误判的 real。选择规则写在页面中。64 条都是训练 demo 数据，不是测试集。SFT 对比起点是已有正式权重，独立 demo SFT 曲线另列。

已逐条核对两个 GRPO 分支：都没有 SFT→OPSD→GRPO 连续两次提高 IoU 的例子，也没有纠正 real 误报的例子；各有 2 条 fake 在 GRPO 后比 OPSD 定位更好。网页每次重建都会重新计算这些数量，不补造改善样本。

32 步指 32 次参数更新。HF / vLLM 指 **GRPO 训练时的生成后端**：两次训练得到两份权重，最终都由 Transformers 推理；不是同一权重切换推理引擎的对比。

## 重建

沿用 `msloc_qwen35`，**不新增环境依赖**。在服务器的 `Msloc_code/` 下执行：

```bash
conda activate msloc_qwen35
python Qwen/demo/presentation/build.py \
  --root ../MSLoc_data/Qwen/demo \
  --model ../Qwen/Qwen3.5-4B \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --output ../MSLoc_data/Qwen/demo/presentation/index.html \
  --overwrite
```

只读取已有预测、训练日志、图表和视频；不会训练或修改权重。CPU 调用官方 processor，生成真实模板文本和 SFT token label 展示。需要已有的 `student_before`、`teacher_before`、`student_after`、两种 GRPO 回载预测、四次训练记录以及 `grpo_compare`。缺文件或数据不一致会报错。

`--overwrite` 只重建指定网页及同名 `.audit.json`，不清理训练结果。输出先写临时文件，成功后替换；中断后重跑该命令即可。审计文件记录样本、统计指标、改善筛选结果和输入文件 SHA256。

生成器放在 `demo/presentation/` 子目录，独立于训练代码；没有改变已有训练/推理版本检查规则。
