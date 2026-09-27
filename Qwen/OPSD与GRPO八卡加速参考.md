# OPSD 与 GRPO：八卡加速参考

本文供单机八卡改造参考。这里的 **vLLM 负责加快学生生成回答**，**Ray 负责安排训练、生成、教师等进程使用哪些 GPU**；参数更新仍由 PyTorch 完成。使用 Ray 本身不等于训练更快。

## 目前OPSD与GRPO代码实现

当前代码使用 ms-swift：OPSD 学生从 SFT LoRA 继续训练，学生在线生成，固定的教师在学生已经生成的前缀上提供逐 token 分布，使用完整词表的 JSD；GRPO 可从 OPSD 或 SFT LoRA 开始，对每条输入采样 4 个回答，计算定位、格式和解释三项奖励。两阶段的学生都看 40 帧视频，按前 20%／中间 60%／后 20% 取 16／8／16 帧，帧的**真实相对时间**另存于 `.timestamps.json`。目前两个训练脚本都设置 `--use_vllm false`，视频在 Transformers 路径处理。见 [OPSD 训练脚本](train_opsd.py)、[GRPO 训练脚本](train_grpo.py)、[视频模板](trace_video_template.py)。

## 两个加速工程参考（均不能直接照跑本任务）

| 项目 | 适合借鉴什么 | 与目前代码实现的距离 |
| --- | --- | --- |
| [ms-swift](https://github.com/modelscope/ms-swift) | 继续沿用现有数据、LoRA、GKD 和 GRPO 训练器，把学生生成换成 vLLM；官方也提供 Megatron＋Ray 的 GRPO/GKD 示例 | **最近**。先考虑它的 vLLM 路径；视频对齐仍须自行验证。 |
| [verl](https://github.com/verl-project/verl) | 参考 Ray 如何分别管理学生训练、vLLM 生成和固定教师；有视觉语言模型的在线蒸馏及 GRPO 启动脚本 | **改动较大**。示例使用图片而非我们的视频；蒸馏示例默认给学生 8 张 GPU、给独立教师另配 4 张，单机八卡不能直接照跑。 |

这两个可作为**加速工程参考**。其中 verl 提供分别可读的[视觉语言模型在线蒸馏脚本](https://github.com/verl-project/verl/blob/main/examples/on_policy_distillation_trainer/run_qwen3_vl_8b_fsdp.sh)和[视觉语言模型 GRPO 脚本](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh)，所以可以作为第二个工程参考。

### 参考一：ms-swift，优先尝试的迁移路线

入口是官方的 [GRPO 文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md)、[蒸馏文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Distillation.md)，以及 [Ray GRPO 示例](https://github.com/modelscope/ms-swift/tree/main/examples/ray/grpo) 和 [Ray GKD 示例](https://github.com/modelscope/ms-swift/tree/main/examples/ray/gkd)。GRPO 文档提供两种 vLLM 生成方式：`colocate` 让 vLLM 和训练模型使用同一组 GPU，轮流占用显存和计算资源；`server` 则让 vLLM 在指定的 GPU 上独立运行，训练用其余 GPU。前者不必专门留出生成用的卡，但切换模型会有开销且可能占满显存；后者要从八张卡中划出一部分给生成。ms-swift 也提供只同步 LoRA 权重的选项。哪个更快要在相同数据和设置下实测。

OPSD 要额外区分**学生生成**和**教师打分**。现有损失是“完整词表 JSD”：学生每写到一个位置，教师和学生分别给所有可能的下一个 token 分配概率；JSD 衡量这两份概率分布的差异，训练学生缩小差异。“完整词表”指所有候选 token 都参与比较，并不是让教师生成所有答案。当前代码设置 `--beta 0.5`，按整个词表计算 JSD。仅把学生生成交给 vLLM，不必改变这项损失；若把教师也搬到独立的 vLLM 服务，ms-swift 文档要求使用教师 top-k logits（只传教师概率最高的 k 个 token），损失就变成 top-k 近似，属于另一组实验，不能把指标直接视为同一方法。教师仍须固定，且每条样本的真假、目标区间和类别只给教师；学生输入保持原样。[ms-swift Ray 文档](https://github.com/modelscope/ms-swift/blob/main/docs/source/Instruction/Ray.md) 还说明：它的 **Megatron** 后端有 Ray GRPO/GKD；当前使用的 **HF Trainer** 后端没有 Ray GRPO。官方对单机也优先推荐非 Ray 方案。因此，为了上 Ray 而从现有 HF Trainer 迁到 Megatron，并不是一个只改启动参数的步骤。

### 参考二：verl，供重构八卡流水线

入口是官方的 [在线蒸馏示例](https://github.com/verl-project/verl/blob/main/examples/on_policy_distillation_trainer/README.md)，其中 `run_qwen3_vl_8b_fsdp.sh` 展示了视觉语言模型、vLLM 生成、固定教师和学生训练如何分工；[多模态 GRPO 示例](https://verl.readthedocs.io/en/latest/examples/multi_modal_example.html) 与 [LoRA 文档](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html) 可供参考。该蒸馏脚本默认给学生训练 8 张 GPU、给独立教师 4 张 GPU，只有单机八卡时必须重新安排角色。示例模型和数据也并非我们的 Qwen3.5 4B、40 帧视频，不能证明视频时间输入已经兼容。

verl 的 [OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html) 当前列出的蒸馏损失是采样 token 的反向 KL 估计，或教师 top-k 的正向 KL；没有与我们现用的**完整词表 JSD**直接对应的现成配置。如果老师选择 verl，要么补写并核验这个损失，要么明确把新训练作为方法消融。还要接入我们逐样本的教师特权提示词、16／8／16 帧和相对时间、教师正确性筛选，以及 GRPO 的三项自定义奖励。它的价值主要是角色分工和调度实现，而非即插即用的训练配置。

## 需要格外注意的地方：视频输入对齐

[ms-swift 的对齐跟踪](https://github.com/modelscope/ms-swift/issues/9668) 将 Qwen3.5 的**图像**训练／vLLM 生成标为对齐，却将**视频**标为未对齐；相关的 [vLLM 问题](https://github.com/vllm-project/vllm/issues/46817) 给出了视频展开时视觉边界 token 不同的复现。两条 issue 在本次调研时仍为 open。这些记录说明不能仅凭“vLLM 能生成文字”判断 OPSD 或 GRPO 可以正确训练；也不能直接断言我们锁定版本必然出现完全相同的 token 差异，需要在实际版本上检查。

**我们现在怎么处理：**先从每个 proposal 对应的原视频中取 40 帧：前 20% 取 16 帧，中间 60% 取 8 帧，后 20% 再取 16 帧。程序把这 40 帧做成一个 MP4，同时把每帧在该 proposal 内的实际出现时间写进旁边的 `.timestamps.json`。[视频模板](trace_video_template.py) 读取两者，把画面和实际时间一起交给 Qwen。这样处理是因为取帧密度前后高、中间低；单看重新编码后的 MP4，模型会把 40 帧误当成等间隔拍摄，无法准确根据画面判断片段内的起止秒数。比如一个 10 秒片段，前 2 秒有 16 帧，中间 6 秒只有 8 帧，不能把它们都按每 0.25 秒一帧来理解。

**接入 vLLM 的风险：**如果只把这个 MP4 路径交给 vLLM，它未必会使用我们另存的逐帧时间；它也可能再次抽帧。于是学生用 vLLM **生成**时看到的是一套画面／时间，而训练器用现有模板给这个回答**计算概率和更新参数**时看到的是另一套。OPSD 的教师要在学生已生成的前缀上逐 token 打分，GRPO 也要对生成结果算概率；两边输入不同，训练信号就可能错位。这里是依据当前代码作出的风险判断，并非已经在我们的 vLLM 版本上复现。改造时需要把同样的 40 帧及其时间交给 vLLM，并实际核对两条路径展开后的输入。Ray 只负责安排进程和 GPU，不会自动修正视频输入。

## 建议按这个顺序验证

1. **先量现有八卡基线。**在相同数据、batch、采样长度和 LoRA 设置下，记录每步的解码、学生生成、教师打分／三项奖励、反向传播各花多少时间。确认瓶颈在生成，再投入 vLLM 改造；不要预设 Ray 或 vLLM 的加速倍数。
2. **先做一条视频的输入对账。**对同一个 proposal，比对 Transformers 训练路径和 vLLM 生成路径拿到的 40 帧内容与顺序、相对时间、文本 token、视觉 token 数量及位置。如果不一致，先修输入，再跑训练。至少覆盖一条异常片段和一条真实片段。
3. **做短训对账。**核对 vLLM 使用的是本步学生 LoRA，OPSD 教师仍是固定的 SFT 权重，教师在相同学生前缀上打分；GRPO 保持 4 个回答和原三项奖励。若改成 top-k、异步旧权重或不同视频处理，分别记录为新设置，不能与原基线混算。
4. **最后比较吞吐与效果。**在同一小批样本上比较每秒完成的 proposal、显存占用、有效回答比例和阶段评测指标，再决定是否做全量八卡。若视频对齐暂时无法解决，保留 Transformers 路径使用八卡并行，仍可加速处理数据，而不改变方法。

**结论：**若需要加速工程参考，可以选择 ms-swift 和 verl。先参考 ms-swift 改造现有训练器的 vLLM 生成，并以视频输入对齐作为启用条件；需要重新拆分训练、生成、教师角色时，再参考 verl 的 Ray 实现。当前代码锁定 `ms-swift==4.5.3`、`vllm==0.19.1`。
