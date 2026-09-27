# OPSD 与 GRPO：八卡加速参考

本文比较 ms-swift 与 verl 的八卡加速实现，并说明本项目的训练方法和视频输入需要怎样适配。先核对视频输入，再比较生成速度与训练效果；Ray 本身没有固定的加速倍数。

## 八卡方案使用哪些库

**训练方案选 ms-swift 的 Megatron 后端＋Ray＋vLLM。**ms-swift 负责 OPSD 和 GRPO 训练，PyTorch 与 LoRA 更新学生参数；vLLM 负责学生在线生成；Ray 安排训练、生成和教师使用的 GPU。Megatron 后端另需 `megatron-core`、`mcore-bridge`、`transformer-engine`，安装要求见[官方说明](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Megatron-SWIFT/Quick-start.md)。现有脚本使用 ms-swift 的 HF Trainer 后端，迁到上述方案需要移植训练代码，不能只加装 Ray。

**verl 只作为第二个代码参考。**它提供另一套 Ray＋vLLM 训练流水线，不与 ms-swift 一起作为本方案的训练库。若最终改用 verl，需重新接入本项目的视频时间戳、教师提示词和完整词表 JSD。当前单卡环境和 `requirements.txt` 保持现状；八卡新增依赖的版本待目标环境验证后固定。

## 当前训练口径

当前代码使用 ms-swift 的 `swift rlhf`（HF Trainer 后端），两阶段均以 LoRA 更新学生，`--use_vllm false`，用 Transformers 处理视频。[OPSD 训练脚本](train_opsd.py)从 SFT LoRA 出发：学生每条样本在线生成一条回答；冻结的 SFT 教师看同一视频、学生已写出的前缀，以及仅给教师的真假标签、片段内目标区间和异常类别，用完整词表 JSD（`beta=0.5`、`lmbda=1`、`sft_alpha=0`）指导学生。训练前还有教师完整生成预检和逐样本筛选。[GRPO 训练脚本](train_grpo.py)可从 OPSD 或 SFT LoRA 出发：每条输入采样 4 个回答，分别算定位、格式、解释奖励，权重为 1.0、0.1、0.3。

两阶段都把 40 帧按前 20%／中间 60%／后 20% 取 16／8／16 帧。相邻的 `.timestamps.json` 保存每帧相对 proposal 起点的真实时间；[视频模板](trace_video_template.py)把画面和时间一起交给 Qwen。迁移框架时，这些输入和训练口径要逐项保留。

## 两个可模仿的项目

| 项目 | 可参考的实现 | 需要适配的部分 |
| --- | --- | --- |
| [ms-swift](https://github.com/modelscope/ms-swift) | 现有 HF Trainer 上的 vLLM 生成接法，以及 Megatron＋Ray 的 GKD/GRPO 角色分配 | 现有 `swift rlhf` 不能只加一个 Ray 参数；Ray 示例属于另一训练后端。视频时间输入仍需适配。 |
| [verl](https://github.com/verl-project/verl) | 视觉模型的 Ray＋vLLM 在线蒸馏示例和多模态 GRPO 示例，适合参考重构后的训练／生成／教师流水线 | 示例是图片任务，默认教师另占 GPU；其现成蒸馏损失不是我们使用的完整词表 JSD。 |

### 参考一：ms-swift，先验证最小改造

**先做 vLLM，不必先上 Ray。**官方 [GRPO 文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md)和[蒸馏文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Distillation.md)允许用 vLLM 给学生在线生成：`colocate` 让生成和训练轮流使用同一组 GPU；`server` 划出 GPU 给独立生成服务。LoRA 可只同步 adapter。教师仍可由训练器本地打分，以保留当前 GKD 损失。不过这一步必须先解决下文的视频输入对齐；仅切换 `--use_vllm` 并不够。哪种部署更快要实测。

**若明确需要 Ray，参考 ms-swift 的 [Ray GKD 示例](https://github.com/modelscope/ms-swift/tree/main/examples/ray/gkd)和[Ray GRPO 示例](https://github.com/modelscope/ms-swift/tree/main/examples/ray/grpo)。**[官方 Ray 文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Ray.md)说明：Megatron 后端可用 Ray 分配 `train`、`rollout`、`teacher` 角色；我们当前用的 HF Trainer 后端没有 Ray GRPO。因此要迁到 Megatron，并检查 SFT LoRA 权重、视频模板、数据字段、自定义奖励在新后端的接法，不能只在原脚本加 `--use_ray true`。官方对单机也建议先用更简单的非 Ray 路线。

OPSD 的教师安排尤其重要。“完整词表 JSD”是在学生回答的每个位置，比较师生对**所有可能的下一个 token** 的概率。ms-swift 的 Ray GKD 文档表明：教师与训练共用 GPU 时支持完整词表；教师单独放进 vLLM 服务时只支持教师 top-k 概率，损失会变成近似。若要保留当前实验口径，就让**固定教师与学生训练共用 GPU，学生生成用 vLLM**；八卡上的共用或分卡数量由显存及吞吐实测决定。不能把改为 top-k 后的结果直接当作同一方法。

### 参考二：verl，供重构八卡流水线

入口是[在线蒸馏说明](https://github.com/verl-project/verl/blob/main/examples/on_policy_distillation_trainer/README.md)、[Qwen3-VL 视觉蒸馏脚本](https://github.com/verl-project/verl/blob/main/examples/on_policy_distillation_trainer/run_qwen3_vl_8b_fsdp.sh)和[Qwen2.5-VL GRPO 脚本](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh)。视觉蒸馏脚本把学生训练设为 8 卡、独立教师设为 4 卡，默认需要超过八卡；单机八卡必须重新安排角色与显存。两个示例都处理图片，不能据此认为已经支持我们的视频帧和时间戳。[LoRA 文档](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)可用于核对权重接续方式。

[verl 的 OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html)列出的现成损失是学生采样 token 的反向 KL 估计，或教师 top-k 的正向 KL；没有当前完整词表 JSD 的直接配置。若采用 verl，需要补写并核验 JSD，或明确把新损失作为另一组实验；还要接入每条样本不同的教师特权提示词、冻结的 SFT 教师、预检筛选，以及 GRPO 的三项奖励。它适合借鉴 Ray 角色分工和 vLLM 生成流水线，迁移工作明显多于沿用 ms-swift。

## 需要格外注意的地方：视频输入对齐

[ms-swift 的对齐跟踪](https://github.com/modelscope/ms-swift/issues/9668) 将 Qwen3.5 的**图像**训练／vLLM 生成标为对齐，却将**视频**标为未对齐；相关的 [vLLM 问题](https://github.com/vllm-project/vllm/issues/46817) 给出了视频展开时视觉边界 token 不同的复现。两条 issue 在本次调研时仍为 open。这些记录说明不能仅凭“vLLM 能生成文字”判断 OPSD 或 GRPO 可以正确训练；也不能直接断言我们锁定版本必然出现完全相同的 token 差异，需要在实际版本上检查。

**我们现在怎么处理**：先从每个 proposal 对应的原视频中取 40 帧：前 20% 取 16 帧，中间 60% 取 8 帧，后 20% 再取 16 帧。程序把这 40 帧做成一个 MP4，同时把每帧在该 proposal 内的实际出现时间写进旁边的 `.timestamps.json`。[视频模板](trace_video_template.py) 读取两者，把画面和实际时间一起交给 Qwen。这样处理是因为取帧密度前后高、中间低；单看重新编码后的 MP4，模型会把 40 帧误当成等间隔拍摄，无法准确根据画面判断片段内的起止秒数。比如一个 10 秒片段，前 2 秒有 16 帧，中间 6 秒只有 8 帧，不能把它们都按每 0.25 秒一帧来理解。

**接入 vLLM 的风险**：如果只把这个 MP4 路径交给 vLLM，它未必会使用我们另存的逐帧时间；它也可能再次抽帧。于是学生用 vLLM **生成**时看到的是一套画面／时间，而训练器用现有模板给这个回答**计算概率和更新参数**时看到的是另一套。OPSD 的教师要在学生已生成的前缀上逐 token 打分，GRPO 也要对生成结果算概率；两边输入不同，训练信号就可能错位。这里是依据当前代码作出的风险判断，并非已经在我们的 vLLM 版本上复现。改造时需要把同样的 40 帧及其时间交给 vLLM，并实际核对两条路径展开后的输入。Ray 只负责安排进程和 GPU，不会自动修正视频输入。

## 建议的实施顺序

1. **先量当前八卡基线**：对相同数据、batch、采样长度和 LoRA 设置，记录视频解码、学生生成、教师打分／三项奖励、反向传播各花多少时间。确认生成是瓶颈，再投入 vLLM 改造。
2. **先对账视频输入**：在一条 fake 和一条 real proposal 上，比对 Transformers 与 vLLM 实际收到的 40 帧内容和顺序、相对时间、文本及视觉 token 的数量和位置。输入不同就先修复，不能直接跑训练。
3. **短训对账方法**：核对 vLLM 用的是本步学生 LoRA；教师仍是最初冻结的 SFT 权重、只在教师提示词中看到真值，并沿同一学生前缀打分；OPSD 保持完整词表 JSD，GRPO 保持四次采样及 1.0／0.1／0.3 的三项奖励。教师预检、样本筛选和最终评测也沿用现有口径。
4. **再比较速度和结果**：在同一批样本上记录每秒完成的 proposal、显存、有效回答比例，以及最终 `Det_Acc`、`Loc_F1`、`Loc_IoU`。如改成 top-k、采样 token KL、异步旧权重或另一视频处理，单列为新实验。视频输入暂时对不齐时，可以先保留 Transformers 路径做八卡并行。

**实施选择**：正式八卡改造选 ms-swift 的 Megatron＋Ray＋vLLM；先用现有 HF Trainer＋vLLM 做视频输入验证。verl 提供另一套 Ray＋vLLM 视觉训练代码供对照，不混进主线。两条参考路线都以视频输入对齐为前提；目前没有实测依据声称哪条更快。上面链接指向上游主分支，迁移时应以实际使用的版本核对参数和行为。
