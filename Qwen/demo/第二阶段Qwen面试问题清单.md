# 第二阶段（Qwen：SFT → OPSD → GRPO）面试问题清单

用法：先照着「第 0 节」把项目讲清楚；再逐题自测。每题后面给了**出处**（相对路径:行号），
答不上就回去读那一行，不要背我的转述。凡是标了 ⚠️ 的题，都是**设计与实现有落差**的地方——
面试官一问就能听出你是"跑通了"还是"想清楚了"，必须提前准备好诚实说法。

---

## 0. 先过第一关：三句话讲清你在做什么

面试官第一句通常是"介绍一下你负责的部分"。按这个顺序说，不要从模型结构开始：

1. **任务**：长视频里有一段被 AIGC 替换/篡改，要把它**找出来（定位起止秒数）**并**说清为什么（解释）**。
2. **两阶段分工**：第一阶段小模型（XCLIP + 神经元探测 + Mamba）在长视频上滑窗，吐出若干**候选 proposal**；
   第二阶段 Qwen3.5-4B 只看这些候选片段，输出 `Explanation:` + `Interval: [start, end]` 或 `Real`。
3. **你的训练链**：SFT 学协议 → OPSD 用**特权教师**在学生自己走错的轨迹上纠偏 → GRPO 用**可验证奖励**
   （定位 / 格式 / 解释）在真实 proposal 上做组内相对优化。

一句话收尾（这是整个项目的立身之本）：
**部署时模型永远看不到真值，训练时也只能靠"学生自己生成的轨迹"来纠错，特权信息只在训练期存在。**

出处：`README.md:219-239`（第二阶段总述与三套区间定义）、`OPSD_OPD_STAGE2_RESEARCH_AND_DESIGN.md:9`

---

## 1. 数据与坐标（最容易在白板上翻车）

**Q1. 一个 proposal 和 GT 标注对不上时，你到底拿什么当训练目标？举个数算一遍。**
答：三套区间必须分清——原始标注区间（数据集给的伪造段）、proposal 内的标注区间（两者交集，
仍以原视频为基准）、片段内目标区间（交集两端各减 proposal 起点，模型学的就是这个）。
例：GT `[100,120]`，proposal `[110,130]` → 交集 `[110,120]` → 目标 `[0,10]`。
**没有交集的 proposal 在片段级一律按 real 处理**，包括来自 fake 视频的误报。
出处：`README.md:233-239`、`opsd_common.py:33-49`、`opsd_common.py:25-32`

**Q2. proposal 命中多个 GT 时怎么处理？**
答：只取**交集最长**的那一个，并在审计文件记 `overlapping_gt_count`。
出处：`prepare_sft.py:34-36`、`opsd_common.py:33`
⚠️ 追问：设计文档明确说"应保持多个 event，不要无提示地只留第一个"
（`OPSD_OPD_STAGE2_RESEARCH_AND_DESIGN.md:96`）。你怎么解释？
→ 诚实说法：Qwen 的输出协议（`prompts/student.txt:10`）第二行**只允许一个区间**，
多事件需要改协议，属于另一组实验，所以当前版本单区间，代价是多事件 proposal 有监督损失。

**Q3. 为什么是 40 帧，而且是 16 / 8 / 16？**
答：沿用 Trace 的 boundary-aware 采样：proposal 前 20% 取 16 帧、中间 60% 取 8 帧、后 20% 取 16 帧。
任务是**定位起止边界**，边界附近信息密度高，所以两端密、中间稀。这是针对"时序定位"的领域设计，
不是随手均匀抽帧。出处：`common.py:214-216`、`README.md:325`

**Q4（工程硬题，必须会）. 40 帧是**非均匀**采的，存成 MP4 之后帧就变均匀了，
Qwen 怎么知道第 7 帧是第几秒？**
答：这是本项目最容易出错的一处。做法是：截取时把每帧相对 proposal 起点的**真实毫秒**写进
`.timestamps.json`；再自定义 ms-swift 的 Qwen3.5 模板，把 `fps=1000`、
`frames_indices=毫秒数组`、`do_sample_frames=False` 塞进 `video_metadata`，
覆盖官方"按均匀间隔生成元数据"的默认行为。不这么做，模型会把 10 秒片段里的
"前 2 秒 16 帧、中间 6 秒 8 帧"全按每 0.25 秒一帧理解，时间全错。
出处：`trace_video_template.py:28-53`、`common.py:219-241`
追问：为什么 GRPO 关掉 vLLM（`train_grpo.py:150`）？
→ vLLM 的视频路径未必吃这套自定义 metadata，甚至可能二次抽帧，
导致"生成时看到的画面/时间"和"训练器算概率时的画面/时间"不是一套，OPSD 的逐 token 教师打分会错位。
出处：`Qwen/OPSD与GRPO八卡加速参考.md:46-56`

---

## 2. SFT

**Q5. SFT 数据里为什么没有 real 样本？**
答：`prepare_sft.py:31-33` 直接 `continue` 掉所有与 GT 无交集的 proposal；`student_sft.txt` 只要求输出
`Interval`，不教 `Real`。所以 SFT 后模型**只会报区间**。
⚠️ 追问：那测试时遇到 real proposal 怎么办？
→ 这是有意的阶段划分：SFT 先学"看得见异常并给出边界"，Real 分支留到 OPSD 阶段的 `student.txt` 再引入
（`prompts/student.txt:5`）。代价是 SFT 单独评测时假阳性会偏高，这也正是后面要用 OPSD/GRPO 补的原因。
不要把这个说成"忘记做负样本了"。

**Q6. 解释答案怎么来的？为什么有时一句有时三句？**
答：直接取标注：完整覆盖 GT 时用 `bnd_cot_st / obj_cot / bnd_cot_ed` 三句（开始、主要异常、结束）；
只覆盖部分 GT 时**只留 obj 一句**，避免模型去描述片段外的画面（它根本看不到）。
`Round4`（spatio-temporal）样本本来就只有一句。出处：`common.py:85-98`、`opsd_common.py:36-44`

**Q7. 为什么冻结视觉编码器和 aligner，只训语言侧 LoRA？**
出处：`train_sft.py:122-123`、`train_opsd.py:214-216`
⚠️ 深水题：那你的训练到底提升了**感知能力**还是**表达能力**？
→ 冻结视觉塔意味着视觉取证能力基本没被训练，被训的是"怎么把结论说成规定的格式和边界"。
设计文档自己写了停止条件：如果指标提升全来自格式/长度，就不能声称"视觉取证变强了"
（`OPSD_OPD_STAGE2_RESEARCH_AND_DESIGN.md:209`）。**能主动说出这个风险，比硬吹提升更得分。**

---

## 3. OPSD（最核心，也最容易被问穿）

**Q8. OPSD 和普通蒸馏/SFT 的本质区别是什么？**
答：三点——
① **on-policy**：学生先自己生成，教师只在**学生已写出的前缀**上给下一 token 分布，
不另写一篇标准答案；数据里 `messages` 只有 user（`prepare_opsd.py:38`），证明答案是学生现生成的。
② **教师有特权信息、学生没有**：教师多看一段 "Verified information"（真假标签、片段内目标区间、
伪造类别、对象及类别），学生只有视频。
③ **损失是完整词表 JSD**（`--beta 0.5`、`--lmbda 1.0`、`--sft_alpha 0`）：
λ=1 表示全部用学生自己生成的序列，sft_alpha=0 表示不掺 GT 监督。
出处：`train_opsd.py:197-247`、`prompts/teacher_opsd.txt:2-11`

**Q9. 教师是谁？为什么教师必须先"通过准入"？**
答：教师 = 把 SFT LoRA **合并成的独立模型**（`merge_sft_teacher.py`），与学生从**同一份 SFT 权重**起步，
训练中冻结，只有学生 LoRA 更新。启动前要跑 `check_teacher.py`：教师整视频 `Loc_F1`/`Loc_IoU` 必须更高、
`Det_Acc` 不降、逐 proposal 的 fake/real 判定与无效回答不退化。
出处：`train_opsd.py:103-112`、`check_teacher.py:114-122`
⚠️ **这是最危险的一题**：教师看到了真值，它当然更好，**这个 gate 是不是循环论证？**
→ 诚实说法（推荐照这个说）：这个 gate **证明的是流程没坏**——特权提示词没有把教师搞成格式崩溃、
teacher/student 两边配置确实一致、教师没有被真值诱导成"乱报"。
它**不能**证明"特权信息本身的视觉有效性"；要证明后者需要错配/时间错位的对照实验
（`OPSD_OPD_STAGE2_RESEARCH_AND_DESIGN.md:192-198` 第 7/8 组），当前实现**没做**。
把这两件事分开讲，面试官会觉得你清醒。

**Q10. 为什么还要逐条筛掉一部分训练 proposal（teacher-min-iou 0.5）？**
答：教师从头回答不正确的样本直接丢：real 必须输出 `Real`；fake 必须输出有效区间且与片段内目标
IoU ≥ 0.5。出处：`opsd_teacher_filter.py:56-88`、`README.md:509`
⚠️ 追问：这不就把**难样本**全丢了？EDGE-OPD 明明说要在困难正样本上做 guided rollout。
→ 是的，这是当前实现与设计文档最大的落差之一
（文档 `OPSD_OPD_STAGE2_RESEARCH_AND_DESIGN.md:108-118` 建议对困难正样本用
`q = α·p_T + (1-α)·p_S` 引导 rollout，实现里没有）。
诚实说法：当前是"保守版"——只蒸馏教师能答对的样本，保证蒸馏目标可信；
代价是最需要纠偏的样本没被训到，guided rollout 是下一步。

**Q11（必问）. 你给教师的是"视觉证据"还是"文字答案"？**
设计文档主张的是 **RPE-OPD**：教师看"真实参考段（上）/ AIGC 候选段（下）**竖向拼接视频**"，
**明确禁止**把真值时间、fake/real 标签、标注文本塞给教师
（`OPSD_OPD_STAGE2_RESEARCH_AND_DESIGN.md:79-81`）。
但**实现里给的就是文字真值**：`opsd_common.py:105-119` 把 verified label、verified interval 精确数字、
forgery category、object/subclass、起止类别全写进了 prompt。
→ 必须准备好这个说法：原始设计依赖 TASLE 资产里的"生成前源片段 ↔ 生成后片段"映射，
文档自己也标注了"开始开发前需检查资产是否确实保存了这层映射"
（`OPSD_OPD_STAGE2_RESEARCH_AND_DESIGN.md:98`）。**当前版本是"文本特权版 OPSD"**，
主张应表述为"训练期可用真值的教师把学生轨迹引向正确标签与边界"，
**不能**表述为"学生学会了真实/生成的视觉对比证据"。这句话一定要自己先说出来。

**Q12. JSD 而不是前向/反向 KL，为什么？beta=0.5 是什么？**
答：JSD = 对称混合，`JSD = β·KL(p_S‖M) + (1-β)·KL(p_T‖M)`，M 是混合分布，β=0.5 即对称。
选它因为 JSD 有界、不会像反向 KL 那样追求 mode-seeking 而崩坏，也不像前向 KL 那样过度平滑。
**保留完整词表**意味着 `Real`、`Interval`、时间数字全都参与分布匹配——这正是要纠偏的位置。
出处：`train_opsd.py:197-198`、`README.md:511`
补充说明：前进/反向 KL 的命名在文献里是乱的，面试时**直接写公式**，说清"用谁的概率加权"，
别只说名字。前向（用教师加权）是 mode-covering，反向（用学生自己加权）是 mode-seeking。
追问：设计文档要的 evidence mask / token 加权（`OPSD_OPD_STAGE2_RESEARCH_AND_DESIGN.md:131-133`）做了吗？
→ 没有，当前是全序列 JSD，属于文档列为"pair-OPSD 全 token KL"的对照档位（第 191 行第 4 组）。

**Q13. 为什么评测一律用贪心（temperature=0）？**
答：SFT 全量测试发现采样解码会让无效回答和定位误差变多，所以采样只留作可选实验。
出处：`README.md:229`、`Qwen/experiments/repetition/README.md`（40 条小样本结论与 1002 条全量结论相反，
这是"小样本结论不成立"的现成例子，很适合用来回答"你怎么避免拍脑袋下结论")
追问：那模型到底有没有"真实定位能力"，还是只学会了模板？
→ 看 `parse_summary.json` 的 `proposal_detection` 拆分（true/false positive、false negative、
invalid_positive/invalid_negative），以及 `metrics.json` 三个域的 Det_Acc / Loc_F1 / Loc_IoU。
出处：`evaluate.py:328-361`

---

## 4. GRPO / RL

**Q14. 为什么最后还要 GRPO？OPSD 不够吗？**
答：分工不同。OPSD 是"模仿教师"，目标仍是把学生拉向某个目标分布；
GRPO 是"在学生自己的真实输出分布上做相对选择"——同一个 proposal 采样 G 条，
用组内均值/标准差算 advantage，**不需要价值网络**。而且奖励是**可验证规则**（IoU、边界误差、解析器），
比神经 reward model 更难被 hack。出处：`GRPO_AFTER_OPD_RESEARCH_AND_DESIGN.md:15`

**Q15. 三项奖励怎么算？权重为什么是 1.0 / 0.1 / 0.3？**
- **定位**（1.0）：real 目标下输出 `Real` 得 +1，误报区间 -1，无效 -0.5；fake 目标下输出 `Real`
  或无效得 -1，输出区间按 `0.25 + 0.55·IoU + 0.20·边界分 − 0.25·未匹配` 计分。
  出处：`grpo_rewards.py:65-82`
- **格式**（0.1）：能解析出规定的解释 + `Real`/合法区间得 +1，否则 -1。权重小，只是协议锚点，
  给大了模型会只学会输出漂亮 token。出处：`grpo_rewards.py:85-88`
- **解释**（0.3）：**只对 fake 且定位 IoU ≥ 0.3 的回答打分**，其余给 0——这是"定位门控"，
  防止"定位错但说得漂亮"拿奖励。出处：`grpo_rewards.py:99-104`
- 三个分量的尺度都压在 [-1,1] 附近再加权，是为了让任何一个分量不因量纲大而独占组内 advantage。

**Q16. 解释奖励具体怎么算？为什么不用 BLEU / 不用 LLM judge？**
答：把标注拆成 1~3 条**证据要点**（object 异常权重 2，起止各 1），把生成解释按 `. ! ? ;`、
`whereas`/`however` 切成原子短句，用**冻结的 NLI cross-encoder**（`nli-deberta-v3-small`，CPU）
以标注为前提、短句为假设，做**一对一最大权匹配**，算加权覆盖率和矛盾度，最终
`score = 加权覆盖率 − 0.50 × 已匹配句子的矛盾程度`。
不用 BLEU：单条 gold 的表面相似度会漏掉不同但有效的解释；不用 LLM judge：长度/文风偏好会被过优化。
出处：`grpo_rewards.py:106-127`、`Trace/trace/text_explanation_reward.py:165-219`
⚠️ 追问 1：**为什么丢掉了 precision？**
→ 代码注释给了理由：标注未必列尽所有可见现象，额外句子没匹配上不等于解释错
（`grpo_rewards.py:124-126`）。风险是：fact 只有 1~3 条，模型可以"把所有常见描述写一遍"把覆盖率刷满——
这正好是文档列出的攻击测试项（`GRPO_AFTER_OPD_RESEARCH_AND_DESIGN.md:133-141`）。
⚠️ 追问 2：**NLI 判断的是"标注是否支持这句话"，不是"视频是否支持这句话"，视觉 grounding 在哪？**
→ 诚实说法：当前是 text-to-text 蕴含，是降级实现；文档要的 P（claim 被 candidate 视频支持的比例）
需要视觉 verifier，属于后续工作。因此解释分的提升**只能**表述为"与标注事实一致"，不能说"画面可证实"。

**Q17. 训练超参为什么这么设？**
`num_generations=4`（组内样本数与显存的权衡）、`temperature=0.8`（要有不同边界才有权重可学）、
`beta=0.02`（到**初始 adapter** 的 KL，不是 base 模型——约束"别跑离 SFT/OPSD 太远"）、
`num_iterations=1`（单步更新，不做多轮 PPO）、`ref_adapters = 初始 adapter`。
出处：`train_grpo.py:137-154`
⚠️ 顺着追问：`num_iterations=1` 意味着每个 batch 只更新一次，`π_old = π_θ`，
重要性比 `ρ ≡ 1`，**PPO 的 clipping 实际不起作用**，退化成 REINFORCE + 组内基线。能主动讲出来很加分。

**Q18. `reward_curve.png` 第三行那条曲线是什么？高了怎么办？**
答：`frac_reward_zero_std` = **同一组 4 条回答得分完全相同的比例**。
组内无方差 → advantage 全 0 → 这一步**没有学习信号**。
高了说明采样退化（温度太低/输出太模板化）或奖励太粗（比如全组都判 invalid），
要调温度、检查无效回答率。出处：`train_grpo.py:236-238`、`README.md:586`

**Q19. 学生会不会通过"输出格式错误"来逃避定位责任？**
答：不会净赚。评测里 invalid 不算 false positive，但也不会产生预测区间；
fake 视频若一个有效预测都没有，整视频被判 real，**直接扣 Det_Acc**。
GRPO 里 invalid 还拿 -1 / -0.5。出处：`evaluate.py:336-346`、`README.md:331-333`
⚠️ 顺带一题：设计文档要的 **structure-aware masked GRPO**（把 loc/fmt/exp 三种 advantage
分别只作用到时间 token / 结构 token / 解释 token，
`GRPO_AFTER_OPD_RESEARCH_AND_DESIGN.md:169-185`）**没实现**，当前是 reward 广播到所有 token 的 vanilla 版。
被问到就说：这是必须首先报告的 baseline，masked 版是消融第 5 组。

---

## 5. 跨阶段与整体判断（决定你能不能拿到"有思考"的评价）

**Q20. 为什么第二阶段从 Trace 换成 Qwen3.5-4B？**
答：Trace 是专用时间 token 的 Video LLM（text/time/score 三解码头 + `<sync>`/`<sep>`），
结构精确但工程复杂、解释能力弱；Qwen3.5-4B 是通用 VLM，用自然语言直接输出
`Explanation` + `Interval`，可解释性强、LoRA 好接。**代价**是放弃了离散时间 token 的精度，
改成自然语言数字，于是"格式合法性"变成必须显式奖励/蒸馏的东西（这就是为什么有格式奖励和教师筛选）。

**Q21. 你的方法真正的创新点是什么？用一句话。**
建议的说法（照抄设计文档的定位，**不要**说成"用了 GRPO 让文字更像标注"）：
**在不改变部署输入的前提下，用训练期才可见的特权教师纠正学生自己走错的轨迹，
再用定位门控的可验证奖励把"时间定位"和"与标注事实一致的解释"联合优化。**
出处：`GRPO_AFTER_OPD_RESEARCH_AND_DESIGN.md:233`

**Q22. 如果实验涨点了，你怎么证明不是 reward hacking / 格式投机？**
答：三件事——① 拆 semantic-ORR 与 format-ORR，若提升全来自 format 就不能算"视觉取证变强"；
② 报 `negative false-event rate`，避免"永远报 fake"的假涨；
③ 做错配标注 / 时间错位 / 通用词堆砌 / 长文本无证据的攻击测试，
以及自动解释分与人工盲评的相关性。出处：
`OPSD_OPD_STAGE2_RESEARCH_AND_DESIGN.md:184, 209`、`GRPO_AFTER_OPD_RESEARCH_AND_DESIGN.md:133-141, 222`

**Q23. 第一阶段给你的 proposal 不准怎么办？**
答：这是整个系统的真实瓶颈，当前第二阶段**完全继承**第一阶段的输出分布（包括误报），
并没有 oracle proposal 对照。设计文档要求分别报 oracle proposal 与真实 stage-1 proposal 两种设置
（`OPSD_OPD_STAGE2_RESEARCH_AND_DESIGN.md:182`）。当前实现只用了真实 proposal。

---

## 6. 工程与框架选型（加分题，答得好很值钱）

**Q24. 训练中断了怎么续？怎么防止"续到了另一个实验"上？**
答：所有阶段首次跑 `--resume none --clean`，中断后保持参数不变改 `--resume auto` 去掉 `--clean`。
脚本会写 `*_config.json`，续训时逐项比对：模型/adapter 的 **sha256**、数据集 sha256、
教师来源 sha256、提示词文本、门控 sha256、设备、超参、甚至奖励代码的 sha256
（`train_grpo.py:184-188`）。不一致直接报错，不做容错。
出处：`train_opsd.py:249-283`、`train_grpo.py:173-204`、`README.md:231`

**Q25. 八卡怎么并行？为什么这么保守？**
答：评测用 `torch.distributed.run --standalone`，每张卡一个完整模型、按 rank 分 proposal，
rank0 显示进度条、最后合并 shard；每个 proposal 用 `sha256(task_id)` 派生独立种子，
**换卡数也不改变采样结果**。训练用 `swift rlhf`（HF Trainer + LoRA + gradient accumulation），
`--use_vllm false`。出处：`evaluate.py:172-188, 245-251`
加速是下一步：ms-swift 的 Megatron + Ray + vLLM，或 verl；
**前提是视频时间输入在两条路径上完全对齐**，且教师若被放进 vLLM 服务，完整词表 JSD 会退化成 top-k 近似，
那就不是同一个方法了。出处：`Qwen/OPSD与GRPO八卡加速参考.md:36-44, 58-64`

**Q26. 为什么训练框架选 ms-swift，不自己写或用 verl？**
答：三个硬约束同时满足的只有它——
① **模型侧**：官方注册了 `qwen3_5` 模板，只需继承 `Qwen3_5Template` 覆写 `replace_tag` 就能接管视频取帧
（`trace_video_template.py:20-54`）；
② **算法侧**：`--rlhf_type gkd` 原生有 on-policy 蒸馏（学生生成 + 教师同前缀打分），
`--rlhf_type grpo` 原生有组内相对优势；
③ **扩展侧**：`--external_plugins` 能把自定义视频模板和奖励函数挂进去（`train_grpo.py:140`）。
**对照**：Trace 阶段是自研的（842 行 `Trace/run_opd_grpo.py` + 覆写 `Trace/trace/trace_trainer.py`），
换 Qwen3.5 后视觉塔和 processor 全变，重写成本翻倍。
**为什么不是 verl**：八卡文档比较过——示例都是图片、默认教师独占 4 卡超单机八卡、
现成损失只有反向 KL 或教师 top-k 正向 KL，**没有完整词表 JSD**，而完整词表 JSD 正是方法口径；
迁移要重写 JSD 并重新接入时间戳/特权提示词/预检筛选/三项奖励，工作量明显更大，
结论是"只作对照，不混进主线"（`Qwen/OPSD与GRPO八卡加速参考.md:44, 65`）。
⚠️ 代价要自己讲出来：ms-swift 4.5.3 + TRL 0.27 + Transformers 5.9 有兼容问题（要打 `trl_compat.py` 补丁）；
`TransformersEngine` 不支持 `presence_penalty`（`Qwen/experiments/repetition/README.md`）；
HF Trainer 后端没有 Ray GRPO，上八卡 Ray 要迁 Megatron 后端等于移植代码。
收尾一句：选框架不是选"功能最多的"，是选"在我的特殊点上留了插手空间、其余部分别让我重写"的。

---

## 7. 最后：面试官问"你有什么想问我的"之前，先自查这 9 句能不能说出来

1. 三套区间的换算（白板算对一个例子）。
2. 40 帧非均匀采样 + `.timestamps.json` + 自定义模板，为什么必须这么做。
3. on-policy 的证据：数据里只有 user 消息，答案是学生现生成的。
4. 教师 = 合并的 SFT 权重 + 特权 prompt，训练中冻结，λ=1、sft_alpha=0 的含义。
5. 教师 gate **证明了什么、没证明什么**。
6. 特权信息是**文字真值**不是视觉参考，以及因此主张必须怎么收窄。
7. 解释奖励的门控（IoU ≥ 0.3）、一对一匹配、为什么没有 precision 和视觉 grounding。
8. `frac_reward_zero_std` 高意味着什么。
9. 为什么选 ms-swift，以及它的三个代价。

说不全没关系，但**不要编**。这行的规矩是：给结论就要给出处，没查到就说"我没做这个实验，这是推测"。
