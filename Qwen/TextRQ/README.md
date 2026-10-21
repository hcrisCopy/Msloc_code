# TextRQ：解释文本质量评测

入口为 `Qwen/TextRQ/evaluate_text_rq.py`（旧版： `evaluate_RQ.py/.sh` ）。读取 `Qwen/evaluate.py` 生成的 `predictions.json`（含 `proposal_results`）及同一份测试 anno，不读取视频、不重新生成解释。

这里的**预测 proposal** 指模型预测的候选片段，**GT 区间**指人工标注的异常片段。二分类以 **Fake 为正类**：预测 proposal 的裁剪窗口与任一异常 GT 区间有交集为 Fake，否则为 Real。

| 分组 | GT 片段类别 | 预测结论 |
| --- | --- | --- |
| TP | Fake | Fake |
| FP | Real | Fake |
| FN | Fake | Real |
| TN | Real | Real |

**只评 TP/TN；FP/FN 不调用裁判。** 格式错误或非法区间单列为 `invalid`，也不评分。低 IoU 的 TP 保留。

**1. 解析预测，决定哪些解释能评分**

模型输出应为两行：第一行 `Explanation: 解释文本`；第二行是 `Real` 或 `Interval: [start, end]`（表示 Fake）。

```text
每个测试视频
├─ 第一阶段未给出预测 proposal → 流程按 Real 输出，第二阶段不生成解释
│  └─ 记录这类视频的数量；没有解释可评，不进入文本均分
└─ 有预测 proposal → 逐个解析对应的模型原始输出
   ├─ Explanation 提取失败 → 不评分
   │  └─ 例如：标签拼错、解释不在首行、Explanation: 后为空
   ├─ 格式错误或区间非法 → 不评分
   │  └─ 例如：结论只写 Fake、Interval 缺数字、起止颠倒、结束超出片段
   └─ 格式、区间合法且 Explanation 非空 → 对照 GT 片段类别
      ├─ FP/FN：二分类错误 → 不评分
      ├─ TN：正常片段预测为 Real → 评流畅性和结论一致性
      └─ TP：异常片段预测为 Fake → 评流畅性和结论一致性
         └─ 有对应异常 GT 参考 → 再评对象、类型、细节，并计算四维 RQ
```

所有排除项保留数量，不进入文本均分，也不补 0。GT 片段类别由预测 proposal 的裁剪窗口决定，最终预测 Interval 用于计算定位 IoU。

**报表里的 N 怎么看**

`Scored N` 是**评过多少条解释**；`Valid N` 是**某个评分项有多少条分数**。

例如，100 个预测 proposal 中，20 个分类错误、10 个输出无效，剩下 50 个 TP 和 20 个 TN 完成评分；假设这 50 个 TP 都有对应异常参考：

| 报表项 | 数量 | 如何计算 |
|---|---:|---|
| `Input N` | 100 | 输入的预测 proposal 总数 |
| `Scored N` | 70 | 完成评分的 TP/TN 解释数，每条只算一次 |
| 流畅性、结论一致性的 `Valid N` | 各 70 | 各项分数之和分别 ÷ **70** |
| 对象、类型、细节各自的 `Valid N` | 各 50 | 各项分数之和分别 ÷ **50** |
| RQ 的 `Valid N`（或 `RQ N`） | 50 | 50 条解释各自的四维平均分之和 ÷ **50** |

不评分、留空的项目都不补 0。数量保存在 `selection_summary.json`：

- `videos_without_proposals`：第一阶段没有给出候选片段的视频数。
- `selection_counts`：`eligible` 为可评分 TP/TN；排除原因依次为 `missing_explanation`（解释提取失败）、`invalid_prediction`（其余格式或区间错误）、`incorrect_classification`（FP/FN），每条只归一类。
- `outcome_counts`：全部输入的 TP/FP/FN/TN/invalid 数量；`parse_status_counts`：格式、区间解析状态。它们是同一批输入的不同统计方式，不能与排除数相加。

**2. 为每条解释选择对应标注**

```text
预测 proposal 的裁剪窗口与几条异常 GT 区间有交集？
├─ 0 条 → 该片段 GT 为 Real（真实视频，或 Fake 视频中的正常片段）
│  └─ TN 评流畅性和结论一致性，不计算四维 RQ；FP 已排除
├─ 1 条 → 该片段 GT 为 Fake，用这一条的标注
└─ 多条 → 该片段 GT 为 Fake，比较最终预测区间与这些 GT 的 IoU
   ├─ 最大值 > 0，且只有一条达到最大值 → 用这一条的标注
   └─ 并列最大、全部为 0、预测区间无效或缺失 → 无法选定，不提供标注

选定 GT 后，组织参考内容
├─ 预测 proposal 完整包含标注事件 + 非 Round4 + 两侧边界标注齐全
│  └─ 三部分：开始 bnd_cot_st → 主体 obj_cot → 结束 bnd_cot_ed
└─ 其他情况 → 仅主体 obj_cot
```

只有通过上述筛选的 TP/TN 进入评分。多个 TP 可以共用同一条 GT，不因 IoU 低而排除。每部分保留全部条目的主体、分类、子分类和描述；标注中的条目数不等于句子数。

**3. 一句／三句怎么对照**

预测格式约定：一句写主体事件，三句按开始—主体—结束描述。TextRQ 将整条解释作为一个评分单位，不按句号拆分，也不要求预测与参考句数相同。

```text
参考包含哪些内容？
├─ 只有主体
│  ├─ 预测一句 → 对照主体描述
│  └─ 预测三句 → 对照有标注支持的内容；额外开始/结束描述不能仅因无标注就判错
│     （两种情况都不要求补写开始/结束）
└─ 开始、主体、结束齐全
   ├─ 预测一句 → 看是否涵盖关键内容；缺少重要变化影响细节分，不因句数少直接扣分
   └─ 预测三句 → 综合对照选中的主体及开始、结束参考，不按句子序号机械配对
```

**4. 输出哪些分数**

```text
每条入选的 TP/TN 解释
├─ 四维 RQ
│  ├─ TP 有对应标注（仅主体或三部分均可）→ 四项分别评分，再取平均
│  │  ├─ 流畅性：文字是否通顺、连贯
│  │  ├─ 对象准确性：主体是否符合标注
│  │  ├─ 异常类型准确性：异常是否符合标注
│  │  └─ 细节一致性：细节、状态变化是否符合可用标注
│  └─ TN，或 TP 无法选定参考 → 此处只评流畅性，另三项及 RQ 留空
└─ 解释—结论一致性（单列，不计入 RQ）
   └─ TP/TN 均评解释是否支持预测的 Fake/Real 结论
```

每项 1–5 分，附一句简洁理由；Real/Fake 标准相同。每项均分 = 该项有效分数之和 ÷ 该项有效数量（`Valid N`）；留空为 `null`，不补 0。RQ 均分只纳入四项齐全的解释。

提示词位于 `Qwen/prompts/`，共用 `text_rq_common.txt`。每条入选解释请求裁判 2 次，有异常参考的 TP 再增加 1 次事实评分：

- **流畅性**：只传预测的 `Explanation` 文本。
- **解释—结论一致性**：传预测的 `Explanation` 文本，以及从 `Interval` / `Real` 解析出的 Fake / Real 结论；不传 `Interval` 的起止数值。
- **对象、异常类型、细节准确性**：传预测的 `Explanation` 文本和对应 GT 的文字标注，不传预测的 `Interval` / `Real`。

事实评分中，Text A 是预测解释，Text B 是 GT 参考。Main 有多条时，裁判先选最相关的一条，其余 Main 不参与评分。选中的 Main 与已有的 Start、End 共同用于对象、异常类型、细节三个维度的评分；只有 Main 时只评主体内容。每条参考保留对象、分类、子分类和 caption。完整输入见 [prompt_inputs.txt](prompt_inputs.txt)。

**结果怎么统计**

- **分组**：评分只包含 TP/TN，另列 `TP 且 IoU > 0.7` 子集；FP/FN/invalid 只显示输入数量，评分显示 `N/A`。TN 区分真实视频和 fake 视频的正常背景。`--min-iou` 只改变高 IoU 子集，不过滤 TP。
- **均分**：`Overall_proposal` 中每条已评分解释权重相同，不先按视频平均；`Macro_domain` 则先算各域均分，再对有有效分数的域等权平均。每项均报告有效数量。
- **域来源**：有参考时取对应 annotation 的 `tool_domain`，缺少该键时取 `model`；无参考时先取视频级字段，再取视频内唯一标注域。多个域记为内部标记 `mixed`（不是 `tool_mixed`），无域的真实视频记为 `real`；来源及域列表均保留。

这些分数表示**分类正确样本的解释质量**。比较模型时，同时报告分类表现和参与评分的样本数。

**文本解释质量与定位准确性的相关性分析**

分析 `Explanation` 分数越高时，预测 `Interval` 是否也越接近 GT：

- **参与样本**：已完成文本评分的 TP，包括低 IoU 的 TP。IoU 取最终预测 `Interval` 与视频全部 GT 区间的最大值。
- **比较方法**：按 IoU 的 `[0,0.3]`、`(0.3,0.5]`、`(0.5,0.7]`、`(0.7,1]` 四组报告文本均分和有效数量；另用全部有效配对计算各评分项及 RQ 与 IoU 的 Spearman 相关系数。正相关表示分数越高，IoU 往往也越高。全体及各 `tool_domain` 分别统计。
- **1000 次重采样**：若有 N 个视频，每轮随机抽 N 次视频（允许重复），带上所抽视频的全部有效 proposal，用已有分数和 IoU 重算相关系数。重复 1000 轮，用结果的 2.5% 和 97.5% 分位数估计 95% 置信区间，反映统计结果的不确定性；**不重新调用裁判**。seed=42，`--bootstrap-repeats 0` 可关闭此步骤。

缺失评分不参与计算；样本不足或数值无变化等导致无法计算的结果留空。

## 环境与裁判模型

所有命令均在项目根目录 `Msloc_code/` 下运行。复用第二阶段的 `msloc_qwen35` 环境；安装步骤见[主 README 的环境配置](../../README.md#环境配置)。

正式评测用 27B，单卡 24GB 用 4B 调试；9B 按需下载用于对照。

```bash
conda activate msloc_qwen35
hf download Qwen/Qwen3.5-4B --local-dir ../Qwen/Qwen3.5-4B
hf download Qwen/Qwen3.5-9B --local-dir ../Qwen/Qwen3.5-9B
hf download Qwen/Qwen3.5-27B --local-dir ../Qwen/Qwen3.5-27B
```

均使用官方原始权重、纯文本输入、BF16 和 `thinking=false`。4B 用于验证流程，27B 用于正式评分。

| 裁判 | `--model` | `--output` |
|---|---|---|
| Qwen3.5-4B | `../Qwen/Qwen3.5-4B` | `../MSLoc_data/Qwen/text_rq/grpo_from_sft_4b` |
| Qwen3.5-9B | `../Qwen/Qwen3.5-9B` | `../MSLoc_data/Qwen/text_rq/grpo_from_sft_9b` |
| Qwen3.5-27B | `../Qwen/Qwen3.5-27B` | `../MSLoc_data/Qwen/text_rq/grpo_from_sft_27b` |

## vLLM 加速

**下面的评测命令已使用 vLLM，无需额外开关或启动 `vllm serve`。** `msloc_qwen35` 环境的 `Qwen/requirements.txt` 已固定 `vllm==0.19.1`；TextRQ 在每张卡上直接创建一个 vLLM 引擎，使用纯文本输入、BF16、`thinking=false`。

代码已开启前缀缓存（复用相同提示词前缀的计算）和分块预填充（将长输入分块，与输出生成一起调度）。前缀缓存加速的是重复输入的处理，不会省掉长 CoT 的逐 token 生成；本任务关闭 thinking，只生成分数和简短理由。参见 [vLLM 前缀缓存](https://docs.vllm.ai/en/v0.19.1/features/automatic_prefix_caching/)和[调度说明](https://docs.vllm.ai/en/v0.19.1/configuration/optimization/)。

| 参数 | 控制什么 |
|---|---|
| `--batch-size 32` | 每卡每批提交 32 个裁判请求，同时将引擎的并发序列上限设为 32；不是 32 个 proposal |
| `--max-num-batched-tokens 8192` | 引擎每个调度步处理的 token 总预算，不是每条请求的长度 |
| `--max-model-len 8192` | 单条请求的输入加输出长度上限 |
| `--max-new-tokens 768` | 每次裁判回答的长度上限；提前结束就停止，不会固定生成 768 个 token |
| `--gpu-memory-utilization 0.9` | 引擎的显存预算比例，不是 GPU 算力利用率 |

当前实现等一批请求全部结束后再提交下一批。单卡 24GB、4B 调试先沿用以上参数；吞吐优化需结合实际耗时和显存调整。启动及推理日志在输出目录的 `logs/rank_0.log`；首次启动耗时应与评分耗时分开看。

## 运行

先验证 GT 对应、公共提示词隔离、计分、分组统计和续跑逻辑（不加载模型）：

```bash
python -m unittest discover -s Qwen/TextRQ -p "test_text_rq.py" -v
```

下面使用 27B 裁判进行八卡全量评测，读取 GRPO-from-SFT 输出，结果写入 `../MSLoc_data/Qwen/text_rq/grpo_from_sft_27b/`。每张卡加载完整裁判并处理一份样本；不同裁判不共用评分缓存。

```bash
python Qwen/TextRQ/evaluate_text_rq.py \
  --predictions ../MSLoc_data/Qwen/grpo_from_sft_eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --model ../Qwen/Qwen3.5-27B \
  --common-prompt Qwen/prompts/text_rq_common.txt \
  --fluency-prompt Qwen/prompts/text_rq_fluency.txt \
  --consistency-prompt Qwen/prompts/text_rq_consistency.txt \
  --factual-prompt Qwen/prompts/text_rq_factual.txt \
  --output ../MSLoc_data/Qwen/text_rq/grpo_from_sft_27b \
  --devices 0,1,2,3,4,5,6,7 \
  --min-iou 0.7 \
  --sample-count -1 \
  --bootstrap-repeats 1000 \
  --batch-size 32 \
  --max-model-len 8192 \
  --max-new-tokens 768 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --temperature 0 \
  --top-p 1 \
  --top-k -1 \
  --resume none
```

- **单卡 24GB 调试 32 条**：改 `--model ../Qwen/Qwen3.5-4B --devices 0 --sample-count 32 --output ../MSLoc_data/Qwen/text_rq/grpo_from_sft_4b_debug`；数量不能超过可评分的 TP/TN 数。调试时可用 `--bootstrap-repeats 0` 跳过置信区间计算。
- **27B 单卡全量**：显存足够容纳完整模型及缓存时，只改 `--devices 0`；24GB 卡使用上面的 4B 调试设置。`--batch-size 32` 表示每卡每批 32 个裁判请求。
- **只检查任务清单**：加 `--prepare-only`，不加载裁判；开始评分时去掉它，改为 `--resume auto`，其余参数保持一致。
- **换待评模型输出**：替换 `--predictions` 和 `--output`。调试、全量或不同裁判使用不同输出目录。

`--sample-count -1` 评全部可评分的 TP/TN；正数在筛选后按 TP/TN 比例抽样（seed=42）。两类均存在且抽样数至少为 2 时，两类都会保留。样本在评分前选定，续跑不重抽。

产物保存在 `--output` 目录下：

| 用途 | 查看位置 |
|---|---|
| 看结果 | `reports/default/report.txt`：评分汇总；同目录 `report.csv`：表格；`iou_analysis.json`：评分与 IoU 的相关性 |
| 查单条评分 | `scores.jsonl`：预测文本、GT 参考、IoU、分数及理由 |
| 查运行情况 | `selection_summary.json`：样本数量及解析失败数量；`failures/`、`logs/`：错误记录和运行日志 |

其余为配置、样本清单和续跑文件。**保留整个输出目录**，后续可续跑或重新生成报表，无需重新评分。

## 重新生成报表（无需重新评分）

评分完成后，可用已保存的分数修改分组、IoU 阈值或打印格式，不需要 GPU。

```bash
python Qwen/TextRQ/text_rq_report.py \
  --run ../MSLoc_data/Qwen/text_rq/grpo_from_sft_27b \
  --output ../MSLoc_data/Qwen/text_rq/grpo_from_sft_27b/reports/by_domain \
  --group-by tool_domain \
  --min-iou 0.7
```

报表写入 `--output`：`report.txt`、`report.csv`、`report_summary.json`；同目录重跑会覆盖这三份报表，原始评分不变。

- **换分组**：修改 `--group-by`；`tool_domain` 按工具域，`outcome` 按 TP/FP/FN/TN/invalid，其余选项见 `--help`。
- **换高 IoU 阈值**：修改 `--min-iou`，例如 `0.7` 表示统计 TP 且 IoU > 0.7 的子集。
- **改打印或统计代码**：分别修改 `Qwen/TextRQ/text_rq_report.py` 或 `Qwen/TextRQ/text_rq_metrics.py`，再运行上面的命令。

修改裁判提示词或 GT 参考内容，需要重新评分。

参考 [vLLM 结构化输出](https://github.com/vllm-project/vllm/blob/v0.19.1/examples/offline_inference/structured_outputs.py)、[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)、[Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) 和 [SciPy Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)。固定 seed 不保证不同 GPU／批次调度下逐字一致。
