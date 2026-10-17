# MSLoc demo：总入口与单卡运行指南

这套 demo 展示正式实验的训练流程：SFT 训练 LoRA → merge 导出固定教师 → OPSD 继续训练学生 LoRA → GRPO 继续训练 OPSD LoRA，并展示回答、loss、奖励和耗时。

所有命令在 `Msloc_code/` 根目录执行。新增代码只在 `Qwen/demo/`；复用现有 Qwen 模板、数据规则和奖励函数，不修改它们。

`train.py` 与正式实验调用同一套 ms-swift SFT/GKD/GRPO 训练器，demo 缩小数据、缩短训练并增加可视化。Ray 是另写的多角色训练循环，用于验证并行和同步。

**单卡路线：加载正式实验权重 → 教师检查 → HF 训练与回载。Ray 单卡是显存允许时的额外流程验证；多卡 DDP 和跨 GPU 并行收益需要多卡机器验证。** 支持 `--devices 0` 不等于任意显存容量都能跑。

<a id="navigation"></a>
## 文档导航：

| 文档 | 回答什么问题 | 什么时候看 |
|---|---|---|
| [README：本页](README.md) | 有什么硬件、先跑什么、命令是什么、结果在哪里？ | **先看；日常操作入口** |
| [训练讲解提纲](训练讲解提纲.md) | SFT / OPSD / GRPO 每一步在算什么，怎么讲？ | 跑之前理解流程，跑之后对照真实结果讲解 |
| [Ray 多角色训练](Ray多角色训练.md) | 角色如何分工、权重如何同步、单卡能验什么、多卡怎么比较速度？ | 单卡基础流程之后按需看；多卡实验的操作手册 |
| [第二阶段 Qwen 面试问题清单](第二阶段Qwen面试问题清单.md) | 面试可能追问什么、原项目代码在哪里？ | 最后自测；这是原项目问题清单，不是新 demo 的执行步骤，引用行号应核对当前代码 |

阅读顺序：**本页选路线 → 讲解提纲理解算法 → 执行所选路线 → 面试清单自测**。选择 Ray 路线时再进入 Ray 文档，不需要把所有文档从头到尾重复读。

本页直达：[单卡能做什么](#single-gpu) · [多卡增加什么](#multi-gpu) · [环境与权重](#setup) · [数据准备](#data) · [查看回答/检查教师](#preflight) · [单卡训练](#single-training) · [单卡 vLLM 对比](#single-vllm) · [Ray 命令入口](#multi-ray) · [查看训练后的回答](#reload) · [中断后接着跑](#resume) · [结果文件](#deliverables)

## 1. 按硬件选择演示范围

| 演示项目 | 单卡能做什么 | 多卡额外能验证什么 |
|---|---|---|
| 完整权重推理、教师检查 | 分别运行学生和教师，生成回答/区间可视化；二者预检不必同时驻留 | 不同 GPU 分担样本，检查并行推理吞吐 |
| SFT | 显存足够时监督训练、保存与回载；已有 SFT 权重可跳过 | DDP 梯度同步及多卡训练吞吐 |
| OPSD / GRPO，`train.py` | 显存足够时完整采样、监督/奖励、反传更新；保持全局 batch=8 | 多 rank 训练及分布式一致性 |
| HF / vLLM 后端对比 | 显存足够时测**同一张卡**的后端差异 | 同卡数下再次比较后端差异；不能将增加 GPU 的收益混入后端收益 |
| Ray 多角色，`ray_launch.py` | 显存足够时三个模型角色共享一张 GPU，验证进程分工、版本和权重同步；只有一个 Trainer，**不执行 DDP** | 多 Trainer DDP、多生成/评分副本，以及角色分布到不同 GPU 的并行效果 |
| CPU 数学/传输测试 | 在装好依赖的服务器上验证损失、优势和 Ray tensor 传输，无需分配 GPU | 不能替代任何 GPU 模型或 NCCL 测试 |

<a id="single-gpu"></a>
### 只有单卡

1. [配置环境与权重](#setup)、[准备数据](#data)。已有正式实验的 SFT LoRA 时直接使用，无需重训。
2. [分别运行学生、教师推理并检查教师效果](#preflight)，查看 `predictions.html`。教师检查通过后才能训练 OPSD；GRPO 可以跳过教师检查。
3. [用 `train.py --backend transformers --devices 0` 跑 OPSD / GRPO](#single-training)，再[回载 checkpoint](#reload)。这是单卡训练的优先路线，不要求先启动 Ray 或 vLLM。
4. 显存有余量，再做[单卡 vLLM 后端对比](#single-vllm)或 [Ray 单卡同步流程验证](Ray多角色训练.md#single-gpu)。这两项是独立扩展，不是完成单卡基础 demo 的前提。
5. 对照[讲解提纲](训练讲解提纲.md)讲清实际跑过的内容，多卡部分标为“已实现、待多卡实测”。

**单卡首先受显存容量限制。** HF OPSD 可能同时需要学生和独立教师，GRPO 还涉及参考策略；vLLM 会增加生成引擎和缓存。Ray 单卡则明确同时驻留 Trainer、Rollout、Scorer 三个模型进程，通常比基础 HF 路线更吃显存。LoRA 减少可训练参数与优化器状态，不会消除底座、教师或生成副本。能否运行取决于 GPU 型号、可用显存和具体配置。

如果某个训练步骤 OOM，记录显存与报错；已经完成的推理、教师检查和 CPU 测试仍可单独展示，但不能称作训练已跑通。当前代码不会自动减小 batch、缩短视频、量化或卸载。`--plan-only` 只检查配置，不能证明显存足够。把步数从 32 改为 8、保存间隔改为 4，只会缩短验收时间，不能解决每步峰值显存。

<a id="multi-gpu"></a>
### 有多卡后：再增加这些实验

重点进入 [Ray 多卡路线](Ray多角色训练.md#multi-gpu)：验证多个 Trainer 的 DDP、多个生成/评分副本及逐轮同步，再做串行 HF / 并行 HF / 并行 vLLM 三组同资源对比。默认八卡分工见下文。

**Ray 入口已改为实际的多角色同步训练**，不再包装 `train.py`。八卡默认 4 个 DDP 训练 worker、2 个生成 worker、2 个教师／参考 worker；GRPO 另有 CPU 奖励 worker。每轮完成版本校验、生成、监督信号、更新及权重同步后，才进入下一轮。操作与验收见 [Ray 多角色训练](Ray多角色训练.md)。

若要运行 HF 多卡基线，把本页 HF 命令的 `--devices 0` 改为 `--devices 0,1,2,3,4,5,6,7`，使用新的输出目录，其他方法参数不变。不要从单卡运行目录以 `--resume auto` 改成八卡继续。HF 与 Ray 是两套训练入口，各自内部比较；**Ray 对比工具要求相同 GPU 数和角色布局，不支持把单卡与八卡结果混算为并行调度加速比。**

<a id="setup"></a>
## 2. 环境与模型

### 完整重建命令（单卡、多卡共用）

**demo 沿用原 Qwen 第二阶段环境，训练依赖版本不变。只有 Ray 多角色功能额外要求 `ray[default]==2.59.0`。** `demo/requirements.txt` 直接引用 `Qwen/requirements.txt`，没有另一套训练依赖。

原 `msloc_qwen35` 环境若已按当前仓库配置完成，可以直接使用；运行 Ray 时再安装 `Qwen/demo/requirements-ray.txt`。下面是独立重建命令，仅将环境名换为 `msloc_demo`。在 Linux 服务器的 `Msloc_code/` 执行，单卡、多卡相同。

**1. 创建环境、安装 PyTorch（原流程，仅环境名改变）。**

```bash
conda create -n msloc_demo python=3.12 -y  # 调整：原名 msloc_qwen35；Python 3.12 不变
conda activate msloc_demo                # 调整：激活独立 demo 环境
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128  # 原有：版本及 cu128 不变
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil  # 原有：构建依赖不变

```

**2. 检查编译工具（原流程）。**

```bash
nvcc --version                          # 原有：检查 CUDA Toolkit 版本
g++ --version                           # 原有：检查 C++ 编译器
ninja --version                         # 原有：检查构建工具
```

**`nvcc` 已显示 `release 12.8`：跳过下面的 CUDA 安装块。** 找不到 `nvcc` 或版本不同：先切换到已有的 12.8；没有可用的 12.8 时才执行：

```bash
conda install -y -c nvidia/label/cuda-12.8.0 cuda-toolkit=12.8.0  # 原有：缺少匹配工具包时才安装
nvcc --version
```

GCC/G++ 沿用原实验的兼容版本，不额外指定 13。继续检查工具包路径：

```bash
# 原有：检查 PyTorch 能否找到 CUDA 编译工具包
python -c 'from pathlib import Path; from torch.utils.cpp_extension import CUDA_HOME; print("CUDA_HOME:", CUDA_HOME); assert CUDA_HOME and (Path(CUDA_HOME) / "bin/nvcc").is_file()'
```

工具包版本看 `nvcc --version`，不是 `nvidia-smi` 中的 CUDA Version。Conda 安装工具包不会更新宿主机驱动。

**3. 安装依赖（原依赖 + Ray）。**

```bash
# 原有依赖、调整入口：demo/requirements.txt 仅引用 ../requirements.txt
# ms-swift 4.5.3、Transformers 5.9.0、TRL 0.27.0、PEFT 0.20.0、vLLM 0.19.1 均不变
# DeepSpeed、Liger、FLA、视频与绘图库也原样继承
python -m pip install --only-binary=av,decord --no-build-isolation -r Qwen/demo/requirements.txt
python -m pip install causal-conv1d==1.7.0 --no-build-isolation  # 原有：版本不变
python -m pip install flash-attn==2.8.3 --no-build-isolation     # 原有：版本不变
python -m pip install -r Qwen/demo/requirements-ray.txt  # 新增：Ray 多角色需要；只跑 HF demo 可跳过
```

仅当对应扩展因预编译 wheel 下载失败而安装退出时，在同一终端执行对应源码编译命令：

```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install causal-conv1d==1.7.0 --no-build-isolation  # 原有：源码编译方式不变
FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install flash-attn==2.8.3 --no-build-isolation  # 原有：源码编译方式不变
```

**4. 检查安装（原流程）。**

```bash
python -m pip check                      # 原有：检查依赖冲突
CUDA_VISIBLE_DEVICES=0 python Qwen/check_env.py  # 原有检查脚本；调整为显式使用 GPU 0
```

`check_env.py` 检查 GPU、CUDA 扩展前向/反向和视频读写，测试视频输出到 `../MSLoc_data/Qwen/env_check/`。新终端使用 `conda activate msloc_demo`。已有的缓存设置可沿用，demo 不要求另设缓存或导出环境清单。

第一阶段的 `msloc` 是 Python 3.10 + PyTorch 2.6.0 cu124 + DeMamba 依赖，不能作为本 demo 的安装基线。导入完整 HF 权重不需要复制权重导出方的 Conda 环境。

安装依据：[PyTorch cu128 官方命令](https://pytorch.org/get-started/previous-versions/)、[CUDA 12.8 安装指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html)、[flash-attn 2.8.3 安装实现](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/setup.py)、[causal-conv1d 1.7.0 安装实现](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/setup.py)。

### 模型下载

开始前备齐权重。已有对应文件时不用重复下载：

```bash
hf download Qwen/Qwen3.5-4B --local-dir ../Qwen/Qwen3.5-4B
hf download cross-encoder/nli-deberta-v3-small --local-dir ../MSLoc_data/Qwen/ckpt/nli-deberta-v3-small
```

<a id="data"></a>
## 3. 准备小规模真实数据（单卡、多卡共用）

直接使用已有 `opsd_data/train.jsonl`、`targets.jsonl` 及视频片段，不重新生成 proposal。若尚未准备这些数据，先按根目录 README 的 `prepare_opsd.py` 步骤准备。

**`--samples 64` 表示整个 demo 一共抽取 64 个 proposal 视频片段，fake 和 real 合计 64 个。** 各阶段共用这一批片段，不分别抽取。按原数据的 fake/real 比例随机抽样，seed=42。

```bash
python Qwen/demo/prepare.py \
  --opsd-dataset ../MSLoc_data/Qwen/opsd_data/train.jsonl \
  --opsd-targets ../MSLoc_data/Qwen/opsd_data/targets.jsonl \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --samples 64 --output ../MSLoc_data/Qwen/demo/data --resume none
```

输出保存在 `../MSLoc_data/Qwen/demo/data/`：

| 文件 | 样本数量与用途 |
|---|---|
| `opsd.jsonl` | 这 64 个片段，用于学生和教师推理；OPSD 训练时只用教师检查通过的部分，因此实际训练样本可能少于 64 个 |
| `grpo.jsonl` | 同样的 64 个片段，用于 GRPO 训练 |
| `sft.jsonl` | 只取这 64 个片段中的 fake，用于 SFT 训练 |
| `targets.jsonl` | 这 64 个片段对应的真实标签和时间区间 |
| `manifest.json` | 记录抽样设置、原数据和抽样后的 fake/real 数量 |

如果教师检查提示缺少 fake 或 real，说明抽取数量太少，将 `--samples` 调大后重新准备数据。

视频换过存放位置时，在上面的命令中加 `--path-map 旧目录=新目录`。每个视频旁的 `.timestamps.json` 也要一起复制，它记录视频帧对应的真实时间。

<a id="preflight"></a>
## 4. 单卡：查看回答，检查教师效果

本节使用正式实验已有的 SFT 权重。只看模型回答，完成 **4.2 的学生推理**即可；要训练 OPSD，则完成本节全部步骤。GRPO 不需要教师检查，可以直接进入 [5.3](#grpo-training)。

### 4.1 合并 SFT 权重，准备教师

将原始底座和 SFT LoRA 合成一个完整教师模型。已经有对应的 `sft_teacher/` 时跳过此步。

**这一步做什么：** LoRA 保存的是训练得到的参数增量。merge 将它加到原始权重上，即 `合并权重 = 原始权重 + LoRA 增量`，再保存为完整模型。这一步不训练；得到的教师包含 SFT 学到的能力，后续 OPSD 中保持冻结。

```bash
python Qwen/merge_sft_teacher.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/sft/last \
  --output ../MSLoc_data/Qwen/sft_teacher \
  --devices 0
```

输入是 `--model` 指定的原始底座和 `--adapter` 指定的 SFT LoRA；完整教师保存到 `../MSLoc_data/Qwen/sft_teacher/`。学生仍继续使用原始底座和原来的 LoRA。

### 4.2 分别运行学生和教师推理

两者都处理前面抽出的 64 个片段。学生使用普通提示词；教师额外看到真假标签、正确时间区间和解释提示。

**这一步怎么算：** 模型读取每个片段的 40 帧，逐个生成回答 token；这里每次选概率最高的 token，即贪心生成。输出包含解释和定位区间，正常片段输出 `Real`。预测时间相对于当前片段，例如原视频 `[100,140]` 秒的片段中，`[10,25]` 表示原视频的 `[110,125]` 秒。这里只推理，不更新参数。

```bash
# 学生：原始底座 + SFT LoRA
python Qwen/demo/inspect_model.py \
  --mode student --model ../Qwen/Qwen3.5-4B --adapter ../MSLoc_data/Qwen/sft/last \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/student_before --resume none

# 教师：合并后的完整模型
python Qwen/demo/inspect_model.py \
  --mode teacher --model ../MSLoc_data/Qwen/sft_teacher --adapter none \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/teacher_before --resume none
```

回答分别保存在 `student_before/predictions.jsonl` 和 `teacher_before/predictions.jsonl`，两个目录都位于 `../MSLoc_data/Qwen/demo/`。

若已有完整 HF 模型，对应命令改为 `--model <实际模型目录> --adapter none`。完整教师不需要再合并。

### 4.3 查看回答，判断教师能否用于 OPSD

先生成对比页面，再运行教师检查：

**为什么先检查：** 给教师正确答案作为提示，也不能保证它一定比学生回答得好。脚本比较同一批片段上的分类、格式和定位结果；定位用 IoU，即预测区间与正确区间的交集长度除以并集长度。先确认教师整体优于学生，再保留教师回答合格的样本用于 OPSD。

```bash
python Qwen/demo/view_predictions.py \
  --runs ../MSLoc_data/Qwen/demo/student_before ../MSLoc_data/Qwen/demo/teacher_before \
  --data ../MSLoc_data/Qwen/demo/data --output ../MSLoc_data/Qwen/demo/predictions.html

python Qwen/demo/teacher_gate.py \
  --student-run ../MSLoc_data/Qwen/demo/student_before \
  --teacher-run ../MSLoc_data/Qwen/demo/teacher_before \
  --data ../MSLoc_data/Qwen/demo/data --min-iou 0.5 \
  --output ../MSLoc_data/Qwen/demo/gate.json
```

打开 `../MSLoc_data/Qwen/demo/predictions.html`，查看视频、两者回答，以及预测区间和正确区间。页面中的视频链接需要原视频文件仍在对应位置。

检查通过需要满足三点：教师在 fake 上的平均定位 IoU 高于学生；fake、real 的分类准确率都不降低；格式错误不增加。

结果写入 `../MSLoc_data/Qwen/demo/gate.json`。脚本还会保留教师定位 IoU ≥ 0.5 的 fake 和判断正确的 real，写入同目录的 `gate.jsonl`，供 OPSD 训练使用。筛选后仍需包含两类。检查失败时先查看回答和指标，处理后重新检查；通过后进入 **5.2**。

<a id="single-training"></a>
## 5. 单卡：运行短训练

本节沿用正式实验的训练方法，每次只跑 **32 次参数更新，每 8 次保存一次**。`--steps` 是更新次数，`--samples` 是抽取的数据总数，两者含义不同。

| 参数 | 含义 |
|---|---|
| `--devices 0` | 使用第 0 张 GPU；单卡运行时保留此参数 |
| `--global-batch-size 8` | SFT/OPSD 每次更新用 8 个输入；GRPO 每次用 2 个输入、每个生成 4 个回答 |
| `--tuner lora --lora-rank 16 --lora-alpha 32` | 沿用正式实验的语言 LoRA 设置，视觉部分保持冻结 |
| `--max-length 8192 --max-new-tokens 256` | 输入加回答最多 8192 个 token，生成回答最多 256 个 token |
| `--resume none` | 开始一次新训练；中断后怎么接着跑见 [7.2](#resume) |

下面每一步先说明计算过程，再给运行命令。更详细的公式和手算例子见 [训练讲解提纲](训练讲解提纲.md)。以下命令都使用 Transformers 生成回答。

### 5.1 SFT：可选，展示监督训练

用原始 Qwen 和这批数据中的 fake 样本训练 LoRA。已有正式 SFT 权重时可跳过。

**每一步怎么算：**

1. 用标注构造标准回答：一条或三条解释，加片段内的正确时间区间。
2. 把视频、提示词和标准回答送入模型。预测某个回答 token 时，前面接的是标准答案中的 token，这叫 teacher forcing。
3. 对回答部分计算交叉熵：模型给正确 token 的概率越低，loss 越大。提示词本身不作为答案监督。
4. 反向传播，累积到本次更新所需的 8 个样本后更新 LoRA。原始底座和视觉部分保持不变。

SFT 学习的是“怎样写出标准回答”，这一阶段不使用奖励函数。

```bash
python Qwen/demo/train.py \
  --stage sft --backend transformers --model ../Qwen/Qwen3.5-4B --adapter none \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --tuner lora --lora-rank 16 --lora-alpha 32 \
  --steps 32 --save-steps 8 --global-batch-size 8 --learning-rate 1e-4 \
  --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/sft --resume none
```

输出 LoRA：`../MSLoc_data/Qwen/demo/sft/checkpoint-32/`。
输出训练曲线：`../MSLoc_data/Qwen/demo/sft/report/report.html`。

这一步用于展示 SFT 的计算和参数更新。后续命令仍使用正式实验的 `sft/last`，不会自动改用这次短训练的权重。

### 5.2 OPSD：继续训练 SFT LoRA

先完成第 4 节教师检查。学生从 `sft/last` 开始更新，教师使用合并后的 `sft_teacher` 并始终保持不变。

已有 LoRA 直接通过 `--adapter` 加载，不用修改 `adapter_config.json`。其中 `target_modules` 写成正则表达式也是 [PEFT 支持的格式](https://huggingface.co/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig)。

`train.py` 读取教师检查和 vLLM 对齐结果时，允许训练入口 `train.py` 自身发生变化；其余记录的 Python 文件仍逐一核对。

**每一步怎么算：**

1. 学生只看视频和普通提示词，按当前模型的概率采样回答。这一步生成的回答就是本轮训练所用的回答。
2. 学生在普通提示词下，重新计算这份回答每个位置的下一个 token 概率分布。
3. 教师在特权提示词下，沿着**同一份学生回答的前缀**计算下一个 token 分布。教师不另外生成一份标准答案。
4. 用 JSD 衡量两个分布的差异，比较范围是完整词表。`beta=0.5` 表示计算 JSD 时师生分布各占一半。
5. 只对学生反向传播并更新原有 LoRA。教师不更新；下一轮用更新后的学生重新生成回答。

这里 `lmbda=1` 表示全部使用学生自己生成的回答，`sft_alpha=0` 表示不再额外计算 SFT 的标准答案 loss。它教的是学生在自己实际写出的前缀后，应该怎样继续生成。

```bash
python Qwen/demo/train.py \
  --stage opsd --backend transformers \
  --model ../Qwen/Qwen3.5-4B --adapter ../MSLoc_data/Qwen/sft/last \
  --teacher-model ../MSLoc_data/Qwen/sft_teacher --teacher-adapter none \
  --teacher-gate ../MSLoc_data/Qwen/demo/gate.json \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --tuner lora --lora-rank 16 --lora-alpha 32 \
  --steps 32 --save-steps 8 --global-batch-size 8 --learning-rate 2e-5 \
  --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/opsd_hf --resume none
```

输出 LoRA：`../MSLoc_data/Qwen/demo/opsd_hf/checkpoint-32/`。
输出训练曲线：`../MSLoc_data/Qwen/demo/opsd_hf/report/report.html`。

**单卡在生成回答时显存不足：** 上面的命令每轮一起生成 8 份回答，学生和教师也同时占用显存。SFT 能跑通不代表这一步也能装进同一张卡。可显式增加 `--generation-batch-size 1`：每轮生成 1 份回答、计算梯度，累积 8 次后更新一次参数，`--global-batch-size 8` 保持不变。不传新参数时仍沿用原来的生成批量。

例如，若第一步就报 `CUDA out of memory`、进度仍为 `0/32`，更新 `train.py` 后可以运行：

```bash
python Qwen/demo/train.py \
  --stage opsd --backend transformers \
  --model ../Qwen/Qwen3.5-4B --adapter ../MSLoc_data/Qwen/sft/last \
  --teacher-model ../MSLoc_data/Qwen/sft_teacher --teacher-adapter none \
  --teacher-gate ../MSLoc_data/Qwen/demo/gate.json \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --tuner lora --lora-rank 16 --lora-alpha 32 \
  --steps 32 --save-steps 8 --global-batch-size 8 --generation-batch-size 1 \
  --learning-rate 2e-5 --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/opsd_hf --resume none --clean
```

`--clean` 清理本次失败的 `opsd_hf` 目录后重新开始，不删除 SFT、教师权重或教师检查结果。已有需要保留的训练结果时，改用新的 `--output` 目录。输出位置与上面相同。

这使用的是 [ms-swift 原有的生成批量参数](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/args_mixin.py)，训练目标和每次更新的样本数不变；分批会改变随机采样顺序，不保证生成相同回答。它能降低生成阶段的峰值显存，但尚未在 24 GB 卡上验证整个训练能否完成。若仍然溢出，需要根据新的报错位置判断，不能保证只改这一项就足够。GRPO 的生成批量还必须是 `--num-generations` 的整数倍，例如 4，不能照抄 1。比较速度时，两次运行应使用相同的生成批量。

如果继续做全参数训练，改为 `--model <完整学生目录> --adapter none --tuner full`，`--teacher-model` 填独立完整教师目录；第 4 节也要使用这两份模型检查。视觉部分仍冻结。

训练完成后，按 [7.1](#reload) 加载新权重查看回答。

<a id="grpo-training"></a>
### 5.3 GRPO：继续训练 OPSD LoRA

默认使用正式实验的 `opsd/last`。模型为每个片段生成多个回答，再根据定位、格式和解释三项奖励更新 LoRA。

**每一步怎么算：**

1. 取 2 个片段，每个采样 4 个回答，共 8 个回答。采样参数为 `temperature=0.8`、`top_p=0.95`。
2. 给每个回答打分：`总奖励 = 定位奖励 + 0.1 × 格式奖励 + 0.3 × 解释奖励`。定位检查真假判断和时间区间；格式检查能否解析答案；解释由冻结的 NLI 模型检查事实覆盖和矛盾。解释奖励只用于定位 IoU ≥ 0.3 且正确解析为 fake 的异常样本。
3. 在同一个片段的 4 个回答内，计算 `优势 = (奖励 − 组内平均奖励) / (组内标准差 + 0.0001)`。高于平均水平的回答得到正优势，低于平均水平的得到负优势。
4. 重新计算回答 token 的概率，用优势构造策略 loss：提高正优势回答的概率、降低负优势回答的概率。概率比使用 clip=0.2 的裁剪，并加入系数为 0.02 的 KL 惩罚，约束模型偏离初始参考模型的程度。
5. 这批回答只更新一次学生 LoRA，然后重新采样。参考模型固定为本次 GRPO 开始时的权重，奖励函数和 NLI 都不更新。

奖励本身不需要求梯度；它通过“优势 × 策略概率”影响学生的更新。同组回答的奖励完全相同时，优势为 0，这一组没有奖励带来的区分信号，但 KL 项仍可能产生梯度。

```bash
python Qwen/demo/train.py \
  --stage grpo --backend transformers \
  --model ../Qwen/Qwen3.5-4B --adapter ../MSLoc_data/Qwen/opsd/last \
  --data ../MSLoc_data/Qwen/demo/data \
  --nli-model ../MSLoc_data/Qwen/ckpt/nli-deberta-v3-small \
  --devices 0 --tuner lora --lora-rank 16 --lora-alpha 32 \
  --steps 32 --save-steps 8 --global-batch-size 8 --num-generations 4 \
  --learning-rate 1e-6 --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/grpo_hf --resume none
```

`--nli-model` 用于计算解释奖励，在 CPU 上运行，不参与训练。
输出 LoRA：`../MSLoc_data/Qwen/demo/grpo_hf/checkpoint-32/`。
输出奖励和 loss 曲线：`../MSLoc_data/Qwen/demo/grpo_hf/report/report.html`。

按需要替换本条命令的输入：

| 从哪里开始 | 修改哪些参数 |
|---|---|
| 接着本页 5.2 的 OPSD 结果训练 | `--adapter ../MSLoc_data/Qwen/demo/opsd_hf/checkpoint-32` |
| 跳过 OPSD，直接从正式 SFT 权重训练 | `--adapter ../MSLoc_data/Qwen/sft/last` |
| 继续做全参数训练 | `--model <完整模型目录> --adapter none --tuner full` |

每次换起点都使用新的 `--output` 目录。训练完成后按 [7.1](#reload) 查看新模型的回答。

<a id="single-vllm"></a>
## 6. 单卡可选：比较 Transformers 和 vLLM 的速度

先跑通第 5 节。需要比较生成速度时再做本节；vLLM 和训练模型同时占用这张 GPU，显存需求会增加。

**加速的是哪一步：** vLLM 负责生成学生回答，教师计算、奖励计算和学生反向传播仍按原方法进行。每次更新后，训练器把新的学生权重同步给 vLLM，再开始下一轮生成。因此要比较的是包含同步开销的整步耗时，而不只是生成耗时。

### 6.1 检查两种后端是否读到了相同的视频输入

**这一步检查什么：** 同一视频经过 Transformers 和 vLLM 处理后，文本 token、视频像素和时空网格应一致，尤其不能改变原有的非均匀采样时间。输入相同后才有条件比较后端速度；这不要求两种后端生成完全相同的回答。

```bash
# 检查底座的输入处理器，这一步不加载 LoRA
python Qwen/demo/inspect_model.py \
  --mode alignment --model ../Qwen/Qwen3.5-4B --adapter none \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --max-length 8192 --max-new-tokens 256 --vllm-memory 0.5 \
  --output ../MSLoc_data/Qwen/demo/alignment --resume none
```

结果保存在 `../MSLoc_data/Qwen/demo/alignment/alignment.json`。`passed` 为 `true` 后才能继续。若更换完整模型，`--model` 也改为对应目录后重新检查。

### 6.2 用 vLLM 重跑同一项训练

复制 5.2 或 5.3 的完整命令，只修改下面这些参数：

| 参数 | OPSD | GRPO |
|---|---|---|
| `--backend` | `vllm` | `vllm` |
| 新增 `--alignment` | `../MSLoc_data/Qwen/demo/alignment/alignment.json` | 同左 |
| 新增 `--vllm-memory` | `0.35` | `0.35` |
| `--output` | `../MSLoc_data/Qwen/demo/opsd_vllm` | `../MSLoc_data/Qwen/demo/grpo_vllm` |

**初始模型、adapter、数据和其余训练参数与对应的 HF 运行保持一致。** 例如两次 OPSD 都从同一份 `sft/last` 开始。`--vllm-memory` 是生成引擎的显存预算，不包括训练模型占用。

### 6.3 生成速度对比报告

完成哪一组，就运行哪一条：

```bash
# 比较两次 OPSD 训练
python Qwen/demo/report.py \
  --runs ../MSLoc_data/Qwen/demo/opsd_hf ../MSLoc_data/Qwen/demo/opsd_vllm \
  --warmup-steps 4 --output ../MSLoc_data/Qwen/demo/opsd_compare

# 比较两次 GRPO 训练
python Qwen/demo/report.py \
  --runs ../MSLoc_data/Qwen/demo/grpo_hf ../MSLoc_data/Qwen/demo/grpo_vllm \
  --warmup-steps 4 --output ../MSLoc_data/Qwen/demo/grpo_compare
```

打开 `../MSLoc_data/Qwen/demo/opsd_compare/report.html` 或 `grpo_compare/report.html`。报告跳过前 4 步，比较每步耗时、生成速度和训练曲线，计时包含权重同步与监测的开销。

这里只比较同一张卡上两种生成后端的速度。是否加速以实际结果为准，同时检查生成长度和训练后回答是否正常。各阶段计时可能重叠，不能直接相加；显存图是运行时采样，不代表所有进程的最高显存占用。

<a id="multi-ray"></a>
## 多卡分支：Ray 训练

Ray 将生成、教师或参考模型、训练等工作分给不同进程。按硬件选择对应文档：

**每轮如何配合：** 生成进程用当前学生权重生成回答 → 教师或奖励进程给出训练信号 → 训练进程更新学生 → 将新权重发给所有生成进程 → 确认加载正确后再开始下一轮。教师和 GRPO 参考模型始终不变。多卡时，不同副本可以并行处理样本，多个训练进程通过 DDP 同步梯度；这里没有跨轮异步生成。

| 硬件 | 入口 | 主要检查什么 |
|---|---|---|
| 单卡，且显存足够 | [Ray 单卡命令](Ray多角色训练.md#single-gpu) | 各进程能否完成训练、同步权重、保存和恢复 |
| 多卡 | [Ray 八卡训练命令](Ray多角色训练.md#multi-training) | 多个训练进程的梯度同步，各角色在不同 GPU 上运行 |
| 多卡，训练已跑通 | [Ray 速度对比命令](Ray多角色训练.md#multi-benchmark) | 相同资源下，串行 HF、并行 HF、并行 vLLM 的耗时差异 |

单卡基础流程不要求运行 Ray。Ray 的 checkpoint 目录和 HF 不同，回载时使用 [Ray 文档中的命令](Ray多角色训练.md#checkpoints)。

<a id="reload"></a>
## 7. 查看训练结果，以及中断后继续运行

### 7.1 加载训练后的模型，查看回答

以下示例加载第 5.2 节输出的 OPSD LoRA。底座仍用原始 Qwen，`--adapter` 换成本次训练保存的 checkpoint。

**为什么要重新加载：** loss 曲线只能说明训练过程中的变化。把保存的模型重新加载并运行推理，可以检查权重是否保存完整、回答是否正常，以及定位结果是否变化。这一步只做推理，不再更新参数。

```bash
python Qwen/demo/inspect_model.py \
  --mode student --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/demo/opsd_hf/checkpoint-32 \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/student_after --resume none

python Qwen/demo/view_predictions.py \
  --runs ../MSLoc_data/Qwen/demo/student_before ../MSLoc_data/Qwen/demo/student_after \
  --data ../MSLoc_data/Qwen/demo/data --output ../MSLoc_data/Qwen/demo/before_after.html
```

打开 `../MSLoc_data/Qwen/demo/before_after.html`，比较同一批片段在 OPSD 训练前后的回答和定位区间。

查看 SFT 或 GRPO 结果时，将 `--adapter` 改为相应训练的 checkpoint，并使用新的推理输出目录。若要比较 GRPO 前后变化，训练前还需用它的初始 OPSD 权重运行一次学生推理，作为比较起点。

如果训练时用了 `--tuner full`，则将 `--model` 改为完整 checkpoint 目录，并传 `--adapter none`。

<a id="resume"></a>
### 7.2 中断后接着跑

重新执行原命令，**把 `--resume none` 改成 `--resume auto`，其他参数不变**。不要加 `--clean`。

| 中断的步骤 | 再次运行时会做什么 |
|---|---|
| 训练 | 从最近一次保存的 checkpoint 继续，恢复模型和优化器；最后一次保存后尚未保存的几步会重跑 |
| 学生或教师推理 | 保留已经写入文件的回答，只处理剩余样本 |
| 数据准备 | 已完成且检查一致则直接复用；未完成则重新生成 JSONL 文件 |

保持原数据、权重路径、代码、GPU 数量和训练总步数不变。恢复功能用于完成原来的运行；换模型、换数据或修改训练设置时，应使用新输出目录开始一次新运行。

若训练在第一次保存前就中断，还没有 checkpoint，需要从头重跑。

### 7.3 从头重跑，或只检查参数

- **从头重跑**：使用新的 `--output` 目录；若确定不要旧结果，也可传 `--resume none --clean`，清空指定输出目录后重跑。
- **重新检查教师**：`teacher_gate.py` 使用新的 `--output` 文件名，后面的 OPSD 命令同步修改 `--teacher-gate`。
- **只检查训练命令**：在 `train.py` 或 `ray_launch.py` 命令末尾加 `--plan-only`。它检查参数和输入文件，不加载 GPU 模型，也不能判断显存是否足够。

换过数据、模型或 demo 代码后，需要重新运行相关推理、教师检查和 vLLM 输入检查，不能继续引用旧检查结果。若之前用旧版均衡抽样生成过数据，也从数据准备这一步重新开始。

<a id="deliverables"></a>
## 8. 跑完后看哪些文件

以下路径均相对于 `../MSLoc_data/Qwen/demo/`：

| 文件 | 看什么 |
|---|---|
| `predictions.html` | 学生和教师的回答、定位区间 |
| `gate.json` | 教师检查是否通过，以及保留多少训练样本 |
| `sft/report/report.html`、`opsd_hf/report/report.html`、`grpo_hf/report/report.html` | 已运行训练的 loss、奖励和耗时曲线 |
| 对应训练目录中的 `checkpoint-32/` | 最后保存的 LoRA 或完整模型 |
| `before_after.html` | 重新加载模型后，训练前后的回答变化 |
| `opsd_compare/report.html`、`grpo_compare/report.html` | 做过 vLLM 对比后，两种后端的速度差异 |

单卡先完成推理、训练、保存后重新加载模型这条流程；OPSD 还要通过教师检查。多卡 Ray 另外查看 [Ray 结果与检查步骤](Ray多角色训练.md#validation)。只展示实际完成的结果，32 步短训练不能代替正式性能评测。

[返回文档导航](#navigation) · [训练讲解](训练讲解提纲.md) · [Ray 操作手册](Ray多角色训练.md) · [面试自测](第二阶段Qwen面试问题清单.md)

## 实现依据

- [Transformers 5.9 视频配置加载](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/video_processing_utils.py)：支持 `processor_config.json` 中的嵌套配置和独立配置文件；师生比较实际加载后的设置。
- 项目现有 `Qwen/common.py`、`opsd_common.py`、`prepare_grpo.py`、`trace_video_template.py`、`grpo_rewards.py`、`grpo_logging.py`。
- [ms-swift 4.5.3 GKD trainer](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/gkd_trainer.py)、[GKD loss](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/gkd_loss.py)、[GRPO trainer](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/grpo_trainer.py)、[rollout 与权重同步](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/rollout_mixin.py)。
- [vLLM 0.19.1 Qwen3-VL 多模态处理器](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/qwen3_vl.py)：Qwen3.5 使用同系列视频处理接口，demo 添加原始时间戳桥接并要求实际对齐检查。
- [Ray Actors](https://docs.ray.io/en/latest/ray-core/actors.html)、[GPU 资源分配](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)。
