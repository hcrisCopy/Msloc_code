# MSLoc demo：总入口与单卡运行指南

这套 demo 用于短训练验证：加载完整权重，执行参数更新，展示回答、loss、奖励和耗时，并比较 Transformers 与 vLLM rollout。

所有命令在 `Msloc_code/` 根目录执行。新增代码只在 `Qwen/demo/`；复用现有 Qwen 模板、数据规则和奖励函数，不修改它们。

**单卡路线：完整权重推理 → 教师检查 → HF 训练与回载。Ray 单卡是显存允许时的额外流程验证；多卡 DDP 和跨 GPU 并行收益需要多卡机器验证。** 支持 `--devices 0` 不等于任意显存容量都能跑。

<a id="navigation"></a>
## 文档导航：

| 文档 | 回答什么问题 | 什么时候看 |
|---|---|---|
| [README：本页](README.md) | 有什么硬件、先跑什么、命令是什么、结果在哪里？ | **先看；日常操作入口** |
| [训练讲解提纲](训练讲解提纲.md) | SFT / OPSD / GRPO 每一步在算什么，怎么讲？ | 跑之前理解流程，跑之后对照真实结果讲解 |
| [Ray 多角色训练](Ray多角色训练.md) | 角色如何分工、权重如何同步、单卡能验什么、多卡怎么比较速度？ | 单卡基础流程之后按需看；多卡实验的操作手册 |
| [第二阶段 Qwen 面试问题清单](第二阶段Qwen面试问题清单.md) | 面试可能追问什么、原项目代码在哪里？ | 最后自测；这是原项目问题清单，不是新 demo 的执行步骤，引用行号应核对当前代码 |

阅读顺序：**本页选路线 → 讲解提纲理解算法 → 执行所选路线 → 面试清单自测**。选择 Ray 路线时再进入 Ray 文档，不需要把所有文档从头到尾重复读。

本页直达：[单卡能做什么](#single-gpu) · [多卡增加什么](#multi-gpu) · [环境与权重](#setup) · [数据准备](#data) · [单卡推理/教师检查](#preflight) · [单卡 HF 训练](#single-training) · [单卡 vLLM 对比](#single-vllm) · [多卡 Ray 入口](#multi-ray) · [回载与恢复](#reload) · [验收产物](#deliverables)

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

1. [配置环境与权重](#setup)、[准备数据](#data)。收到完整 SFT 权重就直接加载，无需先重训 SFT。
2. [分别跑学生、教师推理和准入](#preflight)，查看 `predictions.html`。OPSD 准入失败先处理教师问题，不绕过检查；GRPO 不依赖这个教师 gate。
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

### 如何直接换权重

| 收到的内容 | 传参 |
|---|---|
| 完整微调模型或已合并模型 | `--model ../MSLoc_data/Qwen/received/sft_full --adapter none` |
| 单独 LoRA | `--model ../Qwen/Qwen3.5-4B --adapter ../MSLoc_data/Qwen/received/sft_lora` |
| 原始底座 | `--model ../Qwen/Qwen3.5-4B --adapter none` |

`sft_full` 是示例目录名，替换成收到的实际目录即可。**加载完整权重不要求继续全参数训练**：默认 `--tuner lora` 会在该完整模型上新建 LoRA；`--tuner full` 则更新语言部分的完整参数。两种模式都延续原项目冻结视觉编码器和视觉对齐层的设置。不要因为收到完整权重就误传成 `--adapter`。

完整模型须为 dense Qwen3.5 的 Hugging Face 格式：`config.json`、`model.safetensors` 或完整分片及索引、`tokenizer.json`、`tokenizer_config.json`、`preprocessor_config.json`、`video_preprocessor_config.json` 和原模型的其他模板文件。LoRA 目录至少包含 `adapter_config.json`、`adapter_model.safetensors`。不要求旧项目的 `sft_config.json`、`teacher_source.json`，也不沿用 checkpoint 中保存的旧绝对路径。

只有 DeepSpeed 分片、Megatron 分片或 `.pt/.pth` 训练状态时，需先由对应训练框架导出完整 HF 模型；这些不是可以直接加载的完整模型。不同架构、MoE、量化权重及视觉 LoRA 不在本 demo 验证范围。加载已有 LoRA 时，`--lora-rank` 与 `--lora-alpha` 必须匹配它的配置。

OPSD 保留固定教师语义。下面 `train.py` 路径推荐独立教师目录：仅在从完整模型新建 LoRA 时允许师生同路径，使用官方禁用 adapter 的固定教师机制；其他情况用独立目录。**Ray 路径的教师始终在独立 Actor 中加载并冻结，因此全参数训练也允许师生指向同一初始完整模型目录。**两条路径都要求 tokenizer/processor 一致。

<a id="data"></a>
## 3. 准备小规模真实数据（单卡、多卡共用）

直接使用已有 `opsd_data/train.jsonl`、`targets.jsonl` 及视频片段，不重新生成 proposal。若尚未准备这些数据，先按根目录 README 的 `prepare_opsd.py` 步骤准备。

```bash
python Qwen/demo/prepare.py \
  --opsd-dataset ../MSLoc_data/Qwen/opsd_data/train.jsonl \
  --opsd-targets ../MSLoc_data/Qwen/opsd_data/targets.jsonl \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --samples 64 --output ../MSLoc_data/Qwen/demo/data --resume none
```

输出 `demo/data/{sft,opsd,grpo,targets}.jsonl` 与 `manifest.json`。seed=42，尽量均衡抽取 fake/real；某一类不足时用另一类补足。SFT 仅取其中 fake。**这是便于检查的训练子集，不是论文测试集，也不保持正式数据的类别比例。**

旧 JSONL 中的视频路径若来自其他机器，追加 `--path-map 旧片段目录=新片段目录`，可多次传入。视频及相邻 `.timestamps.json` 都须存在。保留原先 40 帧、16/8/16 非均匀采样及真实相对时间；不将重新编码的视频帧率当作原始时间。

<a id="preflight"></a>
## 4. 单卡：先验证权重和教师

下面假设已收到完整 SFT 学生 `received/sft_full` 和独立固定教师 `received/teacher_full`。两者可以是同一初始 SFT 权重的两份导出。

```bash
python Qwen/demo/inspect_model.py \
  --mode student --model ../MSLoc_data/Qwen/received/sft_full --adapter none \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/student_before --resume none

python Qwen/demo/inspect_model.py \
  --mode teacher --model ../MSLoc_data/Qwen/received/teacher_full --adapter none \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/teacher_before --resume none

python Qwen/demo/teacher_gate.py \
  --student-run ../MSLoc_data/Qwen/demo/student_before \
  --teacher-run ../MSLoc_data/Qwen/demo/teacher_before \
  --data ../MSLoc_data/Qwen/demo/data --min-iou 0.5 \
  --output ../MSLoc_data/Qwen/demo/gate.json

python Qwen/demo/view_predictions.py \
  --runs ../MSLoc_data/Qwen/demo/student_before ../MSLoc_data/Qwen/demo/teacher_before \
  --data ../MSLoc_data/Qwen/demo/data --output ../MSLoc_data/Qwen/demo/predictions.html
```

推理使用贪心生成。每张 GPU 独立处理一部分样本，产出 `predictions.jsonl`。HTML 可查看视频、完整回答、GT 与预测时间条；视频链接依赖原片段文件。教师额外看到真假、proposal 相对 GT 和类别提示，学生看不到。

准入要求：教师 fake 平均 IoU 严格改善，fake/real 分类准确率不下降，格式错误不增加。然后只保留教师 fake IoU≥0.5 或正确识别 real 的样本，写入 `gate.jsonl`；仍须含两类且数量足够。失败时保存诊断并停止，不绕过教师变强验证。小样本准入只支持本次 demo，不能替代全量准入。

<a id="single-training"></a>
## 5. 单卡：跑实际 HF 训练

以下命令明确传 `--devices 0`；脚本本身默认仍为八卡，所以不要省略该参数。全局 batch=8、32 次优化器更新、每 8 步保存，最多保留 3 个 checkpoint。单卡时自动设置梯度累积，维持全局 batch 和方法配置；这不保证生成阶段的峰值显存降低到单个回答大小。GRPO 的 batch=8 表示 8 个回答，即 2 个输入各生成 4 次；SFT/OPSD 为 8 个输入。Ray 使用另一套角色分配和保存布局，见 [Ray 文档](Ray多角色训练.md)。

### SFT：可选，已有 SFT 权重时无需重训

```bash
python Qwen/demo/train.py \
  --stage sft --backend transformers --model ../Qwen/Qwen3.5-4B --adapter none \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --tuner lora --lora-rank 16 --lora-alpha 32 \
  --steps 32 --save-steps 8 --global-batch-size 8 --learning-rate 1e-4 \
  --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/sft --resume none
```

这里的短跑用于观察监督训练流程，不能把 32 步新 LoRA 自动视为合格教师。后续命令仍以收到的已训练权重为起点。

### OPSD：Transformers 基线

```bash
python Qwen/demo/train.py \
  --stage opsd --backend transformers \
  --model ../MSLoc_data/Qwen/received/sft_full --adapter none \
  --teacher-model ../MSLoc_data/Qwen/received/teacher_full --teacher-adapter none \
  --teacher-gate ../MSLoc_data/Qwen/demo/gate.json \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --tuner lora --lora-rank 16 --lora-alpha 32 \
  --steps 32 --save-steps 8 --global-batch-size 8 --learning-rate 2e-5 \
  --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/opsd_hf --resume none
```

教师固定、学生自己 rollout，完整词表 JSD，beta=0.5、lmbda=1、sft_alpha=0、temperature=1。没有把 teacher 的完整答案当作新 SFT 标签，也没有启用旧的特殊 token 裁剪插件。

### GRPO：Transformers 基线

```bash
python Qwen/demo/train.py \
  --stage grpo --backend transformers \
  --model ../MSLoc_data/Qwen/received/sft_full --adapter none \
  --data ../MSLoc_data/Qwen/demo/data \
  --nli-model ../MSLoc_data/Qwen/ckpt/nli-deberta-v3-small \
  --devices 0 --tuner lora --lora-rank 16 --lora-alpha 32 \
  --steps 32 --save-steps 8 --global-batch-size 8 --num-generations 4 \
  --learning-rate 1e-6 --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/grpo_hf --resume none
```

这里演示 SFT→GRPO。要演示 OPSD→GRPO，把 `--model/--adapter` 换成 OPSD 产物，并使用新的输出目录。奖励权重为定位 1、格式 0.1、解释 0.3；NLI 冻结在 CPU。组内标准化、token 级概率比、clip=0.2、参考 KL 系数 0.02、num_iterations=1、temperature=0.8、top_p=0.95。参考策略固定在本次 GRPO 开始时的权重；传已有 adapter 时也将它作为初始参考。

每次训练自动生成 `run.json`、`train.jsonl`、`length_check.json`、`events_rank_*.jsonl`、`checkpoint-*`、`status.json` 和 `report/`。报告包含 HTML、loss/奖励/耗时/显存 PNG 和 JSON 摘要。训练前打印模型路径、方法、batch 和完整底层命令。

<a id="single-vllm"></a>
## 6. 单卡可选：vLLM 后端对比

先完成 HF 训练再按需进行本节。这里测的是同一张 GPU 上更换生成后端的效果，不能据此声称多卡并行加速。训练器与 vLLM 共卡，显存要求高于仅运行推理；如果资源不足，本节可以作为待验证项。

非均匀视频时间戳容易在切换后端时改变含义，因此先实际检查两套处理器产生的 token IDs、视频像素张量和时空网格是否逐项一致：

```bash
python Qwen/demo/inspect_model.py \
  --mode alignment --model ../MSLoc_data/Qwen/received/sft_full --adapter none \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --max-length 8192 --max-new-tokens 256 --vllm-memory 0.5 \
  --output ../MSLoc_data/Qwen/demo/alignment --resume none
```

只有 `alignment/alignment.json` 为通过才允许使用 vLLM。这里检查模型处理器，与是否附加 LoRA 无关；加载其他完整模型后须重做。对齐检查包含真实 vLLM 预处理；如果 hook 未触发或数据不一致，会明确失败。**输入对齐不代表两种数值内核生成的 token 必须完全相同。**

复制上面的 OPSD 或 GRPO 训练命令，只修改这些参数，其余保持一致：

```text
--backend vllm
--alignment ../MSLoc_data/Qwen/demo/alignment/alignment.json
--vllm-memory 0.35
--output ../MSLoc_data/Qwen/demo/opsd_vllm
```

GRPO 输出目录相应使用 `grpo_vllm`。两边必须从同一初始权重开始；不能把 HF 训练后的 checkpoint 作为 vLLM 起点，再宣称公平比较。

vLLM 与训练器 colocate，每 GPU 一份生成引擎、TP=1；不开启异步、不卸载参数、不改变 batch。每轮新 rollout 前使用官方接口同步当前学生参数，LoRA 使用合并权重同步路线。会记录同步次数，并检查生成时的 `_last_loaded_step`。`--vllm-memory` 只控制生成引擎预算，不是整个训练进程的总显存上限。空间不足会报错，不自动缩短输入或减小模型。是否比 Transformers 快由实际计时决定，短回答、小 batch 和 CPU NLI 都可能限制收益。

```bash
python Qwen/demo/report.py \
  --runs ../MSLoc_data/Qwen/demo/opsd_hf ../MSLoc_data/Qwen/demo/opsd_vllm \
  --warmup-steps 4 --output ../MSLoc_data/Qwen/demo/opsd_compare

python Qwen/demo/report.py \
  --runs ../MSLoc_data/Qwen/demo/grpo_hf ../MSLoc_data/Qwen/demo/grpo_vllm \
  --warmup-steps 4 --output ../MSLoc_data/Qwen/demo/grpo_compare
```

报告排除前 4 步，汇总最慢 rank 的 step 周期、吞吐和阶段耗时，拒绝方法、权重、数据、batch 或代码不同的对比。也支持比较 1 卡与 8 卡，但那是 GPU 数量变化的收益，应与后端收益分开解释。多次恢复的训练可单独画图，不计算跨运行加速比。

计时包含同步监测开销；阶段计时有嵌套，不能相加。GPU 设备占用是每步采样、包含同卡其他进程，不是连续观测的峰值；PyTorch 峰值也不覆盖所有 vLLM 子进程。参数探针只检查少量可训练参数，不能替代完整权重比较。同步日志证明调用发生，不能单凭它证明生成引擎每个权重都正确。速度结论还应结合正常生成、稳定训练与后续质量评测。

<a id="multi-ray"></a>
## 多卡分支：Ray 八卡训练入口

**下面是八卡命令，不属于你当前单卡的必跑步骤。** 单卡 Ray 命令和能验证的范围，请直接跳到 [Ray 单卡验证](Ray多角色训练.md#single-gpu)；完整八卡实验见 [Ray 多卡路线](Ray多角色训练.md#multi-gpu)。

```bash
python Qwen/demo/ray_launch.py \
  --stage grpo --backend vllm --schedule parallel \
  --model ../MSLoc_data/Qwen/received/sft_full --adapter none \
  --data ../MSLoc_data/Qwen/demo/data \
  --nli-model ../MSLoc_data/Qwen/ckpt/nli-deberta-v3-small \
  --alignment ../MSLoc_data/Qwen/demo/alignment/alignment.json --vllm-memory 0.35 \
  --devices 0,1,2,3,4,5,6,7 --cpus-per-worker 2 --object-store-gb 8 \
  --tuner lora --lora-rank 16 --lora-alpha 32 \
  --steps 32 --save-steps 8 --global-batch-size 8 --num-generations 4 \
  --learning-rate 1e-6 --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/grpo_ray --resume none
```

此命令启动私有单机 Ray runtime 和多个常驻模型 Actor。`--cpus-per-worker` 指每个角色实例的 CPU 预算，八卡 GRPO 默认总共 20 个逻辑 CPU；`/dev/shm` 要容纳指定的对象存储预算。权重由训练端分块传到生成端，并做内容校验、加载和前向概率核对。旧版 `--cpus ... --` 包装入口已移除。

完整的 OPSD 命令、三组公平加速对比、恢复、产物回载与服务器测试见 [Ray 多角色训练](Ray多角色训练.md)。更新 demo 后须使用当前代码重做教师预检／准入和输入对齐；旧报告的源码指纹不会通过检查。

<a id="reload"></a>
## 7. 单卡训练后验证与通用恢复规则

默认 LoRA 产物这样回载；如果本次用了 `--tuner full`，则把 `--model` 改为该完整 checkpoint，并传 `--adapter none`：

```bash
python Qwen/demo/inspect_model.py \
  --mode student --model ../MSLoc_data/Qwen/received/sft_full \
  --adapter ../MSLoc_data/Qwen/demo/opsd_hf/checkpoint-32 \
  --data ../MSLoc_data/Qwen/demo/data --devices 0 \
  --max-length 8192 --max-new-tokens 256 \
  --output ../MSLoc_data/Qwen/demo/student_after --resume none

python Qwen/demo/view_predictions.py \
  --runs ../MSLoc_data/Qwen/demo/student_before ../MSLoc_data/Qwen/demo/student_after \
  --data ../MSLoc_data/Qwen/demo/data --output ../MSLoc_data/Qwen/demo/before_after.html
```

只加载原有权重看回答时，做到第 4 节的 student 推理即可，无需训练或教师。

- **训练中断**：原命令仅改 `--resume auto`，从最后保存的 checkpoint 恢复参数、优化器等状态，未保存的步会重跑。必须保持参数、数据、代码与 GPU 数不变；此入口不是改变总步数的续训接口。
- **推理中断**：原命令加 `--resume auto`，按 sample ID 复用各 rank 已落盘结果；GPU 数不变。
- **数据准备中断**：`--resume auto` 检查完整清单；未完成时重建小型 JSONL，不重编码视频。
- **重跑**：明确使用 `--resume none --clean`。只清理指定 demo 输出子目录，检查其不覆盖输入。默认不覆盖已有运行。
- **教师 gate 重做**：使用新 `--output` 文件名，不自动覆盖旧准入证据。
- **仅检查传参**：训练命令追加 `--plan-only`；检查文件、版本和准入并打印命令，不加载 GPU、不写输出。
- **异常**：路径错误、依赖不符、缺少时间戳、超长输入、非有限指标均应修正后重跑。官方 GRPO 参数使用 `truncation_strategy=delete`，但先逐条验证长度并开启 `strict=true`，不通过静默丢弃来完成演示。

完整权重身份使用配置 SHA256 和权重文件大小/修改时间核对，避免反复读取大文件；LoRA 计算权重 SHA256。前者不是完整模型的密码学内容校验，替换权重后应使用新目录、重新预检。各次预检与训练须使用同一版 demo 代码；建议整套上传后再开始，避免中途改动导致记录不匹配。

<a id="deliverables"></a>
## 8. 按硬件划分验收产物

### 单卡验收产物

1. `predictions.html`：完整权重可加载，学生/教师输入有何不同，教师是否确实更强。
2. `gate.json`：全局改善与逐样本筛选，失败也如实解释。
3. 训练报告与 checkpoint：真实更新次数、loss/奖励变化、参数探针、保存后回载结果。
4. [训练讲解提纲](训练讲解提纲.md)：用一条 proposal 解释每一步计算，明确哪些已经实测，哪些因资源限制尚未实测。

第 3 项以显存足够并实际完成为前提；若只有推理通过，就带推理结果和训练阻塞信息，不将它称为训练成功。CPU 测试、单卡 vLLM 对比和 Ray 单卡同步日志有实际结果时再附上，不要求一次全做。

### 多卡补充验收产物

1. 多 Trainer 的 DDP 更新与 checkpoint 一致性记录。
2. Ray 三组实验的 `opsd_ray_benchmark/comparison/report.html`、`grpo_ray_benchmark/comparison/report.html`：角色是否真正并行、端到端耗时和同步开销。
3. 结合生成长度和回载质量解释速度变化；HF Trainer 的比较报告另行查看，不与 Ray 混算。

此 demo 用于小规模流程验证与耗时对比，不替代完整 benchmark、论文性能评测或正式训练工程。

[返回文档导航](#navigation) · [训练讲解](训练讲解提纲.md) · [Ray 操作手册](Ray多角色训练.md) · [面试自测](第二阶段Qwen面试问题清单.md)

## 实现依据

- 项目现有 `Qwen/common.py`、`opsd_common.py`、`prepare_grpo.py`、`trace_video_template.py`、`grpo_rewards.py`、`grpo_logging.py`。
- [ms-swift 4.5.3 GKD trainer](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/gkd_trainer.py)、[GKD loss](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/gkd_loss.py)、[GRPO trainer](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/grpo_trainer.py)、[rollout 与权重同步](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/rollout_mixin.py)。
- [vLLM 0.19.1 Qwen3-VL 多模态处理器](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/qwen3_vl.py)：Qwen3.5 使用同系列视频处理接口，demo 添加原始时间戳桥接并要求实际对齐检查。
- [Ray Actors](https://docs.ray.io/en/latest/ray-core/actors.html)、[GPU 资源分配](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)。
