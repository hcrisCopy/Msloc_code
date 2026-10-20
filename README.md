# MSLoc：第一阶段与第二阶段实验

本说明按两个阶段组织：第一阶段利用 XCLIP 特征与 DeMamba 识别异常片段并生成 proposal；第二阶段以 proposal 为基础，开展多模态大模型训练与评测。具体步骤见下方目录。

## 阶段目录

- [第一阶段：小模型实验](#第一阶段小模型实验)
- [第二阶段：大模型实验](#第二阶段大模型实验qwen35版本)

## 第一阶段：小模型实验

【可直接运行】小模型阶段的 DINOv2/DINOv3/UniFormerV2 神经元探测实验，以及 DINOv2/DINOv3/UniFormerV2/XCLIP 在 ActivityForensics 上的泛化测试，详见 [补充说明](SupplyREADME.md)。

### 0. 目录结构

所有命令均在服务器的 `MSLoc_code` 目录执行；命令中的所有路径均为相对路径。数据、预训练模型、缓存、中间结果、模型权重、评测与可视化结果均写入同级目录 `../MSLoc_data`。

### 1. 环境安装与检查

```bash
conda create -n msloc python=3.10 -y
conda activate msloc
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r DeMamba/requirements.txt
```

安装 FFmpeg：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ffmpeg -version
```

确认本地 XCLIP 可以离线加载：

```bash
python -c "from transformers import XCLIPVisionModel; m=XCLIPVisionModel.from_pretrained('../MSLoc_data/DeMamba/pretrained_weights/xclip-base-patch16', local_files_only=True); print('offline XCLIP loaded:', m.config.hidden_size, m.config.num_hidden_layers)"
```

预期输出包含 `offline XCLIP loaded: 768 12`。


### 2. 下载数据集和模型权重

```bash
cd ..
modelscope login --token ms-412c41b7-1f64-483e-9ab2-f81cc7c04525
modelscope download --dataset L67plus/TASLE --local-dir ./
cat MSLoc_assets.tar.gz.part-* > MSLoc_data.tar.gz
tar -xzf MSLoc_data.tar.gz
```

解压后目录如下，**注意文件夹名称需要改成 `MSLoc_data`**

```text
MSLoc_data/
├── data
├── DeMamba
└── Trace
```

### 3. 抽帧

```bash
python DeMamba/Preprocess/video2frame.py \
  --input_root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --output_root ../MSLoc_data/DeMamba/video_frames \
  --num_workers 8
```

### 4. 生成全量配置文件

```bash
mkdir -p ../MSLoc_data/DeMamba/full/configs

cat > ../MSLoc_data/DeMamba/full/configs/xclip_baseline_full.yaml <<'YAML'
model: 'XCLIP_DeMamba_4'
tuning_mode: 'sft'
task: 'many2many'

save_dir: '../MSLoc_data/DeMamba/full/baseline/results'
xclip_model_path: '../MSLoc_data/DeMamba/pretrained_weights/xclip-base-patch16'

max_epoch: 10
bath_per_epoch: 1000
train_batch_size: 2
val_batch_size: 2
num_workers: 2
lr: 0.000001

train_json_path: '../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json'
test_json_path: '../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209.json'
dataset_base_path: '../MSLoc_data/DeMamba/video_frames'
window_length: 2.0
frames_per_window: 8
mode: 'four_class'
transform_config: {
  crop_youku: True
}
YAML

cat > ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml <<'YAML'
model: 'XCLIP_NeuronDeMamba_4'
tuning_mode: 'sft'
task: 'many2many'

save_dir: '../MSLoc_data/DeMamba/full/method/results'
neuron_indices_path: '../MSLoc_data/DeMamba/full/method/neuron_probe/xclip_neuron_indices.json'
xclip_model_path: '../MSLoc_data/DeMamba/pretrained_weights/xclip-base-patch16'

max_epoch: 10
bath_per_epoch: 1000
train_batch_size: 32
val_batch_size: 32
num_workers: 32
lr: 0.000001

train_json_path: '../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json'
test_json_path: '../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json'
dataset_base_path: '../MSLoc_data/DeMamba/video_frames'
window_length: 2.0
frames_per_window: 8
mode: 'four_class'
transform_config: {
  crop_youku: True
}
YAML
```

### 5. 构造全量神经元探测对

```bash
mkdir -p ../MSLoc_data/DeMamba/full/method/neuron_probe

python DeMamba/build_probe_pairs.py \
  --annotations ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --frame-root ../MSLoc_data/DeMamba/video_frames \
  --output ../MSLoc_data/DeMamba/full/method/neuron_probe/train_pairs_full.jsonl \
  --fps 8 \
  --strict
```

### 6. 探测并保存最终 768 个神经元

```bash
python DeMamba/probe_xclip_neurons.py \
  --pairs ../MSLoc_data/DeMamba/full/method/neuron_probe/train_pairs_full.jsonl \
  --frame-root ../MSLoc_data/DeMamba/video_frames \
  --model-path ../MSLoc_data/DeMamba/pretrained_weights/xclip-base-patch16 \
  --output-dir ../MSLoc_data/DeMamba/full/method/neuron_probe \
  --final-neuron-count 768 \
  --crop-youku \
  --amp \
  --strict
```

### 7. 训练与评测

训练使用 PyTorch `DataParallel`。单卡使用 `--device-ids 0`，单机 8 卡使用 `--device-ids 0,1,2,3,4,5,6,7`；不要使用 `torchrun`。


#### 7.1 Baseline：全维冻结 XCLIP 特征 + Mamba + 分类头

训练：

```bash
python DeMamba/train.py \
  --config ../MSLoc_data/DeMamba/full/configs/xclip_baseline_full.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42
```

评测：

```bash
python DeMamba/eval.py \
  --config ../MSLoc_data/DeMamba/full/configs/xclip_baseline_full.yaml \
  --model_path ../MSLoc_data/DeMamba/full/baseline/results/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/full/baseline/eval \
  --device-ids 0 \
  --val-batch-size 16
```

#### 7.2 方法：768 个探测神经元 + Mamba + 分类头

训练：

```bash
python DeMamba/train.py \
  --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42
```

评测：

```bash
python DeMamba/eval.py \
  --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml \
  --model_path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/full/method/eval \
  --device-ids 0 \
  --val-batch-size 16
```

```bash
python evaluate_long.py \
  --gt_file "../MSLoc_data/test_all_1209_0119_long.json" \
  --infer_file "../MSLoc_data/DeMamba/full/method/eval/predictions.json"
```

## 第二阶段：大模型实验（Qwen3.5版本）

【可直接运行】Trace旧版本：大模型阶段以小模型阶段生成的proposal为基础，依次开展 SFT、OPD 与 GRPO 训练；运行说明请参阅 [运行指令](Trace/README_RUN_OPD_GRPO.md)。

【调研文档】八卡运行时使用 Ray、vLLM 加速 OPSD 与 GRPO 的可行路线、工程示例和视频输入对齐风险，见 [OPSD 与 GRPO 八卡加速参考](Qwen/OPSD与GRPO八卡加速参考.md)。

第二阶段代码统一放在 `Qwen/`；以下命令均在本仓库根目录执行。使用后训练版 `Qwen/Qwen3.5-4B`，SFT、OPD/OPSD 和 GRPO 共用独立环境。

本阶段训练命令中的 `--max-steps -1` 表示按 `--epochs` 跑全量。单卡短跑可把 `--devices` 改为 `0`、`--max-steps` 改为 `10`、`--save-steps` 改为 `5`。OPSD 还可用 `--max-samples 256` 固定抽取 fake/real proposal；全量使用 `--max-samples -1`。短跑与全量使用不同输出目录，不能互相续训。

本阶段所有 Qwen 完整生成评测（原模型、SFT、教师预检、OPSD、GRPO）统一使用贪心解码（`temperature=0`）和 256 个生成 token。SFT 全量测试发现采样解码使无效回答和定位误差增加，因此采样仅保留为可选实验。切换解码方式时使用新输出目录，不能用 `--resume auto` 接续另一种解码方式的结果。

本阶段带 `--resume` 的命令首次使用 `--resume none --clean`；中断后保持其余参数不变，改用 `--resume auto` 并去掉 `--clean`。SFT、OPSD 数据准备从上次完成的视频继续，已有片段会复用；GRPO 数据准备和评测复用已完成的记录；训练从最近的 checkpoint 继续。旧版 SFT/OPSD 脚本留下的临时 JSONL 不能直接续建，首次使用新版脚本时用 `--clean` 重建数据文件，已有片段仍会复用。

下文统一使用三个名称，单位均为秒。第一阶段 proposal 和数据集标注都以**原视频起点**为基准：

- **原始标注区间**：数据集给出的伪造区间。
- **proposal 内的标注区间**：原始标注区间与 proposal 的交集，仍以原视频起点为基准。
- **片段内目标区间**：交集的两个端点各减去 proposal 起点，以截取片段的起点为基准；这是模型训练和回答所用的区间。

例如，原始标注区间为 `[100, 120]`，proposal 为 `[110, 130]`，则 proposal 内的标注区间为 `[110, 120]`，片段内目标区间为 `[0, 10]`。没有交集的 proposal 在片段级按 real 处理。

### 环境配置

Linux + NVIDIA GPU，命令在Msloc_code/下执行；第二阶段使用独立环境。视频读写使用 Decord/PyAV，无需单独安装 FFmpeg。

**1. 创建环境并安装 PyTorch**

```bash
conda create -n msloc_qwen35 python=3.12 -y
conda activate msloc_qwen35
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
```

**2. 检查编译工具**

```bash
nvcc --version
g++ --version
ninja --version
```

`nvcc` 显示 `release 12.8` 时跳过下面的安装；找不到 `nvcc` 或版本不同，则执行：

```bash
conda install -y -c nvidia/label/cuda-12.8.0 cuda-toolkit=12.8.0
nvcc --version
```

确认 PyTorch 能找到工具包；输出路径必须有效，不能是 `None`：

```bash
python -c 'from pathlib import Path; from torch.utils.cpp_extension import CUDA_HOME; print("CUDA_HOME:", CUDA_HOME); assert CUDA_HOME and (Path(CUDA_HOME) / "bin/nvcc").is_file(), "未找到 CUDA 编译工具包"'
```

**3. 安装项目依赖和 CUDA 扩展**

```bash
python -m pip install --only-binary=av,decord --no-build-isolation -r Qwen/requirements.txt
python -m pip install causal-conv1d==1.7.0 --no-build-isolation
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

正常使用上面的 pip 命令，扩展会先尝试下载 GitHub 预编译包。若预编译包下载失败且安装退出，执行对应命令，直接源码编译：

```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install causal-conv1d==1.7.0 --no-build-isolation
FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install flash-attn==2.8.3 --no-build-isolation
```

命令依据对应版本的官方安装实现：[flash-attn](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/setup.py)、[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/setup.py)。

使用 Ceph 缓存时，将上面的 Flash Attention 安装命令替换为：

```bash
export FLASH_ATTENTION_CACHE_PATH=/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/root/.cache/flash_attn
pip install flash-attn==2.8.3 \
  --cache-dir /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/root/.cache/pip \
  --no-build-isolation
```

**4. 检查安装**

```bash
python -m pip check
python Qwen/check_env.py
```

检查依赖、GPU、两个 CUDA 扩展的前向/反向计算及视频读写；检查视频输出到 `../MSLoc_data/Qwen/env_check/`。通过后再按下文运行训练和评测。若出现 PTX/驱动版本错误，需要处理宿主机驱动，安装工具包不会升级驱动。

新终端只需 `conda activate msloc_qwen35`。

### 模型下载

环境配置后、开始第二阶段实验前，从本仓库根目录下载模型。Qwen3.5-4B 保存在 `../Qwen/Qwen3.5-4B/`，GRPO 使用的 NLI 模型保存在 `../MSLoc_data/Qwen/ckpt/nli-deberta-v3-small/`：

```bash
hf download Qwen/Qwen3.5-4B --local-dir ../Qwen/Qwen3.5-4B
hf download cross-encoder/nli-deberta-v3-small --local-dir ../MSLoc_data/Qwen/ckpt/nli-deberta-v3-small
```

第二个模型是 GRPO 解释奖励使用的冻结 NLI 判别器；GRPO 开始前下载即可。

### Qwen3.5-4B 直接评测

输入第一阶段测试 proposal、待检视频和 `_0119` 测试标注。沿用 Trace 的取帧方式：每个 proposal 的前 20%、中间 60%、后 20% 分别取 16、8、16 帧。SFT、教师预检、OPSD 和正式评测共用这一设置。

片段 MP4 旁的 `.timestamps.json` 以毫秒记录每帧相对 proposal 起点的时间；`Qwen/trace_video_template.py` 将它转换成 Qwen3.5 所需的帧索引和 FPS，避免非均匀帧被当成匀速视频。

模型先解释，最后对伪造片段给出片段内目标区间，对真实片段输出 `Real`。程序给预测区间的两个端点加上 proposal 起点，换回原视频时间，再调用 `evaluate_long.py` 计算与 Trace 相同的指标。

逐 proposal 统计时，只看它是否与原始标注区间相交：相交为 fake，不相交为 real。因此，即使原视频是 fake，未碰到原始标注区间的 proposal 也按 real 片段统计。回答格式错误或区间越界记为 `invalid`，不算模型回答了 `Real`，也不会生成预测的伪造区间。

在 `metrics.json` 的整视频指标中，如果一条视频最终没有任何有效的伪造预测，它会被算成预测 real。当原视频是 fake 时，这种情况就会降低`Det_Acc`；没有预测区间也会影响 `Loc_F1` 和 `Loc_IoU`。原视频是 real 时，`Det_Acc` 算对，定位指标不会因此产生误报。

结果保存在 `../MSLoc_data/Qwen/base_eval/`：`predictions.json` 保存模型原文和每个 proposal 的判定；`parse_summary.json` 汇总逐 proposal 的误报、漏报及无效回答；`metrics.json` 使用原始标注区间计算整视频指标。

```bash
python Qwen/evaluate.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter none \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --prompt-file Qwen/prompts/student.txt \
  --output ../MSLoc_data/Qwen/base_eval \
  --devices 0,1,2,3,4,5,6,7 \
  --frames 40 \
  --max-new-tokens 256 \
  --max-proposals -1 \
  --temperature 0 \
  --resume none \
  --clean
```

### Qwen3.5-4B SFT

先把与原始标注区间相交的训练 proposal 做成视频训练集。每个样本先取 proposal 内的标注区间，再换成片段内目标区间作为模型的定位答案；一个 proposal 命中多个原始标注区间时选交集最长的一处，并在审计文件记录命中数。proposal 完整覆盖原始标注区间时，解释按标注组织为开始、主要异常、结束三句，或仅主要异常一句；只覆盖部分原始标注区间时仅保留主要异常一句，避免引用片段外的起止画面。三句话直接写在 `Explanation:` 后，用句号分隔。训练提示词只要求定位异常；正常和未命中原始标注区间的 proposal 不参加这一步 SFT。数据与视频片段写入 `../MSLoc_data/Qwen/sft_data/`。

```bash
python Qwen/prepare_sft.py \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --prompt-file Qwen/prompts/student_sft.txt \
  --output-dir ../MSLoc_data/Qwen/sft_data \
  --frames 40 \
  --workers 4 \
  --video-threads 1 \
  --resume none \
  --clean
```

`--workers 4` 同时处理 4 个视频，`--video-threads 1` 限制每个视频的编解码线程数；串行处理设 `--workers 1`。续建时改为 `--resume auto` 并去掉 `--clean`，可以调整并发数。

这一步打印信息：`SFT samples` 是与原始标注区间相交、写入训练集的 proposal 数；`negative_excluded` 是没有交集、未用于 SFT 的 proposal 数；`multiple_gt` 是与多个原始标注区间相交的 proposal 数，已包含在 `SFT samples` 中，每条只取交集最长的一处。`train.jsonl` 是训练数据；`targets.jsonl` 同时记录所选的 proposal 内的标注区间和片段内目标区间；`clips/` 存放 40 帧片段。

训练读取 `train.jsonl`：每条样本输入一个按 16/8/16 取出的 40 帧 proposal 片段、`student_sft.txt` 提示词和片段时长；目标回答是先写一或三句 `Explanation`，再写片段内目标区间 `Interval: [start, end]`。只训练语言侧的 LoRA，冻结视觉编码器和对齐层，关闭 thinking。模型输入看不到标注区间。

终端每步显示的 `loss` 是模型预测目标回答中下一个 token 的平均交叉熵，解释和区间都参与计算，视频与提示词不计入；每步使用 8 条样本。权重和 loss 曲线保存在 `../MSLoc_data/Qwen/sft/`。

```bash
python Qwen/train_sft.py \
  --model ../Qwen/Qwen3.5-4B \
  --dataset ../MSLoc_data/Qwen/sft_data/train.jsonl \
  --output ../MSLoc_data/Qwen/sft \
  --devices 0,1,2,3,4,5,6,7 \
  --epochs 2 \
  --max-steps -1 \
  --global-batch-size 8 \
  --learning-rate 1e-4 \
  --max-length 8192 \
  --save-steps 100 \
  --resume none \
  --clean
```

### Qwen3.5-4B SFT 评测

用相同测试 proposal、提示词与指标评测 SFT 的 `last` adapter。结果写入 `../MSLoc_data/Qwen/sft_eval/`。

```bash
python Qwen/evaluate.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/sft/last \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --prompt-file Qwen/prompts/student.txt \
  --output ../MSLoc_data/Qwen/sft_eval \
  --devices 0,1,2,3,4,5,6,7 \
  --frames 40 \
  --max-new-tokens 256 \
  --max-proposals -1 \
  --temperature 0 \
  --resume none \
  --clean
```

### Qwen3.5-4B OPSD：教师评测，学生训练与评测

OPSD 开始前，先用同一份 SFT LoRA 对训练 proposal 做两次完整生成：学生只看 40 帧片段和 `student.txt`；教师看相同片段、相同任务说明，还会通过训练时的 `teacher_opsd.txt` 得知该片段是 fake 还是 real。fake 片段另给教师片段内目标区间、异常对象和类别，指导其生成解释和定位；不提供标注原句。教师这次完整生成既用于和学生比较整体表现，也用于训练前逐条筛选。正式测试仍只评测不看真值的学生。以下全量命令分别输出到 `../MSLoc_data/Qwen/opsd_student_precheck/` 和 `../MSLoc_data/Qwen/opsd_teacher_precheck/`。

单卡调试时，两条命令都使用 `--devices 0 --max-proposals 256`，输出目录分别改为 `../MSLoc_data/Qwen/opsd_student_precheck_debug` 和 `../MSLoc_data/Qwen/opsd_teacher_precheck_debug`。抽样沿用训练 proposal 的 fake/real 比例：当前全量为 5886/1111，256 条抽出 215 fake、41 real。两次预检和后续调试训练使用同一批片段。抽样结果可比较这批片段上的真假判定、fake 片段的定位 IoU 和无效回答数；它不能代替全量预检，也不能据此计算整视频指标。

```bash
python Qwen/evaluate.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/sft/last \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --prompt-file Qwen/prompts/student.txt \
  --output ../MSLoc_data/Qwen/opsd_student_precheck \
  --devices 0,1,2,3,4,5,6,7 \
  --frames 40 \
  --max-new-tokens 256 \
  --max-proposals -1 \
  --temperature 0 \
  --resume none \
  --clean
```

```bash
python Qwen/evaluate.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/sft/last \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --prompt-file Qwen/prompts/student.txt \
  --teacher-prompt-file Qwen/prompts/teacher_opsd.txt \
  --output ../MSLoc_data/Qwen/opsd_teacher_precheck \
  --devices 0,1,2,3,4,5,6,7 \
  --frames 40 \
  --max-new-tokens 256 \
  --max-proposals -1 \
  --temperature 0 \
  --resume none \
  --clean
```

正式训练前，还要完成教师和学生的回答的全量比较：教师的整视频 `Loc_F1`、`Loc_IoU` 必须更高，`Det_Acc` 不降低；fake、real 片段的判断正确率和无效回答数也不能变差。检查结果写入 `../MSLoc_data/Qwen/opsd_teacher_gate.json`，逐条对比另存为 `../MSLoc_data/Qwen/opsd_teacher_gate_samples.jsonl`。下面是全量命令。

抽样调试时，将两个评测目录和 gate 输出文件改为对应的 `_debug` 名称。抽样调试时，分别看 fake、real 片段的判断正确率，以及 fake 片段的平均定位 IoU 和无效回答数。若教师的平均 IoU 更高，其余三项不变差，就可以继续抽样训练。这只说明教师在抽中的片段上表现更好。

```bash
python Qwen/check_teacher.py \
  --student-eval ../MSLoc_data/Qwen/opsd_student_precheck \
  --teacher-eval ../MSLoc_data/Qwen/opsd_teacher_precheck \
  --output ../MSLoc_data/Qwen/opsd_teacher_gate.json \
  --clean
```

准备 OPSD 数据时，先记录第一阶段生成的全部训练 proposal。与原始伪造标注相交的片段标为 fake，目标是交集最长的一处，换算成片段内秒数；没有交集的片段标为 real，包括来自 fake 视频的误报。训练启动前再用训练教师提示词的逐条回答筛选这些片段。

学生收到 40 帧片段、片段时长和 `Qwen/prompts/student.txt`。SFT 使用的 `student_sft.txt` 只要求输出异常解释和 `Interval: [start, end]`；这里的提示词还允许判断片段为 real：先用一或三句描述正常画面，第二行输出 `Real`。fake 片段仍先解释，再输出区间。

教师看到与学生相同的片段和任务提示，另外得知片段真假。对于 fake 片段，教师还得到目标区间和解释线索：时空伪造类别、对象 `obj` 及其 `bnd_class`/`bnd_sub_class`、起止边界各自的 `bnd_class`。教师提示词说明这些线索应写入解释的哪一句，但不提供标注原句。

预检和训练都使用 `teacher_opsd.txt`。提示词明确区分两种情况：尚无回答时，教师从头生成完整的两行答案；已有学生前缀时，教师只考虑怎样续写。训练时教师在学生已写出的每个前缀上预测下一个 token，不另写一篇完整答案。例子说明前缀已有错误时怎样尽量把剩余输出引向真实标签和目标区间。训练数据、目标记录和视频片段保存在 `../MSLoc_data/Qwen/opsd_data/`。

```bash
python Qwen/prepare_opsd.py \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --student-prompt-file Qwen/prompts/student.txt \
  --teacher-opsd-prompt-file Qwen/prompts/teacher_opsd.txt \
  --output-dir ../MSLoc_data/Qwen/opsd_data \
  --frames 40 \
  --workers 4 \
  --video-threads 1 \
  --resume none \
  --clean
```

并发参数与 SFT 相同：同时处理 4 个视频，每个视频使用 1 个编解码线程。

先将通过预检的 SFT LoRA 合并成独立模型，作为固定教师，保存在 `../MSLoc_data/Qwen/sft_teacher/`；重复执行会复用完整产物。教师和学生从同一份 SFT 权重起步；训练中只更新学生 LoRA，教师始终保持初始权重。同一版提示词生成的数据和评测结果可以复用。

```bash
python Qwen/merge_sft_teacher.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/sft/last \
  --output ../MSLoc_data/Qwen/sft_teacher \
  --devices 0
```

训练脚本复用教师预检时用 `teacher_opsd.txt` 生成的完整回答，逐条筛选 proposal：real 片段必须输出 `Real`；fake 片段必须输出有效区间，且与片段内目标区间的 IoU 至少为 `0.5`。格式错误、真假判错和 fake 定位不足的片段会被排除。筛选结果写入 `../MSLoc_data/Qwen/opsd/teacher_filter.jsonl` 和 `teacher_filter_summary.json`，实际训练数据写入同目录的 `train_teacher_filtered.jsonl`。

每一步先让学生在不看真值的情况下生成回答；固定教师再根据同一视频、特权提示词和学生已写出的前缀，给下一个 token 的概率。训练使用 JSD（`--beta 0.5`），比较师生在完整词表上的下一 token 分布。教师预检未通过，或固定教师不是由预检时的 SFT 权重合并而成，程序会拒绝训练。

下面是全量训练命令，先检查全部训练 proposal，再使用通过教师逐条筛选的 fake/real 片段。权重和训练曲线保存在 `../MSLoc_data/Qwen/opsd/`。单卡调试时，将 `--teacher-gate` 改为对应的 `_debug` 路径，并设置 `--devices 0 --max-samples 256 --max-steps 32 --save-steps 8`；256 指筛选前的片段数。筛选后的数据不能续接旧 OPSD checkpoint；重跑时使用下方的 `--resume none --clean`。

```bash
python Qwen/train_opsd.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/sft/last \
  --teacher-model ../MSLoc_data/Qwen/sft_teacher \
  --dataset ../MSLoc_data/Qwen/opsd_data/train.jsonl \
  --teacher-gate ../MSLoc_data/Qwen/opsd_teacher_gate.json \
  --output ../MSLoc_data/Qwen/opsd \
  --devices 0,1,2,3,4,5,6,7 \
  --epochs 1 \
  --max-steps -1 \
  --max-samples -1 \
  --global-batch-size 8 \
  --learning-rate 2e-5 \
  --beta 0.5 \
  --teacher-min-iou 0.5 \
  --max-length 8192 \
  --max-completion-length 256 \
  --save-steps 100 \
  --resume none \
  --clean
```

OPSD 训练结束后，用学生提示词和第一阶段的测试 proposal 评测新 LoRA。结果保存在 `../MSLoc_data/Qwen/opsd_eval/`；其 `metrics.json` 可与 SFT 评测结果比较。

```bash
python Qwen/evaluate.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/opsd/last \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --prompt-file Qwen/prompts/student.txt \
  --output ../MSLoc_data/Qwen/opsd_eval \
  --devices 0,1,2,3,4,5,6,7 \
  --frames 40 \
  --max-new-tokens 256 \
  --max-proposals -1 \
  --temperature 0 \
  --resume none \
  --clean
```

### Qwen3.5-4B GRPO

复用 OPSD 已制作的 16/8/16 共 40 帧片段，把片段真假、片段内目标区间和标注解释事实写入奖励字段，不放进学生提示词。学生仍使用 `Qwen/prompts/student.txt`，先解释，再输出 `Real` 或单个 `Interval: [start, end]`。数据写入 `../MSLoc_data/Qwen/grpo_data/`；续建时程序会核对输入文件。

```bash
python Qwen/prepare_grpo.py \
  --opsd-dataset ../MSLoc_data/Qwen/opsd_data/train.jsonl \
  --opsd-targets ../MSLoc_data/Qwen/opsd_data/targets.jsonl \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --prompt-file Qwen/prompts/student.txt \
  --output-dir ../MSLoc_data/Qwen/grpo_data \
  --workers 4 \
  --resume none \
  --clean
```

`--workers 4` 并发检查 4 条 proposal 的片段和时间戳文件；GRPO 复用已有视频，无需 `--video-threads`。设为 `1` 则串行，续建使用 `--resume auto` 并去掉 `--clean`。

训练默认从 OPSD LoRA 继续。学生只看 40 帧视频、片段时长和 `student.txt`，对同一个 proposal 生成 4 个回答；真假标签、片段内目标区间和标注解释只用于打分，不进入学生提示词。关闭 thinking，回答先写解释，再写 `Real` 或 `Interval: [start, end]`。

每个回答分别计算三项奖励，再按 **定位 1.0、格式 0.1、解释 0.3** 加权。三类奖励参考 Trace；解释奖励考虑到标注可能不完整，不因额外的未匹配句子直接扣分：

- **定位**：目标为 real 时，输出 `Real` 得 1 分，误报区间得 -1 分，无效回答得 -0.5 分。目标为 fake 时，输出 `Real` 或无效回答得 -1 分；输出区间则按与片段内目标区间的 IoU 和起止边界误差评分。
- **格式**：能解析出规定的解释及 `Real` 或有效区间得 1 分；格式错误、区间越界等得 -1 分。
- **解释**：只给目标为 fake、输出有效区间且定位 IoU 至少 0.3 的回答打分，其余为 0。评分依据是标注的一句或三句描述：一句时取对象异常的 `obj_caption`；三句时按开始、对象异常、结束，分别取两个 `bnd_caption` 和一个 `obj_caption`。生成解释按 `. ! ? ;` 以及 `whereas`、`however` 拆成短句。NLI 以标注描述为前提、生成短句为假设，分别估计“标注是否支持这句话”及“这句话是否与标注矛盾”。每条标注最多匹配一句，每句生成解释也最多匹配一条标注；对象异常的覆盖权重为 2，开始和结束各为 1。最终解释分数为 `标注覆盖率 − 0.50 × 已匹配句子的矛盾程度`。没匹配到标注的额外句子不直接扣分，因为标注未必列尽所有可见现象。

训练时用 Transformers 生成回答（`--use_vllm false`），使模型按实际时间戳读取 16/8/16 非均匀采样的 40 帧。单卡运行将 `--devices` 改为 `0`。

训练权重和 `reward_curve.png` 保存在 `../MSLoc_data/Qwen/grpo/`。曲线显示总奖励、三项奖励各自的均值，以及同一组回答得分完全相同的比例。该目录下的 `completions.jsonl` 记录每条回答、三项得分和 `sample_id`；用 `sample_id` 可在 `../MSLoc_data/Qwen/grpo_data/train.jsonl` 找到对应的 proposal。

```bash
python Qwen/train_grpo.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/opsd/last \
  --dataset ../MSLoc_data/Qwen/grpo_data/train.jsonl \
  --nli-model ../MSLoc_data/Qwen/ckpt/nli-deberta-v3-small \
  --output ../MSLoc_data/Qwen/grpo \
  --devices 0,1,2,3,4,5,6,7 \
  --epochs 1 \
  --max-steps -1 \
  --global-batch-size 8 \
  --num-generations 4 \
  --learning-rate 1e-6 \
  --max-length 8192 \
  --max-completion-length 256 \
  --save-steps 100 \
  --resume none \
  --clean
```

如果要比较 **SFT → GRPO**，先照常运行上面的 `prepare_opsd.py` 和 `prepare_grpo.py`，但跳过教师预检与 `train_opsd.py`。GRPO 仍需要这些准备步骤提供真实、异常片段及奖励真值；学生只看到 `student.txt` 和视频。训练器会检查 SFT 与 GRPO 是否来自同一份训练 proposal 和标注。下面直接从 SFT LoRA 开始，产物单独写入 `../MSLoc_data/Qwen/grpo_from_sft/`：

```bash
python Qwen/train_grpo.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/sft/last \
  --dataset ../MSLoc_data/Qwen/grpo_data/train.jsonl \
  --nli-model ../MSLoc_data/Qwen/ckpt/nli-deberta-v3-small \
  --output ../MSLoc_data/Qwen/grpo_from_sft \
  --devices 0,1,2,3,4,5,6,7 \
  --epochs 1 \
  --max-steps -1 \
  --global-batch-size 8 \
  --num-generations 4 \
  --learning-rate 1e-6 \
  --max-length 8192 \
  --max-completion-length 256 \
  --save-steps 100 \
  --resume none \
  --clean
```

### Qwen3.5-4B GRPO 评测

使用同一学生提示词和测试 proposal，按 SFT、OPSD 相同的 Trace 定位指标评测 `last` adapter。下面先评测从 OPSD 继续训练的权重，结果写入 `../MSLoc_data/Qwen/grpo_eval/`。

```bash
python Qwen/evaluate.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/grpo/last \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --prompt-file Qwen/prompts/student.txt \
  --output ../MSLoc_data/Qwen/grpo_eval \
  --devices 0,1,2,3,4,5,6,7 \
  --frames 40 \
  --max-new-tokens 256 \
  --max-proposals -1 \
  --temperature 0 \
  --resume none \
  --clean
```

从 SFT 直接训练的 GRPO 权重使用相同设置评测，结果单独写入 `../MSLoc_data/Qwen/grpo_from_sft_eval/`：

```bash
python Qwen/evaluate.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/grpo_from_sft/last \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --prompt-file Qwen/prompts/student.txt \
  --output ../MSLoc_data/Qwen/grpo_from_sft_eval \
  --devices 0,1,2,3,4,5,6,7 \
  --frames 40 \
  --max-new-tokens 256 \
  --max-proposals -1 \
  --temperature 0 \
  --resume none \
  --clean
```

## TextRQ：解释文本质量评测

入口为 `Qwen/TextRQ/evaluate_text_rq.py`（旧版： `evaluate_RQ.py/.sh` ）。读取上面 `Qwen/evaluate.py` 生成的 `predictions.json`（含 `proposal_results`）及同一份测试 anno，不读取视频、不重新生成解释。

这里的**预测 proposal** 指模型预测的候选片段，**GT 区间**指人工标注的异常片段。二分类以 **Fake 为正类**：预测 proposal 的裁剪窗口与任一异常 GT 区间有交集为 Fake，否则为 Real。

| 分组 | GT 片段类别 | 预测结论 |
| --- | --- | --- |
| TP | Fake | Fake |
| FP | Real | Fake |
| FN | Fake | Real |
| TN | Real | Real |

格式错误或非法区间单列为 `invalid`，不归入上述四类。

**1. 解析预测，决定哪些解释能评分**

模型输出应为两行：第一行 `Explanation: 解释文本`；第二行是 `Real` 或 `Interval: [start, end]`（表示 Fake）。

```text
每个测试视频
├─ 第一阶段未给出预测 proposal → 流程按 Real 输出，第二阶段不生成解释
│  └─ 记录这类视频的数量；没有解释可评，不进入文本均分
└─ 有预测 proposal → 逐个解析对应的模型原始输出
   ├─ ① Explanation 提取失败
   │  ├─ 例如：标签拼错、解释不在首行、Explanation: 后为空
   │  └─ 无法确定要评的文本 → 整条不评分，计入“Explanation 提取失败数”
   └─ Explanation 提取成功 → 评流畅性；有对应标注时再评另外三维及 RQ
      ├─ 第二行是 Real，完整两行格式正确
      │  └─ 按预测 Real 评“解释—结论一致性”；归入 FN 或 TN
      ├─ 第二行是 Interval: [start, end]，完整两行格式正确
      │  ├─ 区间合法 → 按预测 Fake 评一致性；归入 TP 或 FP
      │  └─ ② 区间数值非法（起止颠倒、结束超出片段等）
      │     └─ 仍能识别预测 Fake，照常评一致性；归入 invalid，IoU 留空
      └─ ③ Real/Interval 输出格式错误
         ├─ 例如：只写 Fake、Interval 缺数字/括号、漏写第二行或夹杂额外行
         └─ 程序不认定 Real/Fake → 不评一致性；归入 invalid，IoU 留空
```

① 是**解释提取失败**，整条无法评分；②③ 是**结论或区间出错**，已提取的解释仍评。`invalid` 表示预测格式或区间无效，不代表文字没有评分。

**报表里的 N 怎么看**

`Scored N` 是**评过多少条解释**；`Valid N` 是**某个评分项有多少条分数**。

例如，全量评测 100 个预测 proposal：10 条 Explanation 提取失败，剩下 90 条完成评分；其中 80 条能识别 Real/Fake，60 条有对应异常标注：

| 报表项 | 数量 | 如何计算 |
|---|---:|---|
| `Input N` | 100 | 输入的预测 proposal 总数 |
| `Scored N` | 90 | 完成评分的解释数，每条只算一次 |
| 流畅性的 `Valid N` | 90 | 流畅性分数之和 ÷ **90** |
| 结论一致性的 `Valid N` | 80 | 一致性分数之和 ÷ **80** |
| 对象、类型、细节各自的 `Valid N` | 各 60 | 各项分数之和分别 ÷ **60** |
| RQ 的 `Valid N`（或 `RQ N`） | 60 | 60 条解释各自的四维平均分之和 ÷ **60** |

不评分、留空的项目都不补 0。数量保存在 `selection_summary.json`：

- `videos_without_proposals`：第一阶段没有给出候选片段的视频数。
- `selection_counts.missing_explanation`：① Explanation 提取失败数。
- `parse_status_counts.range_error`：② 区间数值非法数。
- `parse_status_counts.format_error`：整体格式错误数，包含③，也可能与①重叠；不能与①相加。

**2. 为每条解释选择对应标注**

```text
预测 proposal 的裁剪窗口与几条异常 GT 区间有交集？
├─ 0 条 → 该片段 GT 为 Real（真实视频，或 Fake 视频中的正常片段）
│  └─ 无异常文字标注：评流畅性、可解析结论的一致性，不计算四维 RQ
├─ 1 条 → 该片段 GT 为 Fake，用这一条的标注
└─ 多条 → 该片段 GT 为 Fake，比较最终预测区间与这些 GT 的 IoU
   ├─ 最大值 > 0，且只有一条达到最大值 → 用这一条的标注
   └─ 并列最大、全部为 0、预测区间无效或缺失 → 无法选定，不提供标注

选定 GT 后，组织参考内容
├─ 预测 proposal 完整包含标注事件 + 非 Round4 + 两侧边界标注齐全
│  └─ 三部分：开始 bnd_cot_st → 主体 obj_cot → 结束 bnd_cot_ed
└─ 其他情况 → 仅主体 obj_cot
```

多个预测 proposal 可以共用同一条 GT，不因分类错误或 IoU 低而排除。每部分保留全部条目的主体、分类、子分类和描述；标注中的条目数不等于句子数。

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
   └─ 预测三句 → 按事件阶段、主体和含义对照，不按句子序号机械配对
```

**4. 输出哪些分数**

```text
每条已解析的解释
├─ 四维 RQ
│  ├─ 有对应标注（仅主体或三部分均可，不论预测真假）→ 四项分别评分，再取平均
│  │  ├─ 流畅性：文字是否通顺、连贯
│  │  ├─ 对象准确性：主体是否符合标注
│  │  ├─ 异常类型准确性：异常是否符合标注
│  │  └─ 细节一致性：细节、状态变化是否符合可用标注
│  └─ GT 为 Real，或 GT 为 Fake 但无法选定参考 → 只评流畅性，另三项及 RQ 留空
└─ 解释—结论一致性（单列，不计入 RQ）
   ├─ 能解析 Real/Fake 结论 → 评解释是否支持该结论，不论结论是否符合 GT
   └─ 结论解析失败 → 此项留空
```

每项 1–5 分，附一句中文理由；Real/Fake 标准相同。每项均分 = 该项有效分数之和 ÷ 该项有效数量（`Valid N`）；留空为 `null`，不补 0。RQ 均分只纳入四项齐全的解释。

提示词位于 `Qwen/prompts/`，共用 `text_rq_common.txt`。每条 `Explanation` 按适用条件请求裁判 1–3 次：

- **流畅性**：只传预测的 `Explanation` 文本。
- **解释—结论一致性**：传预测的 `Explanation` 文本，以及从 `Interval` / `Real` 解析出的 Fake / Real 结论；不传 `Interval` 的起止数值。
- **对象、异常类型、细节准确性**：传预测的 `Explanation` 文本和对应 GT 的文字标注，不传预测的 `Interval` / `Real`。

**结果怎么统计**

- **分组**：保留全体、预测真假、GT 片段真假、TP/FP/FN/TN/invalid，以及 `TP 且 IoU > 0.7` 子集。TN 另区分真实视频和 fake 视频的正常背景。`--min-iou` 只改变高 IoU 子集，不过滤全体。
- **均分**：`Overall_proposal` 中每条已评分解释权重相同，不先按视频平均；`Macro_domain` 则先算各域均分，再对有有效分数的域等权平均。每项均报告有效数量。
- **域来源**：有参考时取对应 annotation 的 `tool_domain`，缺少该键时取 `model`；无参考时先取视频级字段，再取视频内唯一标注域。多个域记为内部标记 `mixed`（不是 `tool_mixed`），无域的真实视频记为 `real`；来源及域列表均保留。

**文本解释质量与定位准确性的相关性分析**

分析 `Explanation` 分数越高时，预测 `Interval` 是否也越接近 GT：

- **参与样本**：已完成文本评分、输出合法 `Interval`，且视频有异常 GT 的预测 proposal。IoU 取该 `Interval` 与视频全部 GT 区间的最大值；`Real`、无效输出和无异常 GT 的视频不参与。
- **比较方法**：按 IoU 的 `[0,0.3]`、`(0.3,0.5]`、`(0.5,0.7]`、`(0.7,1]` 四组报告文本均分和有效数量；另用全部有效配对计算各评分项及 RQ 与 IoU 的 Spearman 相关系数。正相关表示分数越高，IoU 往往也越高。全体及各 `tool_domain` 分别统计。
- **1000 次重采样**：若有 N 个视频，每轮随机抽 N 次视频（允许重复），带上所抽视频的全部有效 proposal，用已有分数和 IoU 重算相关系数。重复 1000 轮，用结果的 2.5% 和 97.5% 分位数估计 95% 置信区间，反映统计结果的不确定性；**不重新调用裁判**。seed=42，`--bootstrap-repeats 0` 可关闭此步骤。

缺失评分不参与计算；样本不足或数值无变化等导致无法计算的结果留空。

### 环境与裁判模型

在`MSLoc_code/` 下运行，直接复用第二阶段配置的 `msloc_qwen35` 环境。

按需下载 9B、27B；比较两种裁判时下载两者。

```bash
conda activate msloc_qwen35
hf download Qwen/Qwen3.5-9B --local-dir ../Qwen/Qwen3.5-9B
hf download Qwen/Qwen3.5-27B --local-dir ../Qwen/Qwen3.5-27B
```

两者均使用官方原始权重、纯文本输入、BF16 和 `thinking=false`。9B 用于调试及轻量裁判对照，27B 作为正式裁判候选；正式选型依据人工评分一致性。每张卡须能容纳完整模型及推理缓存，27B 不适用于单张 24GB 显卡的 BF16 推理。

| 裁判 | `--model` | `--output` |
|---|---|---|
| Qwen3.5-9B | `../Qwen/Qwen3.5-9B` | `../MSLoc_data/Qwen/text_rq/sft_v2_9b` |
| Qwen3.5-27B | `../Qwen/Qwen3.5-27B` | `../MSLoc_data/Qwen/text_rq/sft_v2_27b` |

### 运行

先验证 GT 对应、公共提示词隔离、计分、分组统计和续跑逻辑（不加载模型）：

```bash
python -m unittest discover -s Qwen/TextRQ -p "test_text_rq.py" -v
```

下面示例指令使用 9B 评测 SFT 输出；结果写入 `../MSLoc_data/Qwen/text_rq/sft_v2_9b/`。切换 27B 时按上表同时替换 `--model` 和 `--output`，其余参数相同；不同裁判不共用评分缓存。每张卡加载一个完整裁判，独立处理一份样本，卡内由 vLLM 批处理；终端显示一个全局进度条。

```bash
python Qwen/TextRQ/evaluate_text_rq.py \
  --predictions ../MSLoc_data/Qwen/sft_eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --model ../Qwen/Qwen3.5-9B \
  --common-prompt Qwen/prompts/text_rq_common.txt \
  --fluency-prompt Qwen/prompts/text_rq_fluency.txt \
  --consistency-prompt Qwen/prompts/text_rq_consistency.txt \
  --factual-prompt Qwen/prompts/text_rq_factual.txt \
  --output ../MSLoc_data/Qwen/text_rq/sft_v2_9b \
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

单卡全量只改 `--devices 0`。小样本调试改 `--sample-count 32 --output ../MSLoc_data/Qwen/text_rq/sft_v2_9b_debug`（27B 使用 `sft_v2_27b_debug`）；数量不能超过有解释的片段数。查看任务清单加 `--prepare-only`，不加载裁判；开始评分时去掉它，改为 `--resume auto`。评测 OPSD／GRPO 时替换预测文件和输出目录。`--batch-size` 是每卡每批裁判请求数；`--bootstrap-repeats 0` 可关闭相关性置信区间。

`--sample-count -1` 评全量；正数按 GT 片段的 fake/real 比例抽样（seed=42）。两类均存在且抽样数至少为 2 时，两类都会保留。样本在评分前选定，续跑不重抽。

产物保存在 `--output` 目录下：

| 用途 | 查看位置 |
|---|---|
| 看结果 | `reports/default/report.txt`：评分汇总；同目录 `report.csv`：表格；`iou_analysis.json`：评分与 IoU 的相关性 |
| 查单条评分 | `scores.jsonl`：预测文本、GT 参考、IoU、分数及理由 |
| 查运行情况 | `selection_summary.json`：样本数量及解析失败数量；`failures/`、`logs/`：错误记录和运行日志 |

其余为配置、样本清单和续跑文件。**保留整个输出目录**，后续可续跑或重新生成报表，无需重新评分。

### 重新生成报表（无需重新评分）

评分完成后，可用已保存的分数修改分组、IoU 阈值或打印格式，不需要 GPU。

```bash
python Qwen/TextRQ/text_rq_report.py \
  --run ../MSLoc_data/Qwen/text_rq/sft_v2_9b \
  --output ../MSLoc_data/Qwen/text_rq/sft_v2_9b/reports/by_domain \
  --group-by tool_domain \
  --min-iou 0.7
```

报表写入 `--output`：`report.txt`、`report.csv`、`report_summary.json`；同目录重跑会覆盖这三份报表，原始评分不变`。

- **换分组**：修改 `--group-by`；`tool_domain` 按工具域，`outcome` 按 TP/FP/FN/TN/invalid，其余选项见 `--help`。
- **换高 IoU 阈值**：修改 `--min-iou`，例如 `0.7` 表示统计 TP 且 IoU > 0.7 的子集。
- **改打印或统计代码**：分别修改 `Qwen/TextRQ/text_rq_report.py` 或 `Qwen/TextRQ/text_rq_metrics.py`，再运行上面的命令。

修改裁判提示词或 GT 参考内容，需要重新评分。

参考 [vLLM 结构化输出](https://github.com/vllm-project/vllm/blob/v0.19.1/examples/offline_inference/structured_outputs.py)、[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)、[Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) 和 [SciPy Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)。固定 seed 不保证不同 GPU／批次调度下逐字一致。
