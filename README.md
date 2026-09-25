# MSLoc：第一阶段与第二阶段实验

本说明按两个阶段组织：第一阶段利用 XCLIP 特征与 DeMamba 识别异常片段并生成 proposal；第二阶段以 proposal 为基础，开展多模态大模型训练与评测。具体步骤见下方目录。

## 阶段目录

- [第一阶段：小模型实验](#第一阶段小模型实验)
- [第二阶段：大模型实验](#第二阶段大模型实验qwen35版本)

## 第一阶段：小模型实验

【可直接运行】小模型阶段的 DINOv2/DINOv3 神经元探测实验，以及 DINOv2/DINOv3/XCLIP 在 ActivityForensics 上的泛化测试，详见 [补充说明](SupplyREADME.md)。

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

第二阶段代码统一放在 `Qwen/`；以下命令均在本仓库根目录执行。使用后训练版 `Qwen/Qwen3.5-4B`，SFT、OPD/OPSD 和 GRPO 共用独立环境。

本阶段训练命令中的 `--max-steps -1` 表示按 `--epochs` 跑全量。单卡短跑时，把 `--devices` 改为 `0`、`--max-steps` 改为 `10`、`--save-steps` 改为 `5`。OPSD 仍要求所用 SFT 权重通过教师预检。若以后在同一目录跑全量，需要用 `--clean` 重新开始，不能用 `--resume auto` 接续短跑。

本阶段带 `--resume` 的命令首次使用 `--resume none --clean`；中断后保持其余参数不变，改用 `--resume auto` 并去掉 `--clean`。SFT、OPSD 数据准备从上次完成的视频继续，已有片段会复用；GRPO 数据准备和评测复用已完成的记录；训练从最近的 checkpoint 继续。旧版 SFT/OPSD 脚本留下的临时 JSONL 不能直接续建，首次使用新版脚本时用 `--clean` 重建数据文件，已有片段仍会复用。

下文统一使用三个名称，单位均为秒。第一阶段 proposal 和数据集标注都以**原视频起点**为基准：

- **原始标注区间**：数据集给出的伪造区间。
- **proposal 内的标注区间**：原始标注区间与 proposal 的交集，仍以原视频起点为基准。
- **片段内目标区间**：交集的两个端点各减去 proposal 起点，以截取片段的起点为基准；这是模型训练和回答所用的区间。

例如，原始标注区间为 `[100, 120]`，proposal 为 `[110, 130]`，则 proposal 内的标注区间为 `[110, 120]`，片段内目标区间为 `[0, 10]`。没有交集的 proposal 在片段级按 real 处理。

### 环境配置

创建第二阶段环境 `msloc_qwen35`，并安装训练依赖：

```bash
conda create -n msloc_qwen35 python=3.12 -y
conda activate msloc_qwen35
conda install -y -c nvidia/label/cuda-12.8.0 cuda-toolkit=12.8.0
conda install -y -c conda-forge cuda-compat=12.8.1 ffmpeg=7
source Qwen/env.sh
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install --no-build-isolation -r Qwen/requirements.txt
CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install causal-conv1d==1.7.0 --no-build-isolation
FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install flash-attn==2.8.3 --cache-dir ../MSLoc_data/.cache/pip --no-build-isolation
```

`Qwen/requirements.txt` 固定了训练、视频解码和其他加速依赖。`env.sh` 将 Flash Attention 的缓存和安装临时文件放在 `../MSLoc_data/.cache/`。新开终端时都要运行：

```bash
conda activate msloc_qwen35
source Qwen/env.sh
```

### 模型下载

环境配置后、开始任何第二阶段实验前，从本仓库根目录下载一次模型。产物是 `../Qwen/Qwen3.5-4B/`，与后续代码目录 `Qwen/` 分开：

```bash
hf download Qwen/Qwen3.5-4B --local-dir ../Qwen/Qwen3.5-4B
hf download cross-encoder/nli-deberta-v3-small --local-dir ../cross-encoder/nli-deberta-v3-small
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
  --resume none \
  --clean
```

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
  --resume none \
  --clean
```

### Qwen3.5-4B OPSD：教师评测与训练

先在训练 proposal 上用相同的 SFT adapter、视频和学生提示词各做一次完整生成评测。预检教师沿用学生的任务说明、格式要求与示例，只额外收到当前 proposal 的片段内目标区间或“此片段 real”的文字真值，以及异常对象与类别。提示词按一或三句解释明确这些类别对应的内容，不提供标注原句。这一步用于验证初始教师，最终测试仍由不看真值的学生完成。两份评测分别写入 `../MSLoc_data/Qwen/opsd_student_precheck/` 和 `../MSLoc_data/Qwen/opsd_teacher_precheck/`。

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
  --teacher-prompt-file Qwen/prompts/teacher_precheck.txt \
  --output ../MSLoc_data/Qwen/opsd_teacher_precheck \
  --devices 0,1,2,3,4,5,6,7 \
  --frames 40 \
  --max-new-tokens 256 \
  --resume none \
  --clean
```

比较 Trace 的 `Total` 指标：教师 `Loc_F1`、`Loc_IoU` 均提高，`Det_Acc` 和无效格式数不退化，才允许训练。结果写入 `../MSLoc_data/Qwen/opsd_teacher_gate.json`。

```bash
python Qwen/check_teacher.py \
  --student-eval ../MSLoc_data/Qwen/opsd_student_precheck \
  --teacher-eval ../MSLoc_data/Qwen/opsd_teacher_precheck \
  --output ../MSLoc_data/Qwen/opsd_teacher_gate.json \
  --clean
```

把全部训练 proposal 做成 OPSD 数据，异常片段使用与 SFT 一致的单个片段内目标区间；fake 视频中未命中原始标注区间的 proposal 按 Trace 记为近邻难负例或误报，教师对这些片段给 real 真值。真实视频的误报也保留。学生消息沿用 `Qwen/prompts/student.txt`；教师在相同视频上额外看到片段真假、异常片段的片段内目标区间，以及标注的时空伪造类别、`obj`、对象 `bnd_class` 和 `bnd_sub_class`、起止边界各自的 `bnd_class`。提示词说明这些类别对应解释中的哪一句，但不提供标注原句。`teacher_precheck.txt` 用于完整生成评测，`teacher_opsd.txt` 用于学生在线生成后的逐 token 蒸馏；两者分别写真假指令，训练器将学生已经生成的 token 接在教师输入之后。数据、审计记录和片段写入 `../MSLoc_data/Qwen/opsd_data/`。

```bash
python Qwen/prepare_opsd.py \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --student-prompt-file Qwen/prompts/student.txt \
  --teacher-precheck-prompt-file Qwen/prompts/teacher_precheck.txt \
  --teacher-opsd-prompt-file Qwen/prompts/teacher_opsd.txt \
  --output-dir ../MSLoc_data/Qwen/opsd_data \
  --frames 40 \
  --resume none \
  --clean
```

从 SFT 的 LoRA 继续训练。使用 ms-swift GKD/OPSD：学生在线生成；同一当前 LoRA 权重在教师特权提示词下停止梯度并提供分布监督，学生更新后教师权重也随之更新。初始教师评测未通过时程序会拒绝训练。视频 rollout 使用 Transformers 路径。单卡时只把 `--devices` 改成 `0`；续训时程序会核对原训练参数和输入文件。权重、训练曲线及续训配置写入 `../MSLoc_data/Qwen/opsd/`。

```bash
python Qwen/train_opsd.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/sft/last \
  --dataset ../MSLoc_data/Qwen/opsd_data/train.jsonl \
  --teacher-gate ../MSLoc_data/Qwen/opsd_teacher_gate.json \
  --output ../MSLoc_data/Qwen/opsd \
  --devices 0,1,2,3,4,5,6,7 \
  --epochs 1 \
  --max-steps -1 \
  --global-batch-size 8 \
  --learning-rate 2e-5 \
  --max-length 8192 \
  --max-completion-length 256 \
  --save-steps 100 \
  --resume none \
  --clean
```

最后用原学生提示词和测试集评测 OPSD adapter。结果写入 `../MSLoc_data/Qwen/opsd_eval/`，可与 SFT 的 `metrics.json` 直接比较。

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
  --resume none \
  --clean
```

从 OPSD LoRA 继续训练 LoRA。三项奖励沿用 Trace 的权重：定位 1.0、格式 0.1、解释 0.3。定位比较片段内单区间 IoU 和边界误差；真实片段回答 `Real` 得分。解释只对定位 IoU 至少 0.3 的异常回答评分，使用本地冻结 NLI 模型比较生成解释与标注事实。`--use_vllm false` 保证在线采样经过自定义视频时间戳模板。4 次采样组成一个 GRPO 组，关闭 thinking，与正式评测格式一致。单卡只改 `--devices` 为 `0`。权重、采样记录和奖励曲线写入 `../MSLoc_data/Qwen/grpo/`。

```bash
python Qwen/train_grpo.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/opsd/last \
  --dataset ../MSLoc_data/Qwen/grpo_data/train.jsonl \
  --nli-model ../cross-encoder/nli-deberta-v3-small \
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

### Qwen3.5-4B GRPO 评测

使用同一学生提示词和测试 proposal，按 SFT、OPSD 相同的 Trace 定位指标评测 `last` adapter。结果写入 `../MSLoc_data/Qwen/grpo_eval/`。

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
  --resume none \
  --clean
```
