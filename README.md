# MSLoc：第一阶段与第二阶段实验

本说明按两个阶段组织：第一阶段利用 XCLIP 特征与 DeMamba 识别异常片段并生成 proposal；第二阶段以 proposal 为基础，开展多模态大模型训练与评测。具体步骤见下方目录。

## 阶段目录

- [第一阶段：小模型实验](#第一阶段小模型实验)
- [第二阶段：大模型实验](#第二阶段大模型实验)

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
```

### Qwen3.5-4B 直接评测

输入第一阶段测试 proposal、待检视频和 `_0119` 测试标注。沿用 Trace 的取帧方式：每个 proposal 的前 20%、中间 60%、后 20% 分别取 16、8、16 帧。片段 MP4 旁的 `.timestamps.json` 以毫秒记录每帧相对 proposal 起点的时间；`Qwen/trace_video_template.py` 将它转换成 Qwen3.5 所需的帧索引和 FPS，避免非均匀帧被当成匀速视频。SFT、教师预检、OPSD 和正式评测共用这一设置；已有均匀 40 帧数据和 adapter 需要重新生成与训练。模型先解释，最后对伪造片段给出片段内相对秒数，对真实片段输出 `Real`。程序将区间换算成原视频绝对秒数，再调用 `evaluate_long.py` 计算与 Trace 相同的指标。格式或时间范围错误单独记为 `invalid`；`predictions.json` 保留原文与逐 proposal 的真假判定，`parse_summary.json` 汇总误报、漏报和格式错误。fake 视频中未命中 GT 的 proposal 按片段记为 real；整视频的 `metrics.json` 仍使用原视频 GT。结果在 `../MSLoc_data/Qwen/base_eval/`。

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

先把训练 proposal 中与伪造 GT 相交的片段做成视频训练集。标签只用 proposal 内的 GT 交集；一个 proposal 命中多个 GT 时选交集最长的一处，并在审计文件记录命中数。完整覆盖 GT 时，解释按标注组织为开始、主要异常、结束三句，或仅主要异常一句；只覆盖部分 GT 时仅保留主要异常一句，避免引用片段外的起止画面。输出时不加阶段标签。训练提示词只要求定位异常；正常和未命中 GT 的 proposal 不参加这一步 SFT。数据与视频片段写入 `../MSLoc_data/Qwen/sft_data/`。

```bash
python Qwen/prepare_sft.py \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --prompt-file Qwen/prompts/student_sft.txt \
  --output-dir ../MSLoc_data/Qwen/sft_data \
  --frames 40 \
  --clean
```

用 LoRA 训练 Qwen3.5-4B，关闭 thinking。单卡时只把 `--devices` 改成 `0`；程序会保持全局 batch 为 8。断点继续时用 `--resume auto` 并去掉 `--clean`。权重和训练曲线在 `../MSLoc_data/Qwen/sft/`。

```bash
python Qwen/train_sft.py \
  --model ../Qwen/Qwen3.5-4B \
  --dataset ../MSLoc_data/Qwen/sft_data/train.jsonl \
  --output ../MSLoc_data/Qwen/sft \
  --devices 0,1,2,3,4,5,6,7 \
  --epochs 2 \
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

先在训练 proposal 上用相同的 SFT adapter、视频和学生提示词各做一次完整生成评测。预检教师沿用学生的任务说明、格式要求与示例，只额外收到当前 proposal 的片段内 GT 区间或“此片段 real”的文字真值，以及异常对象与类别。提示词按一或三句解释明确这些类别对应的内容，不提供标注原句。这一步用于验证初始教师，最终测试仍由不看真值的学生完成。两份评测分别写入 `../MSLoc_data/Qwen/opsd_student_precheck/` 和 `../MSLoc_data/Qwen/opsd_teacher_precheck/`。

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

把全部训练 proposal 做成 OPSD 数据，异常片段使用与 SFT 一致的单区间 GT；fake 视频中未命中 GT 的 proposal 按 Trace 记为近邻难负例或误报，教师对这些片段给 real 真值。真实视频的误报也保留。学生消息沿用 `Qwen/prompts/student.txt`；教师在相同视频上额外看到片段真假、异常片段内相对 GT，以及标注的时空伪造类别、`obj`、对象 `bnd_class` 和 `bnd_sub_class`、起止边界各自的 `bnd_class`。提示词说明这些类别对应解释中的哪一句，但不提供标注原句。`teacher_precheck.txt` 用于完整生成评测，`teacher_opsd.txt` 用于学生在线生成后的逐 token 蒸馏；两者分别写真假指令，训练器将学生已经生成的 token 接在教师输入之后。数据、审计记录和片段写入 `../MSLoc_data/Qwen/opsd_data/`。

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
  --clean
```

从 SFT 的 LoRA 继续训练。使用 ms-swift GKD/OPSD：学生在线生成；同一当前 LoRA 权重在教师特权提示词下停止梯度并提供分布监督，学生更新后教师权重也随之更新。初始教师评测未通过时程序会拒绝训练。视频 rollout 使用 Transformers 路径。单卡时只把 `--devices` 改成 `0`；断点继续用 `--resume auto` 并去掉 `--clean`，程序会核对原训练参数和输入文件。权重、训练曲线及续训配置写入 `../MSLoc_data/Qwen/opsd/`。

```bash
python Qwen/train_opsd.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/sft/last \
  --dataset ../MSLoc_data/Qwen/opsd_data/train.jsonl \
  --teacher-gate ../MSLoc_data/Qwen/opsd_teacher_gate.json \
  --output ../MSLoc_data/Qwen/opsd \
  --devices 0,1,2,3,4,5,6,7 \
  --epochs 1 \
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

grpo

grpo评测
