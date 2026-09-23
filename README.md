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

qwen3.5评测

qwen3.5 SFT训练

qwen3.5 SFT评测

qwen3.5带特权信息教师评测

opd

opd学生评测

grpo

grpo评测
