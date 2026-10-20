# 第一阶段补充实验：编码器训练与评测

[返回主 README](README.md#第一阶段小模型实验)。本页包含 DINOv2、DINOv3、UniFormerV2 普通版与神经元版的完整流程，以及 XCLIP 的 ActivityForensics 泛化评测。

所有命令均从项目根目录 `MSLoc_code` 执行。默认使用单机 8 卡；单卡正式测试时，仅将 `--device-ids 0,1,2,3,4,5,6,7` 改为 `--device-ids 0`，其他参数保持不变。

- [1. DINOv2](#1-dinov2)
- [2. DINOv3](#2-dinov3)
- [3. XCLIP 神经元版：ActivityForensics 评测](#3-xclip-神经元版activityforensics-评测)
- [4. XCLIP 普通版：ActivityForensics 评测](#4-xclip-普通版activityforensics-评测)
- [5. UniFormerV2](#5-uniformerv2)

## 1. DINOv2

### 1.1 权重下载

```bash
hf download facebook/dinov2-base \
  --local-dir ../MSLoc_data/DeMamba/pretrained_weights/dinov2_hf
```

### 1.2 不使用神经元探测的基线训练与评测

该基线直接把 DINOv2 最后一层的全部 patch-token 隐藏特征输入 DeMamba，不读取神经元索引。

训练结果输出到 `../MSLoc_data/DeMamba/results/dinov2_baseline_4/`；中断后从最新 epoch 继续。

```bash
python DeMamba/train.py \
  --config DeMamba/configs/DINOv2_Tasle.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42 \
  --resume-from-checkpoint auto
```

读取训练得到的 `best_acc.pth` 评测原论文测试集，结果输出到 `../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval/`；中断后继续，完成后直接复用。

```bash
python DeMamba/eval.py \
  --config DeMamba/configs/DINOv2_Tasle.yaml \
  --model_path ../MSLoc_data/DeMamba/results/dinov2_baseline_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval \
  --device-ids 0,1,2,3,4,5,6,7 \
  --val-batch-size 16 \
  --cache-data \
  --resume \
  --reuse-existing
```

读取上一步的预测并汇总指标，输出到 `../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval/metrics_long.json`；文件已存在时直接复用。

```bash
python evaluate_long.py \
  --gt_file ../MSLoc_data/test_all_1209_0119_long.json \
  --infer_file ../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval/predictions.json \
  --output_file ../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval/metrics_long.json \
  --reuse-existing
```

#### ActivityForensics 泛化评测

读取该基线的 `best_acc.pth` 在新 benchmark 上生成预测，结果输出到 `../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval_activityforensics/`。

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/DINOv2_Tasle.yaml \
  --model-path ../MSLoc_data/DeMamba/results/dinov2_baseline_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval_activityforensics \
  --device-ids 0,1,2,3,4,5,6,7 \
  --batch-size 16 \
  --clean
```

读取上一步的预测评估 proposal 质量，结果输出到 `../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval_activityforensics/proposal_quality/`。

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

### 1.3 使用神经元探测的训练与评测

#### 第一步：构造真假帧对

从训练标注和已抽取视频帧生成神经元探测所需的真假帧对，输出到 `../MSLoc_data/DeMamba/neuron_probe_dinov2/train_pairs.jsonl`；文件已存在时直接复用。

```bash
python DeMamba/build_probe_pairs.py \
  --annotations /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/annos/train_all_1209.json \
  --frame-root /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/video_frames \
  --output ../MSLoc_data/DeMamba/neuron_probe_dinov2/train_pairs.jsonl \
  --fps 8 \
  --strict \
  --reuse-existing
```

#### 第二步：探测 DINOv2 神经元

读取上一步的真假帧对并计算敏感神经元，输出得分和索引到 `../MSLoc_data/DeMamba/neuron_probe_dinov2/`；中断后继续，完成后直接复用。

```bash
python DeMamba/probe_dinov2_neurons.py \
  --pairs ../MSLoc_data/DeMamba/neuron_probe_dinov2/train_pairs.jsonl \
  --frame-root /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/video_frames \
  --dinov2-hf-model-path ../MSLoc_data/DeMamba/pretrained_weights/dinov2_hf \
  --output-dir ../MSLoc_data/DeMamba/neuron_probe_dinov2 \
  --image-batch-size 32 \
  --crop-youku \
  --amp \
  --strict \
  --resume
```

#### 第三步：训练 DINOv2 + DeMamba

读取上一步的神经元索引进行训练，checkpoint 输出到 `../MSLoc_data/DeMamba/results/dinov2_neurons_4/`；中断后从最新 epoch 继续。

```bash
python DeMamba/train.py \
  --config DeMamba/configs/DINOv2_Tasle_neurons.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42 \
  --resume-from-checkpoint auto
```

#### 第四步：评测原论文测试集

读取训练得到的 `best_acc.pth` 生成预测，结果输出到 `../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval/`；中断后继续，完成后直接复用。

```bash
python DeMamba/eval.py \
  --config DeMamba/configs/DINOv2_Tasle_neurons.yaml \
  --model_path ../MSLoc_data/DeMamba/results/dinov2_neurons_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval \
  --device-ids 0,1,2,3,4,5,6,7 \
  --val-batch-size 16 \
  --cache-data \
  --resume \
  --reuse-existing
```

#### 第五步：汇总原论文指标

读取上一步的 `predictions.json` 计算最终指标，输出到 `../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval/metrics_long.json`；文件已存在时直接复用。

```bash
python evaluate_long.py \
  --gt_file ../MSLoc_data/test_all_1209_0119_long.json \
  --infer_file ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval/predictions.json \
  --output_file ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval/metrics_long.json \
  --reuse-existing
```

#### 第六步：评测 ActivityForensics

读取神经元版的 `best_acc.pth` 在新 benchmark 上生成预测，结果输出到 `../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics/`。

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/DINOv2_Tasle_neurons.yaml \
  --model-path ../MSLoc_data/DeMamba/results/dinov2_neurons_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics \
  --device-ids 0,1,2,3,4,5,6,7 \
  --batch-size 16 \
  --clean
```

DINOv2 评测器会按配置中的 `image_size: 196` 和 `normalization: dinov2` 处理原始视频；数据未下载完整时只评测当前可用视频，并标记 `evaluation_scope: partial`。

#### 第七步：评估 proposal 质量

读取上一步的预测评估 proposal 质量，结果输出到 `../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics/proposal_quality/`。

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

## 2. DINOv3

### 2.1 权重下载

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir "../MSLoc_data/DeMamba/pretrained_weights/dinov3_hf"
```

### 2.2 不使用神经元探测的基线训练与评测

该基线直接把 DINOv3 最后一层去除 CLS 和 register token 后的全部 patch-token 隐藏特征输入 DeMamba，不读取神经元索引。

训练结果输出到 `../MSLoc_data/DeMamba/results/dinov3_baseline_4/`；中断后从最新 epoch 继续。

```bash
python DeMamba/train.py \
  --config DeMamba/configs/DINOv3_Tasle.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42 \
  --resume-from-checkpoint auto
```

读取训练得到的 `best_acc.pth` 评测原论文测试集，结果输出到 `../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval/`；中断后继续，完成后直接复用。

```bash
python DeMamba/eval.py \
  --config DeMamba/configs/DINOv3_Tasle.yaml \
  --model_path ../MSLoc_data/DeMamba/results/dinov3_baseline_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval \
  --device-ids 0,1,2,3,4,5,6,7 \
  --val-batch-size 16 \
  --cache-data \
  --resume \
  --reuse-existing
```

读取上一步的预测并汇总指标，输出到 `../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval/metrics_long.json`；文件已存在时直接复用。

```bash
python evaluate_long.py \
  --gt_file ../MSLoc_data/test_all_1209_0119_long.json \
  --infer_file ../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval/predictions.json \
  --output_file ../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval/metrics_long.json \
  --reuse-existing
```

#### ActivityForensics 泛化评测

读取该基线的 `best_acc.pth` 在新 benchmark 上生成预测，结果输出到 `../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval_activityforensics/`。

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/DINOv3_Tasle.yaml \
  --model-path ../MSLoc_data/DeMamba/results/dinov3_baseline_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval_activityforensics \
  --device-ids 0,1,2,3,4,5,6,7 \
  --batch-size 16 \
  --clean
```

读取上一步的预测评估 proposal 质量，结果输出到 `../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval_activityforensics/proposal_quality/`。

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

### 2.3 使用神经元探测的训练与评测

#### 第一步：构造真假帧对

从训练标注和已抽取视频帧生成神经元探测所需的真假帧对，输出到 `../MSLoc_data/DeMamba/neuron_probe_dinov3/train_pairs.jsonl`；文件已存在时直接复用。

```bash
python DeMamba/build_probe_pairs.py \
  --annotations /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/annos/train_all_1209.json \
  --frame-root /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/video_frames \
  --output ../MSLoc_data/DeMamba/neuron_probe_dinov3/train_pairs.jsonl \
  --fps 8 \
  --strict \
  --reuse-existing
```

#### 第二步：探测 DINOv3 神经元

读取上一步的真假帧对并计算敏感神经元，输出得分和索引到 `../MSLoc_data/DeMamba/neuron_probe_dinov3/`；中断后继续，完成后直接复用。

```bash
python DeMamba/probe_dinov3_neurons.py \
  --pairs ../MSLoc_data/DeMamba/neuron_probe_dinov3/train_pairs.jsonl \
  --frame-root /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/video_frames \
  --dinov3-backend huggingface \
  --dinov3-hf-model-path ../MSLoc_data/DeMamba/pretrained_weights/dinov3_hf \
  --output-dir ../MSLoc_data/DeMamba/neuron_probe_dinov3 \
  --image-batch-size 32 \
  --crop-youku \
  --amp \
  --strict \
  --resume
```

#### 第三步：训练 DINOv3 + DeMamba

读取上一步的神经元索引进行训练，checkpoint 输出到 `../MSLoc_data/DeMamba/results/dinov3_neurons_4/`；中断后从最新 epoch 继续。

```bash
python DeMamba/train.py \
  --config DeMamba/configs/DINOv3_Tasle_neurons.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42 \
  --resume-from-checkpoint auto
```

#### 第四步：评测原论文测试集

读取训练得到的 `best_acc.pth` 生成预测，结果输出到 `../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval/`；中断后继续，完成后直接复用。

```bash
python DeMamba/eval.py \
  --config DeMamba/configs/DINOv3_Tasle_neurons.yaml \
  --model_path ../MSLoc_data/DeMamba/results/dinov3_neurons_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval \
  --device-ids 0,1,2,3,4,5,6,7 \
  --val-batch-size 16 \
  --cache-data \
  --resume \
  --reuse-existing
```

#### 第五步：汇总原论文指标

读取上一步的 `predictions.json` 计算最终指标，输出到 `../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval/metrics_long.json`；文件已存在时直接复用。

```bash
python evaluate_long.py \
  --gt_file ../MSLoc_data/test_all_1209_0119_long.json \
  --infer_file ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval/predictions.json \
  --output_file ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval/metrics_long.json \
  --reuse-existing
```

#### 第六步：评测 ActivityForensics

读取神经元版的 `best_acc.pth` 在新 benchmark 上生成预测，结果输出到 `../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval_activityforensics/`。

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/DINOv3_Tasle_neurons.yaml \
  --model-path ../MSLoc_data/DeMamba/results/dinov3_neurons_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval_activityforensics \
  --device-ids 0,1,2,3,4,5,6,7 \
  --batch-size 16 \
  --clean
```

#### 第七步：评估 proposal 质量

读取上一步的预测评估 proposal 质量，结果输出到 `../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval_activityforensics/proposal_quality/`。

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

## 3. XCLIP 神经元版：ActivityForensics 评测

```bash
python DeMamba/eval_activityforensics.py \
  --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml \
  --model-path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/full/method/eval_activityforensics \
  --device-ids 0,1,2,3,4,5,6,7 \
  --batch-size 16 \
  --clean
```

完成上述评测后，运行proposal质量评测：

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/full/method/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/full/method/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/full/method/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

## 4. XCLIP 普通版：ActivityForensics 评测

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/XCLIP_Tasle.yaml \
  --model-path ../MSLoc_data/DeMamba/results/all_Class_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/all_Class_4/eval_activityforensics \
  --device-ids 0,1,2,3,4,5,6,7 \
  --batch-size 16 \
  --clean
```

完成上述评测后，运行proposal质量评测：

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/results/all_Class_4/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/results/all_Class_4/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/results/all_Class_4/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

三个评测都会输出 TASLE 风格的 `Det_Acc`、`F1Det`、`F1Loc`，以及
ActivityForensics 的 AP/AR，并分别汇总 all、in-domain、out-of-domain 和各生成器结果。
`predictions.json` 同时保存真实时序标注和模型预测，因此可同时作为proposal质量评测的
`--gt-file` 与 `--infer-file`。

## 5. UniFormerV2

沿用第一阶段环境，使用预训练的 UniFormerV2-B/16，输入 8 帧、224×224，每帧 196 个 patch、每个 patch 768 维。两版均使用 `sft` 微调，数据路径和预处理由各自 YAML 配置指定。指标计算和打印沿用现有评测脚本。

### 5.1 权重下载

```bash
hf download OpenGVLab/InternVideo1.0 --include "uniformerv2/k400/k400_uniformerv2_b16_8x224.pyth" --local-dir ../MSLoc_data/DeMamba/pretrained_weights/InternVideo1.0
```

### 5.2 不使用神经元探测的基线训练与评测

该基线将第 12 层全部 768 个 patch 通道输入 Mamba，不读取神经元索引。全局分支保留原生的第 9～12 层跨帧聚合，并与最后一层的帧 CLS 均值加权融合，得到 768 维视频向量；与 Mamba 输出展平后的局部特征拼接为 151296 维，再做四分类。

训练结果输出到 `../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/`；中断后从最新 epoch 继续。

```bash
python DeMamba/train.py \
  --config DeMamba/configs/UniFormerV2_Tasle.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42 \
  --resume-from-checkpoint auto
```

读取训练得到的 `best_acc.pth` 评测原论文测试集，结果输出到 `../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/eval/`；中断后继续，完成后直接复用。

```bash
python DeMamba/eval.py \
  --config DeMamba/configs/UniFormerV2_Tasle.yaml \
  --model_path ../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/eval \
  --device-ids 0,1,2,3,4,5,6,7 \
  --val-batch-size 16 \
  --cache-data \
  --resume \
  --reuse-existing
```

读取上一步的预测并汇总指标，输出到 `../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/eval/metrics_long.json`；文件已存在时直接复用。

```bash
python evaluate_long.py \
  --gt_file ../MSLoc_data/test_all_1209_0119_long.json \
  --infer_file ../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/eval/predictions.json \
  --output_file ../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/eval/metrics_long.json \
  --reuse-existing
```

#### ActivityForensics 泛化评测

读取该基线的 `best_acc.pth` 在新 benchmark 上生成预测，结果输出到 `../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/eval_activityforensics/`。

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/UniFormerV2_Tasle.yaml \
  --model-path ../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/eval_activityforensics \
  --device-ids 0,1,2,3,4,5,6,7 \
  --batch-size 16 \
  --clean
```

读取上一步的预测评估 proposal 质量，结果输出到 `../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/eval_activityforensics/proposal_quality/`。

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/results/uniformerv2_baseline_4/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

### 5.3 使用神经元探测的训练与评测

从第 1～12 层各选 64 个 patch 通道，拼成 768 维输入 Mamba。神经元选择只作用于局部分支，全局分支保留与基线相同的原生聚合结构，分类器输入维度不变。

#### 第一步：构造真假帧对

从训练标注和已抽取视频帧生成神经元探测所需的真假帧对，输出到 `../MSLoc_data/DeMamba/neuron_probe_uniformerv2/train_pairs.jsonl`；文件已存在时直接复用。

```bash
python DeMamba/build_probe_pairs.py \
  --annotations ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --frame-root ../MSLoc_data/DeMamba/video_frames \
  --output ../MSLoc_data/DeMamba/neuron_probe_uniformerv2/train_pairs.jsonl \
  --fps 8 \
  --strict \
  --reuse-existing
```

#### 第二步：探测 UniFormerV2 神经元

读取上一步的真假帧对，按 8 帧组片段探测，输出 `uniformerv2_neuron_scores.npz` 和 `uniformerv2_neuron_indices.json` 到 `../MSLoc_data/DeMamba/neuron_probe_uniformerv2/`；中断后继续，完成后直接复用。

```bash
python DeMamba/probe_uniformerv2_neurons.py \
  --pairs ../MSLoc_data/DeMamba/neuron_probe_uniformerv2/train_pairs.jsonl \
  --frame-root ../MSLoc_data/DeMamba/video_frames \
  --uniformerv2-weights-path ../MSLoc_data/DeMamba/pretrained_weights/InternVideo1.0/uniformerv2/k400/k400_uniformerv2_b16_8x224.pyth \
  --output-dir ../MSLoc_data/DeMamba/neuron_probe_uniformerv2 \
  --device-ids 0,1,2,3,4,5,6,7 \
  --image-batch-size 32 \
  --max-frame-pairs 0 \
  --final-neuron-count 768 \
  --crop-youku \
  --amp \
  --strict \
  --resume
```

#### 第三步：训练 UniFormerV2 + DeMamba

读取上一步的神经元索引进行训练，checkpoint 输出到 `../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/`；中断后从最新 epoch 继续。

```bash
python DeMamba/train.py \
  --config DeMamba/configs/UniFormerV2_Tasle_neurons.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42 \
  --resume-from-checkpoint auto
```

#### 第四步：评测原论文测试集

读取训练得到的 `best_acc.pth` 生成预测，结果输出到 `../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/eval/`；中断后继续，完成后直接复用。

```bash
python DeMamba/eval.py \
  --config DeMamba/configs/UniFormerV2_Tasle_neurons.yaml \
  --model_path ../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/eval \
  --device-ids 0,1,2,3,4,5,6,7 \
  --val-batch-size 16 \
  --cache-data \
  --resume \
  --reuse-existing
```

#### 第五步：汇总原论文指标

读取上一步的 `predictions.json` 计算最终指标，输出到 `../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/eval/metrics_long.json`；文件已存在时直接复用。

```bash
python evaluate_long.py \
  --gt_file ../MSLoc_data/test_all_1209_0119_long.json \
  --infer_file ../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/eval/predictions.json \
  --output_file ../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/eval/metrics_long.json \
  --reuse-existing
```

#### 第六步：评测 ActivityForensics

读取神经元版的 `best_acc.pth` 在新 benchmark 上生成预测，结果输出到 `../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/eval_activityforensics/`。

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/UniFormerV2_Tasle_neurons.yaml \
  --model-path ../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/eval_activityforensics \
  --device-ids 0,1,2,3,4,5,6,7 \
  --batch-size 16 \
  --clean
```

#### 第七步：评估 proposal 质量

读取上一步的预测评估 proposal 质量，结果输出到 `../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/eval_activityforensics/proposal_quality/`。

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/results/uniformerv2_neurons_4/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```
