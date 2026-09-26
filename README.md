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

本阶段训练命令中的 `--max-steps -1` 表示按 `--epochs` 跑全量。单卡短跑可把 `--devices` 改为 `0`、`--max-steps` 改为 `10`、`--save-steps` 改为 `5`。OPSD 还可用 `--max-samples 256` 固定抽取 fake/real proposal；全量使用 `--max-samples -1`。短跑与全量使用不同输出目录，不能互相续训。

本阶段所有 Qwen 完整生成评测（原模型、SFT、教师预检、OPSD、GRPO）统一使用贪心解码（`temperature=0`）和 256 个生成 token。SFT 全量测试发现采样解码使无效回答和定位误差增加，因此采样仅保留为可选实验。切换解码方式时使用新输出目录，不能用 `--resume auto` 接续另一种解码方式的结果。

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
  --max-proposals -1 \
  --temperature 0 \
  --resume none \
  --clean
```

### Qwen3.5-4B OPSD：教师评测与训练

OPSD 开始前，先用同一份 SFT LoRA 对训练 proposal 做两次完整生成：学生只看 40 帧片段和 `student.txt`；教师看相同片段、相同任务说明，还会通过 `teacher_precheck.txt` 得知该片段是 fake 还是 real。fake 片段另给教师片段内目标区间、异常对象和类别，指导其生成解释和定位；不提供标注原句。比较两次回答，是为了检查这些文字信息是否让初始教师表现优于学生。正式测试仍只评测不看真值的学生。以下全量命令分别输出到 `../MSLoc_data/Qwen/opsd_student_precheck/` 和 `../MSLoc_data/Qwen/opsd_teacher_precheck/`。

单卡调试时，两条命令都使用 `--devices 0 --max-proposals 256`，输出目录分别改为 `../MSLoc_data/Qwen/opsd_student_precheck_debug` 和 `../MSLoc_data/Qwen/opsd_teacher_precheck_debug`。程序固定抽取相同的 128 个 fake 和 128 个 real proposal。抽样结果可直接比较这 256 个片段上的真假判定、fake 片段的定位 IoU 和无效回答数；它只说明教师在这批片段上的表现，不能代替全量预检，也不能据此计算整视频指标。

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
  --teacher-prompt-file Qwen/prompts/teacher_precheck.txt \
  --output ../MSLoc_data/Qwen/opsd_teacher_precheck \
  --devices 0,1,2,3,4,5,6,7 \
  --frames 40 \
  --max-new-tokens 256 \
  --max-proposals -1 \
  --temperature 0 \
  --resume none \
  --clean
```

正式训练前，还要完成教师和学生的回答的全量比较：教师的整视频 `Loc_F1`、`Loc_IoU` 必须更高，`Det_Acc` 不降低；fake、real 片段的判断正确率和无效回答数也不能变差。检查结果写入 `../MSLoc_data/Qwen/opsd_teacher_gate.json`，逐条对比另存为 `../MSLoc_data/Qwen/opsd_teacher_gate_samples.jsonl`。逐条结果仅供分析，训练仍使用全部 proposal。README提供了全量命令

抽样调试时，将两个评测目录和 gate 输出文件改为对应的 `_debug` 名称。抽样调试时，分别看 fake、real 片段的判断正确率，以及 fake 片段的平均定位 IoU 和无效回答数。若教师的平均 IoU 更高，其余三项不变差，就可以继续抽样训练。这只说明教师在抽中的片段上表现更好。

```bash
python Qwen/check_teacher.py \
  --student-eval ../MSLoc_data/Qwen/opsd_student_precheck \
  --teacher-eval ../MSLoc_data/Qwen/opsd_teacher_precheck \
  --output ../MSLoc_data/Qwen/opsd_teacher_gate.json \
  --clean
```

准备 OPSD 数据时，保留第一阶段生成的全部训练 proposal。与原始伪造标注相交的片段标为 fake，目标是交集最长的一处，换算成片段内秒数；没有交集的片段标为 real，包括来自 fake 视频的误报。两类片段都参加 OPSD 训练。

> 先用全部训练 proposal ：预检比较的是教师和学生各自从头生成的完整回答，而 OPSD 训练时教师沿学生已经生成的前缀提供逐 token 监督；教师某次完整回答出错，不等于它在该样本的学生前缀上也无法提供有效监督。因此暂不按预检结果逐条筛选。

学生收到 40 帧片段、片段时长和 `Qwen/prompts/student.txt`。SFT 使用的 `student_sft.txt` 只要求输出异常解释和 `Interval: [start, end]`；这里的提示词还允许判断片段为 real：先用一或三句描述正常画面，第二行输出 `Real`。fake 片段仍先解释，再输出区间。

教师看到与学生相同的片段和任务提示，另外得知片段真假。对于 fake 片段，教师还得到目标区间和解释线索：时空伪造类别、对象 `obj` 及其 `bnd_class`/`bnd_sub_class`、起止边界各自的 `bnd_class`。教师提示词说明这些线索应写入解释的哪一句，但不提供标注原句。

`teacher_precheck.txt` 用于预检：教师从空白回答开始，独立生成解释和最终结论，供程序与学生的完整回答比较。`teacher_opsd.txt` 用于训练：学生先生成回答，教师在学生已写出的每个前缀上预测下一个 token；提示词额外说明怎样在前缀已有错误时，尽量把剩余输出引向真实标签和目标区间。教师在这一步不另写一篇完整答案。训练数据、目标记录和视频片段保存在 `../MSLoc_data/Qwen/opsd_data/`。

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

先将通过预检的 SFT LoRA 合并成独立模型，作为固定教师，保存在 `../MSLoc_data/Qwen/sft_teacher/`；重复执行会复用完整产物。教师和学生从同一份 SFT 权重起步；训练中只更新学生 LoRA，教师始终保持初始权重。已生成的 OPSD 数据和教师预检结果可以复用。

```bash
python Qwen/merge_sft_teacher.py \
  --model ../Qwen/Qwen3.5-4B \
  --adapter ../MSLoc_data/Qwen/sft/last \
  --output ../MSLoc_data/Qwen/sft_teacher \
  --devices 0
```

每一步先让学生在不看真值的情况下生成回答；固定教师再根据同一视频、特权提示词和学生已写出的前缀，给下一个 token 的概率。教师预检未通过，或固定教师不是由预检时的 SFT 权重合并而成，程序会拒绝训练。

下面是全量训练命令，使用全部 fake/real proposal，权重和训练曲线保存在 `../MSLoc_data/Qwen/opsd/`。单卡调试时，将 `--teacher-gate` 改为 `../MSLoc_data/Qwen/opsd_teacher_gate_debug.json`，并设置 `--devices 0 --max-samples 256 --max-steps 10 --save-steps 5 --output ../MSLoc_data/Qwen/opsd_debug`。续训时保持原参数不变；旧版动态教师的 checkpoint 不能接到固定教师实验。

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
  --resume none \
  --clean
```

训练默认从 OPSD LoRA 继续。学生只看 40 帧视频、片段时长和 `student.txt`，对同一个 proposal 生成 4 个回答；真假标签、片段内目标区间和标注解释只用于打分，不进入学生提示词。关闭 thinking，回答先写解释，再写 `Real` 或 `Interval: [start, end]`。

每个回答分别计算三项奖励，再按 **定位 1.0、格式 0.1、解释 0.3** 加权。三类奖励参考 Trace；解释奖励考虑到标注可能不完整，不因额外的未匹配句子直接扣分：

- **定位**：目标为 real 时，输出 `Real` 得 1 分，误报区间得 -1 分，无效回答得 -0.5 分。目标为 fake 时，输出 `Real` 或无效回答得 -1 分；输出区间则按与片段内目标区间的 IoU 和起止边界误差评分。
- **格式**：能解析出规定的解释及 `Real` 或有效区间得 1 分；格式错误、区间越界等得 -1 分。
- **解释**：只给目标为 fake、输出有效区间且定位 IoU 至少 0.3 的回答打分，其余为 0。评分依据是标注的一句或三句描述：一句时取对象异常的 `obj_caption`；三句时按开始、对象异常、结束，分别取两个 `bnd_caption` 和一个 `obj_caption`。生成解释按 `. ! ? ;` 以及 `whereas`、`however` 拆成短句。NLI 以标注描述为前提、生成短句为假设，分别估计“标注是否支持这句话”及“这句话是否与标注矛盾”。每条标注最多匹配一句，每句生成解释也最多匹配一条标注；对象异常的覆盖权重为 2，开始和结束各为 1。最终解释分数为 `标注覆盖率 − 0.50 × 已匹配句子的矛盾程度`。没匹配到标注的额外句子不直接扣分，因为标注未必列尽所有可见现象。

视频采样使用 Transformers 路径（`--use_vllm false`），以保留自定义的 16/8/16 帧时间戳。单卡运行只改 `--devices` 为 `0`。权重和奖励曲线写入 `../MSLoc_data/Qwen/grpo/`；其中 `completions.jsonl` 逐条记录回答、三项得分和 `sample_id`，可据此在 `../MSLoc_data/Qwen/grpo_data/train.jsonl` 找到原 proposal。

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
