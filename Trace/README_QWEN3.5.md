# 第二阶段运行说明（MSLoc-Qwen3.5）

本路线只替换第二阶段的基础 MLLM：`TRACE-uni` 改为 `Qwen/Qwen3.5-9B`。MSLoc-PR 的 40 帧自适应采样、DAM、EAM、三个 anomaly-aware token 和 LAA/CLoss 均保留。原 TRACE SFT、OPD、GRPO 和测试代码没有修改，仍按 [原第二阶段运行说明](README_RUN_OPD_GRPO.md) 执行。

Qwen 代码集中在 `Trace/qwen3_5/`。SFT、后续 OPD 和 GRPO 必须共用这里的模型结构、数据构造和输出解析器，不能在不同阶段更换输出协议。

## 1. 环境配置

Qwen3.5 依赖 Transformers 5.x，不能安装到旧 TRACE 的 Transformers 4.40.1 环境。所有命令均在 `msloc_code` 根目录运行。

```bash
conda create -n qwen35_trace python=3.11 -y
conda activate qwen35_trace
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r Trace/qwen3_5/requirements.txt
```

## 2. 下载 Qwen3.5-9B

使用 Hugging Face 下载，权重保存到代码目录同级的 `../Qwen/Qwen3.5-9B/`。

```bash
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3.5-9B \
  --local-dir ../Qwen/Qwen3.5-9B
```

下载产物：`../Qwen/Qwen3.5-9B/`。训练和测试启用 `local_files_only`，缺少文件时直接报错，不会在运行中联网下载。

## 3. Qwen3.5 SFT 训练

每条 SFT 样本对应第一阶段 DeMamba 给出的一个 proposal：脚本读取 proposal、训练 GT 和原视频，proposal 与 GT 伪造片段有交集时构造正样本，否则构造负样本；随后从 proposal 内按“左边界 16 帧、中间事件 8 帧、右边界 16 帧”采样 40 帧，经 Qwen3.5 官方视觉编码器提取特征，再用 DAM 处理左右边界、EAM 聚合中间事件，并将得到的视觉特征、3 个异常感知 token、任务提示词和 proposal 时长一起送入 Qwen3.5。正样本的 assistant 目标是结构固定的 JSON，其中 `label` 为 `fake`，`type` 表示 `temporal` 或 `spatio-temporal`，`segment` 是 GT 与 proposal 交集相对于 proposal 起点的时间，`explanation` 来自 GT 的边界和主体解释；负样本目标是 `{"label":"real","events":[]}`。训练采用 teacher forcing，只对 assistant 的 JSON token 计算文本交叉熵；同时 3 个异常感知 token 分别预测 GT 的开始边界、伪造对象和结束边界类别，使用 `class_features_bge.pt` 的类别顺序计算 CLoss，总损失为 `text_loss + closs_weight × closs`。正式训练冻结 Qwen 视觉编码器，更新 Qwen 语言骨干、DAM、EAM、异常感知 token 和 CLoss head；与原 TRACE 相比，数据构造、40 帧采样、DAM/EAM、CLoss 和最终 `evaluate_long.py` 指标保持一致，但 TRACE 用专用时间词表/head输出时间，Qwen3.5 改为生成下面的 JSON 文本。

正样本目标示例：

```json
{"label":"fake","events":[{"type":"temporal","segment":[2.1,4.6],"explanation":"..."}]}
```

负样本目标示例：

```json
{"label":"real","events":[]}
```

首次训练：

```bash
torchrun --standalone --nproc_per_node=8 Trace/qwen3_5/train_sft.py \
  --model ../Qwen/Qwen3.5-9B \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --class-features ../MSLoc_data/Trace/class_features_bge.pt \
  --deepspeed Trace/scripts/zero2.json \
  --output ../MSLoc_data/Trace/experiments/qwen3_5/student_sft \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 2 \
  --learning-rate 3e-5 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --max-length 4096 \
  --max-pixels 112896 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --bnd-ratio 0.2 \
  --closs-weight 1.0 \
  --num-workers 4 \
  --max-samples 0 \
  --freeze-backbone false \
  --freeze-vision true \
  --resume none \
  --clean
```

输出产物：`../MSLoc_data/Trace/experiments/qwen3_5/student_sft/`，其中包含最终模型、processor、MSLoc 自定义模块、训练状态和按 epoch 保存的 `checkpoint-*`。

训练中断后，删除 `--clean` 并将 `--resume none` 改为 `--resume auto`。

训练不会截断 GT 答案。如果某条样本的完整对话超过 `--max-length`，脚本会报出具体 `sample_id`，需要先检查该样本或显式增大 `--max-length`，不会用残缺 JSON 继续训练。

## 4. Qwen3.5 SFT 评测

使用 SFT 后的学生模型处理第一阶段输出的测试 proposal，输出伪造片段和解释，并计算与原 TRACE 相同的评测指标。

```bash
torchrun --standalone --nproc_per_node=8 Trace/qwen3_5/evaluate_sft.py \
  --model ../MSLoc_data/Trace/experiments/qwen3_5/student_sft \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --output ../MSLoc_data/Trace/inference_results/qwen3_5/student_sft_test \
  --num-frames 40 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --bnd-ratio 0.2 \
  --max-new-tokens 256 \
  --max-pixels 112896 \
  --max-samples 0 \
  --clean
```

一次测试生成：

- `../MSLoc_data/Trace/inference_results/qwen3_5/student_sft_test/predictions.json`：与原 TRACE 评测兼容的预测；
- `../MSLoc_data/Trace/inference_results/qwen3_5/student_sft_test/metrics.json`：由原 `evaluate_long.py` 计算的指标；
- `predictions_rank*.jsonl`：各 GPU 的可恢复进度。

测试中断后删除 `--clean` 并加入 `--resume`，同时保持 GPU 数量、模型和推理参数不变。

当前完成的是正式 SFT 结构。OPD 和 GRPO 还没有迁移，不能用旧 TRACE 的 OPD/GRPO 脚本直接训练 Qwen checkpoint；后续迁移必须复用本目录的模型结构和事件级协议。
