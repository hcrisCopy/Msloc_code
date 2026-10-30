# MSLoc-AF：ActionFormer 实验


## 1. 环境与路径配置

已有原 msloc 环境可直接复用

```bash
conda activate msloc
cd msloc_code/MSLoc-AF

cd vendor/actionformer/libs/utils
python setup.py build_ext --inplace
cd ../../../..

python verify_official.py --device cuda:0
```

最后一条命令会检查官方模型和 NMS 是否能正常运行


## 2. 提取两路特征

```bash
python extract_features.py \
    --config ./configs/features.yaml \
    --device-ids 0,1,2,3,4,5,6,7 \
    --batch-size 2 \
    --frame-count-tolerance 4
```

一次前向同时取得 Mamba 输入 `x` 和输出 `res`，分别保存到`MSLoc_data/artifacts/features`下的`xclip/`、`mamba/`，配对关系在 `manifest.json`。中断后重复相同命令，完成的视频跳过，中断视频重提。


## 3. 训练

分别训练三个版本；每条命令只训练指定版本：

```bash
python train.py \
    --config ./configs/ActionFormer_xclip.yaml \
    --device-ids 0,1,2,3,4,5,6,7 \
    --max_epoch 10 \
    --train-batch-size 32 \
    --num-workers 0 \
    --save_dir ../../MSLoc_data/ActionFormer/results/xclip \
    --seed 42
```

```bash
python train.py \
    --config ./configs/ActionFormer_mamba.yaml \
    --device-ids 0,1,2,3,4,5,6,7 \
    --max_epoch 10 \
    --train-batch-size 32 \
    --num-workers 0 \
    --save_dir ../../MSLoc_data/ActionFormer/results/mamba \
    --seed 42
```

```bash
python train.py \
    --config ./configs/ActionFormer_fusion.yaml \
    --device-ids 0,1,2,3,4,5,6,7 \
    --max_epoch 10 \
    --train-batch-size 32 \
    --num-workers 0 \
    --save_dir ../../MSLoc_data/ActionFormer/results/fusion \
    --seed 42
```

结果分别保存在`MSLoc_data/ActionFormer/results/`下的`xclip/`、`mamba/`、`fusion/`


## 4. 推理

```bash
python eval.py \
    --config ./configs/ActionFormer_xclip.yaml \
    --model_path ../../MSLoc_data/ActionFormer/results/xclip/final.pt \
    --output_dir ../../MSLoc_data/ActionFormer/results/xclip/eval \
    --device-ids 0,1,2,3,4,5,6,7 \
    --resume
```

```bash
python eval.py \
    --config ./configs/ActionFormer_mamba.yaml \
    --model_path ../../MSLoc_data/ActionFormer/results/mamba/final.pt \
    --output_dir ../../MSLoc_data/ActionFormer/results/mamba/eval \
    --device-ids 0,1,2,3,4,5,6,7 \
    --resume
```

```bash
python eval.py \
    --config ./configs/ActionFormer_fusion.yaml \
    --model_path ../../MSLoc_data/ActionFormer/results/fusion/final.pt \
    --output_dir ../../MSLoc_data/ActionFormer/results/fusion/eval \
    --device-ids 0,1,2,3,4,5,6,7 \
    --resume
```

预测阈值预先固定为 `score_threshold: 0.5`



## 5. 先用 10 个完整视频检查

先单独生成固定名单（总计 8 个训练 + 2 个测试）：

```bash
python prepare_subset.py \
    --train-json ../../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
    --test-json ../../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
    --output ./artifacts/subset_10 \
    --max-videos 10 \
    --seed 42
```

然后分别执行；这里展示 xclip，其他版本替换配置及输出目录：

```bash
python extract_features.py \
    --config ./configs/features.yaml \
    --annotations ./artifacts/subset_10/train.json ./artifacts/subset_10/test.json \
    --device-ids 0
```

```bash
python train.py \
    --config ./configs/ActionFormer_xclip.yaml \
    --device-ids 0 \
    --train-json ./artifacts/subset_10/train.json \
    --output ./artifacts/runs_10_ddp/xclip \
    --epochs 1 \
    --warmup-epochs 0 \
    --train-batch-size 8 \
    --num-workers 0 \
    --checkpoint-every-updates 1 \
    --seed 42
```

```bash
python eval.py \
    --config ./configs/ActionFormer_xclip.yaml \
    --device-ids 0 \
    --annotations ./artifacts/subset_10/test.json \
    --model_path ./artifacts/runs_10_ddp/xclip/final.pt \
    --output_dir ./artifacts/runs_10_ddp/xclip/test \
    --resume
```
