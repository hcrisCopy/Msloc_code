1. Student SFT（加 --freeze-backbone true）
bash

python Trace/run_opd_grpo.py student-sft \
  --devices 0 \
  --nproc-per-node 1 \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --base-model ../MSLoc_data/Trace/ckpts/trace-uni \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --deepspeed Trace/scripts/zero2.json \
  --class-feature-path ../MSLoc_data/Trace/class_features_bge.pt \
  --output ../MSLoc_data/Trace/experiments/opd_grpo_smoke/student_sft \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --num-workers 0 \
  --run-name smoke_student_sft \
  --max-samples 4 \
  --save-total-limit 2 \
  --freeze-backbone true \
  --resume none \
  --clean
产物：../MSLoc_data/Trace/experiments/opd_grpo_smoke/student_sft/

2. 测 Student SFT
bash

python Trace/run_opd_grpo.py test \
  --devices 0 \
  --model ../MSLoc_data/Trace/experiments/opd_grpo_smoke/student_sft \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/inference_results/opd_grpo_smoke/student_sft_test \
  --metrics-output ../MSLoc_data/Trace/inference_results/opd_grpo_smoke/student_sft_test/metrics.json \
  --prompt-file Trace/trace/prompts/dvc.txt \
  --num-frames 40 \
  --max-new-tokens 128 \
  --batch-size 1 \
  --sample-num 2 \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --clean
3. 构建三份样本
bash

python Trace/run_opd_grpo.py build-samples \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --output ../MSLoc_data/Trace/experiments/opd_grpo_smoke/train_paired_samples.json \
  --near-negative-seconds 1.0 \
  --max-records 32 \
  --stratified-debug \
  --paired-only \
  --require-reference \
  --clean

python Trace/run_opd_grpo.py build-samples \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --output ../MSLoc_data/Trace/experiments/opd_grpo_smoke/test_paired_samples.json \
  --near-negative-seconds 1.0 \
  --max-records 4 \
  --stratified-debug \
  --paired-only \
  --require-reference \
  --clean

python Trace/run_opd_grpo.py build-samples \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --output ../MSLoc_data/Trace/experiments/opd_grpo_smoke/grpo_training_samples.json \
  --near-negative-seconds 1.0 \
  --max-records 6 \
  --stratified-debug \
  --clean
4. Teacher SFT（加 --freeze-backbone true）
bash

python Trace/run_opd_grpo.py train-paired-teacher \
  --devices 0 \
  --nproc-per-node 1 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo_smoke/train_paired_samples.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --base-model ../MSLoc_data/Trace/ckpts/trace-uni \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --deepspeed Trace/scripts/zero2.json \
  --class-feature-path ../MSLoc_data/Trace/class_features_bge.pt \
  --output ../MSLoc_data/Trace/experiments/opd_grpo_smoke/teacher_sft \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --num-workers 0 \
  --run-name smoke_teacher_sft \
  --max-samples 4 \
  --save-total-limit 2 \
  --freeze-backbone true \
  --resume none \
  --clean
产物：.../opd_grpo_smoke/teacher_sft/。这步会走 paired 输入 + 两个身份向量的 1152 token 路径。

5. 测 Teacher SFT
bash

python Trace/run_opd_grpo.py test-teacher \
  --devices 0 \
  --nproc-per-node 1 \
  --test-samples ../MSLoc_data/Trace/experiments/opd_grpo_smoke/test_paired_samples.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --teacher-model ../MSLoc_data/Trace/experiments/opd_grpo_smoke/teacher_sft \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/inference_results/opd_grpo_smoke/teacher_sft_test \
  --metrics-output ../MSLoc_data/Trace/inference_results/opd_grpo_smoke/teacher_sft_test/metrics.json \
  --prompt-file Trace/trace/prompts/dvc.txt \
  --version v1_mistral \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --max-new-tokens 128 \
  --teacher-iou-gate 0.3 \
  --max-samples 2 \
  --resume none \
  --clean
6. 预检筛选
bash

python Trace/run_opd_grpo.py check-teacher \
  --devices 0 \
  --nproc-per-node 1 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo_smoke/train_paired_samples.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --student-model ../MSLoc_data/Trace/experiments/opd_grpo_smoke/student_sft \
  --teacher-model ../MSLoc_data/Trace/experiments/opd_grpo_smoke/teacher_sft \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/experiments/opd_grpo_smoke/teacher_precheck.json \
  --selected-output ../MSLoc_data/Trace/experiments/opd_grpo_smoke/opd_selected_samples.json \
  --prompt-file Trace/trace/prompts/dvc.txt \
  --version v1_mistral \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --max-new-tokens 128 \
  --teacher-iou-gate 0.3 \
  --max-samples 16 \
  --enforce-better true \
  --resume none \
  --clean
依赖门槛：opd_selected_samples.json 至少要有 1 条 Teacher 严格优于 Student 的样本，否则第 7 步 OPD 没数据。如果 selected records=0，把第 1、4 步的 --max-samples 从 4 提到 16 重训再预检。

7. OPD（默认就是 freeze_backbone=true，无需加）
bash

python Trace/run_opd_grpo.py opd \
  --devices 0 \
  --nproc-per-node 1 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo_smoke/opd_selected_samples.json \
  --student-model ../MSLoc_data/Trace/experiments/opd_grpo_smoke/student_sft \
  --teacher-model ../MSLoc_data/Trace/experiments/opd_grpo_smoke/teacher_sft \
  --teacher-check-result ../MSLoc_data/Trace/experiments/opd_grpo_smoke/teacher_precheck.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --deepspeed Trace/scripts/zero2.json \
  --class-feature-path ../MSLoc_data/Trace/class_features_bge.pt \
  --output ../MSLoc_data/Trace/experiments/opd_grpo_smoke/opd \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --num-workers 0 \
  --run-name smoke_opd \
  --max-samples 2 \
  --save-total-limit 2 \
  --rollout-max-new-tokens 64 \
  --save-rollouts true \
  --rollout-output ../MSLoc_data/Trace/experiments/opd_grpo_smoke/opd/rollouts \
  --resume none \
  --clean
提醒：OPD 单卡要同时装「student + 冻结 teacher」两个 7B 模型，显存会到 ~35GB，比前面几步紧张。如果这步单卡 OOM，说明 OPD 在单卡上放不下两个模型，那属于多卡才能做的环节，到时告诉我。

8. 测 OPD
bash

python Trace/run_opd_grpo.py test \
  --devices 0 \
  --model ../MSLoc_data/Trace/experiments/opd_grpo_smoke/opd \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/inference_results/opd_grpo_smoke/opd_test \
  --metrics-output ../MSLoc_data/Trace/inference_results/opd_grpo_smoke/opd_test/metrics.json \
  --prompt-file Trace/trace/prompts/dvc.txt \
  --num-frames 40 \
  --max-new-tokens 128 \
  --batch-size 1 \
  --sample-num 2 \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --clean
9. GRPO（默认 freeze_backbone=true，无需加；已按最新代码去掉 --text-max-words）
bash

python Trace/run_opd_grpo.py grpo \
  --devices 0 \
  --nproc-per-node 1 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo_smoke/grpo_training_samples.json \
  --starting-model ../MSLoc_data/Trace/experiments/opd_grpo_smoke/opd \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --deepspeed Trace/scripts/zero2.json \
  --class-feature-path ../MSLoc_data/Trace/class_features_bge.pt \
  --entailment-model-path ../MSLoc_data/Trace/ckpts/nli-deberta-v3-small \
  --output ../MSLoc_data/Trace/experiments/opd_grpo_smoke/grpo \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --num-workers 0 \
  --run-name smoke_grpo \
  --max-samples 2 \
  --save-total-limit 2 \
  --group-size 2 \
  --max-new-tokens 64 \
  --entailment-device cuda \
  --entailment-batch-size 4 \
  --save-rollouts true \
  --rollout-output ../MSLoc_data/Trace/experiments/opd_grpo_smoke/grpo/rollouts \
  --resume none \
  --clean
注意 --group-size 2 是下限（必须 ≥2 才能算组内 advantage）。

10. 测 GRPO
bash

python Trace/run_opd_grpo.py test \
  --devices 0 \
  --model ../MSLoc_data/Trace/experiments/opd_grpo_smoke/grpo \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/inference_results/opd_grpo_smoke/grpo_test \
  --metrics-output ../MSLoc_data/Trace/inference_results/opd_grpo_smoke/grpo_test/metrics.json \
  --prompt-file Trace/trace/prompts/dvc.txt \
  --num-frames 40 \
  --max-new-tokens 128 \
  --batch-size 1 \
  --sample-num 2 \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --clean