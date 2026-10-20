#!/bin/bash

# Configuration
# API_KEY="0ad05f171df94b16aa9873208eba5a22.aV7kCNxAzFpygOEq"
API_KEY="7117a451621443b1b550a08acadad5e1.Mljp89b7Rwz7SeiA"
MODEL="GLM-4.6V-Flash"

# PRED_FILE="RQ/our_ablation_row2.json" 
# PRED_FILE="RQ/our_best.json" 
PRED_FILE="RQ/trace.json" 
# PRED_FILE="RQ/trace_MSLoc_PG.json"
GT_FILE="test_all_1209_0119_long.json"

# OUTPUT_FILE="cot_eval_results_our_ablation_row2.jsonl" 
OUTPUT_FILE="cot_eval_results_trace_11.jsonl" 
# OUTPUT_FILE="cot_eval_results_trace_MSLoc_PG.jsonl"

echo "Starting Evaluation..."
echo "GT File: ${GT_FILE}"
echo "Pred File: ${PRED_FILE}"
echo "Output File: ${OUTPUT_FILE}"

python evaluate_RQ.py \
    --api_key "${API_KEY}" \
    --model "${MODEL}" \
    --gt_file "${GT_FILE}" \
    --pred_file "${PRED_FILE}" \
    --output_file "${OUTPUT_FILE}"
