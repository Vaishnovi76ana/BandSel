#!/bin/bash

# ===========================
# Model Settings
# ===========================
MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_PATH="Qwen/Qwen2.5-7B-Instruct"
MODEL_TYPE="qwen"

# ===========================
# Dataset
# ===========================
BENCHMARK="gsm8k"          # gsm8k | math
# ===========================
# Output
# ===========================
OUTPUT_DIR="outputs/${MODEL_NAME}/bandit/${BENCHMARK}"

# ===========================
# Evaluation
# ===========================
MAX_NUM_EXAMPLES=100000000000000
MAX_NEW_TOKENS=512
EVAL_BATCH_SIZE=32
RANDOM_SEED=42

# ===========================
# Bandit Settings
# ===========================
ARM_SIZE=25
NUM_TRAINING_ARMS=10
NUM_CHALLENGER_ARMS=10
NUM_TOTAL_TRACES=1000
NUM_VALIDATION=1000
NUM_EXPLORATION=5

MAX_ITERATIONS=30

# ===========================
# Hyperparameters
# ===========================
LAMBDA_REG=1.0
CONFIDENCE_SCALE=1.0
BETA=0.0
VAL_EVAL_SIZE=50

# Skip SFT?
# 0 = perform SFT
# 1 = skip SFT
SKIP_SFT=0

CUDA_VISIBLE_DEVICES=0 python case_bandit.py \
    --output-dir ${OUTPUT_DIR} \
    --model-path ${MODEL_PATH} \
    --tokenizer-path ${TOKENIZER_PATH} \
    --model-name ${MODEL_NAME} \
    --model-type ${MODEL_TYPE} \
    --benchmark ${BENCHMARK} \
    --data-type ${DATA_TYPE} \
    --max_num_examples ${MAX_NUM_EXAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --random_seed ${RANDOM_SEED} \
    --arm_size ${ARM_SIZE} \
    --num_training_arms ${NUM_TRAINING_ARMS} \
    --num_challenger_arms ${NUM_CHALLENGER_ARMS} \
    --num_total_traces ${NUM_TOTAL_TRACES} \
    --num_validation ${NUM_VALIDATION} \
    --num_exploration ${NUM_EXPLORATION} \
    --max_iterations ${MAX_ITERATIONS} \
    --lambda_reg ${LAMBDA_REG} \
    --confidence_scale ${CONFIDENCE_SCALE} \
    --beta ${BETA} \
    --val_eval_size ${VAL_EVAL_SIZE} \
    --skip_sft ${SKIP_SFT}