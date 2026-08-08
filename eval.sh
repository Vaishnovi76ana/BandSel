BENCHMARK="gsm8k" # "gsm8k", "math"
OUPTUT_DIR="outputs/Qwen2.5-7B-Instruct/bandit/${BENCHMARK}/"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
MODEL_TYPE="qwen" # "llama3", "qwen"
DATA_TYPE="test" # "train", "test"

# Generation Settings
MAX_NUM_EXAMPLES=100000000000000
MAX_NEW_TOKENS=512 # 512 for gsm8k, 1024 for math
EVAL_BATCH_SIZE=128
TEMPERATURE=0.0
SEED=42

# TokenSkip Settings
#ADAPTER_PATH="model/GSM8K-Compressed-Qwen2.5-7B-Instruct"
ADAPTER_PATH="outputs/Qwen2.5-7B-Instruct/bandit/math/adapter_iter_30"
COMPRESSION_RATIO=1.0


CUDA_VISIBLE_DEVICES=0 python ./evaluation.py --output-dir ${OUPTUT_DIR} --model-path ${MODEL_PATH} --tokenizer-path ${MODEL_PATH} \
    --model-type ${MODEL_TYPE} --data-type ${DATA_TYPE}  --max_num_examples ${MAX_NUM_EXAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} --eval_batch_size ${EVAL_BATCH_SIZE} --temperature ${TEMPERATURE} --seed ${SEED} --benchmark ${BENCHMARK} --use_vllm \
    --adapter-path ${ADAPTER_PATH} --compression_ratio ${COMPRESSION_RATIO} --use_adapter