MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"

OUTPUT_DIR="./output"
DATA_DIR="./data"
DATA_SPLIT="python_human_test"
SAVE_POSTFIX="python_human_test"

PREDICTION_TYPE="classification" # regression

CKPT_DIR="./ckpts/code_efficiency_classifier"

mkdir -p ${OUTPUT_DIR}
python inference.py --data_split=${DATA_SPLIT} --ckpt_dir=${CKPT_DIR} \
    --data_dir=${DATA_DIR} --model_path=${MODEL_PATH} --save_postfix=${SAVE_POSTFIX} \
    --output_dir=${OUTPUT_DIR} \
    --prediction_type=${PREDICTION_TYPE}