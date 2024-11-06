INPUT_PATH="./data/python_train_sample.jsonl"
EXTRACTED_KEY="input"
GENERATED_OUTPUT_PATH="./data/generated/python_train_sample.json"
PREPROCESS_OUTPUT_PATH="./data/preprocessed/python_train_sample.jsonl"


python generate.py \
    --data_path $INPUT_PATH \
    --extracted_key $EXTRACTED_KEY \
    --output_path $GENERATED_OUTPUT_PATH \
    --max_length 2048 \
    --model_path deepseek-ai/deepseek-coder-1.3b-instruct

python preprocess.py \
    --input_path $INPUT_PATH \
    --output_path $PREPROCESS_OUTPUT_PATH \
    --result_path $GENERATED_OUTPUT_PATH