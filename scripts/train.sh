DATA_PATH="data/python_cpp_all_train.jsonl"
DATA_EVAL_PATH="data/python_cpp_all_val.jsonl"
OUTPUT_PATH="ckpts/code_efficiency_classifier"
MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"
mkdir -p $OUTPUT_PATH

deepspeed --master_port=8000 train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --data_eval_path $DATA_EVAL_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 10 \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 50 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "wandb" \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True