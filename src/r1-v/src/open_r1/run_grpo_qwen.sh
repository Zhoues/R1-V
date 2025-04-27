export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=1
torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    grpo.py \
    --output_dir /home/zhouenshen/code/VILA/runs/grpo/Clevr_CoGenT_TrainA_70K_Complex \
    --model_name_or_path /home/vlm/pretrain_model/Qwen2.5-VL-7B-Instruct \
    --dataset_name /home_sfs/zhouenshen/dataset/Reasoning/Clevr_CoGenT_TrainA_70K_Complex \
    --deepspeed /home/zhouenshen/code/VILA/R1-V/src/r1-v/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2_5-VL-7B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance