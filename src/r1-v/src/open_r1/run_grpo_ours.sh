export DEBUG_MODE="true"
export LOG_PATH="./debug_log_7b.txt"
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=2
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    grpo_ours.py \
    --output_dir /home/zhouenshen/code/VILA/runs/grpo/Clevr_CoGenT_TrainA_70K_Complex \
    --model_name_or_path /home/zhouenshen/code/VILA/runs/train/NVILA-8B-depth-sft-mlp-all/model \
    --dataset_name sat_176k \
    --deepspeed /home/zhouenshen/code/VILA/R1-V/src/r1-v/local_scripts/zero3_vila.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name NVILA8B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 1   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance

# --model_name_or_path /home/zhouenshen/code/VILA/runs/train/NVILA-8B-depth-align-osd+sat-mlp-9M/model \
# --model_name_or_path /home/zhouenshen/code/VILA/ckpt/pretrain_weights/NVILA-8B \