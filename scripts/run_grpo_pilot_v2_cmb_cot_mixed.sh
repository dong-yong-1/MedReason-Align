#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

python training/grpo_training.py \
  --model_name_or_path "${BASE_MODEL_PATH:-/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct}" \
  --tokenizer_name_or_path "${BASE_MODEL_PATH:-/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct}" \
  --peft_path outputs/sft/cmb_cot_mixed \
  --ref_peft_path outputs/sft/cmb_cot_mixed \
  --train_file_dir data/grpo/cmb_grpo_pilot_v2_train.jsonl \
  --train_samples -1 \
  --reward_type med_choice_v2 \
  --max_steps 120 \
  --num_train_epochs 1 \
  --save_steps 60 \
  --save_strategy steps \
  --save_total_limit 2 \
  --output_dir outputs/grpo/cmb_cot_mixed_grpo_pilot_v2 \
  --dtype bfloat16 \
  --bf16 True \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing False \
  --beta 0.02 \
  --learning_rate 5.0e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --use_vllm False \
  --logging_steps 5 \
  --eval_strategy no \
  --use_peft True \
  --qlora False \
  --load_in_4bit False \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --num_generations 4 \
  --gradient_accumulation_steps 4 \
  --max_completion_length 64 \
  --preprocessing_num_workers 2
