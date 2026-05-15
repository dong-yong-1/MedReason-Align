cd /root/autodl-tmp/MedicalGPT
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/autodl-tmp/hf-cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1

python training/supervised_finetuning.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
  --train_file_dir data/sft/cmb_baseline \
  --validation_file_dir data/sft/cmb_baseline \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples 5000 \
  --max_eval_samples 280 \
  --model_max_length 1024 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --eval_steps 200 \
  --eval_strategy steps \
  --save_steps 500 \
  --save_strategy steps \
  --save_total_limit 3 \
  --gradient_accumulation_steps 16 \
  --preprocessing_num_workers 2 \
  --output_dir outputs/sft/cmb_baseline \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --cache_dir ./cache