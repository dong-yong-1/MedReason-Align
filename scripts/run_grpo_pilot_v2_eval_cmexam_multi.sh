#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/root/autodl-tmp/models/Qwen/Qwen/Qwen2___5-7B-Instruct}"
TASK_DIR="${TASK_DIR:-tasks/cmexam_choice_clean}"
if [ ! -d "$TASK_DIR" ]; then
  TASK_DIR="tasks/cmexam_choice"
fi

mkdir -p logs outputs/eval_samples

run_eval() {
  local name="$1"
  local peft_path="$2"
  local output_path="outputs/eval_samples/${name}_cmexam_multi"
  local log_path="logs/eval_${name}_cmexam_multi_$(date +%Y%m%d-%H%M%S).log"
  mkdir -p "$output_path"
  echo "Evaluating ${name}; log=${log_path}; output=${output_path}"
  lm_eval \
    --model hf \
    --model_args "pretrained=${BASE_MODEL_PATH},peft=${peft_path},dtype=bfloat16,trust_remote_code=True" \
    --tasks cmexam_test_multi \
    --include_path "$TASK_DIR" \
    --batch_size 1 \
    --log_samples \
    --output_path "$output_path" \
    > "$log_path" 2>&1
}

run_eval "cmb_cot_mixed" "outputs/sft/cmb_cot_mixed"
run_eval "cmb_cot_mixed_grpo_pilot" "outputs/grpo/cmb_cot_mixed_grpo_pilot"
run_eval "cmb_cot_mixed_grpo_pilot_v2" "outputs/grpo/cmb_cot_mixed_grpo_pilot_v2"
