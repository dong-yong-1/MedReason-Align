#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

if [ ! -f data/grpo/cmb_grpo_pilot_train.jsonl ]; then
  python scripts/build_grpo_pilot_data.py
fi

cot_log="logs/grpo_cmb_cot_mixed_pilot_$(date +%Y%m%d-%H%M%S).log"
bash scripts/run_grpo_pilot_cmb_cot_mixed.sh > "$cot_log" 2>&1
echo "CoT/Mixed GRPO log: $cot_log"

direct_log="logs/grpo_cmb_optimized_pilot_$(date +%Y%m%d-%H%M%S).log"
bash scripts/run_grpo_pilot_cmb_optimized.sh > "$direct_log" 2>&1
echo "Direct GRPO log: $direct_log"
