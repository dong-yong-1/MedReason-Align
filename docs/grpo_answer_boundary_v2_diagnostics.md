# Answer-boundary GRPO v2 诊断报告

## 1. 实验目标

GRPO v2 的目标不是继续训练长 CoT，而是先做答案边界对齐：降低多选题 extra option rate，提高 precision，同时避免 missed_gold_rate 明显恶化。

v1 已经证明 pipeline 能跑通，但 CoT/Mixed + GRPO v1 的 extra_option_rate 未下降，precision 略降，recall 略升，并且训练中 completion 经常打满 `max_completion_length=256`，`clipped_ratio` 接近 1.0。因此 v2 改为 answer-only prompt，先把最终答案边界训稳。

## 2. 数据构造

v2 使用 `scripts/build_grpo_pilot_v2_data.py` 构造 GRPO prompt pool。该数据不是离线生成的模型 completion，而是训练时供 policy on-policy 生成 completion 的题目池，并保留 gold answer 供 reward 计算。

服务器无卡模式下已生成：

- output: `data/grpo/cmb_grpo_pilot_v2_train.jsonl`
- summary: `data/grpo/cmb_grpo_pilot_v2_summary.md`
- total: 1899
- single: 1000
- multi: 899
- input fallback: `data/sft/cmb_cot_mixed/cmb_sft_cot_mixed.jsonl`

服务器上的 `data/processed/cmb_clean/cmb_train_vector_dedup_bucket.jsonl` 不存在，因此脚本按优先级 fallback 到 CoT/Mixed SFT 数据。多选 2/3/4/6 桶未取满，属于当前服务器数据可用性限制。

## 3. Reward 设计

v2 reward 位于 `scripts/grpo_med_choice_rewards_v2.py`，核心是 precision / boundary oriented：

- valid extract: +0.2
- invalid extract: -1.0
- exact match: +2.0
- precision reward: +0.8 * precision
- recall reward: +0.2 * recall
- extra option penalty: -0.7 * extra_count
- missing option penalty: -0.2 * missing_count
- 恰好一个 `答案：`: +0.2
- 没有 `答案：`: -0.3
- 多个 `答案：`: -0.5
- completion 过长: -0.3
- 答案后有明显多余解释: -0.3

## 4. Reward Sanity Check

服务器已运行：

```bash
python scripts/reward_sanity_check_v2.py
```

结果：PASS。输出文件：

- `data/grpo/reward_sanity_check_v2.md`

关键排序符合预期：

- `答案：ABDE` reward = 3.4000
- `答案：ABCDE` reward = 0.5400
- `答案：ABE` reward = 1.1500
- `没有答案` reward = -2.1000
- `答案：ABCDE\n答案：ABDE` reward = 2.7000

这说明 clean exact 最高，extra option 被明显压低，无答案最低，多答案标签和解释性尾巴会被扣分。

## 5. 训练前 Sample Completion Debug

脚本已准备：

- `scripts/sample_grpo_completions_debug_v2.py`

无卡模式下已验证脚本会安全退出，不会强行在 CPU 加载 7B 模型。需要 GPU 后运行：

```bash
python scripts/sample_grpo_completions_debug_v2.py \
  --num-samples 30 \
  --num-generations 4 \
  --max-new-tokens 64
```

输出：

- `data/grpo/cot_mixed_before_grpo_v2_samples.jsonl`
- `data/grpo/cot_mixed_before_grpo_v2_samples.md`

该步骤必须在训练前查看，用于判断 answer-only prompt 下模型是否仍输出长分析、多个答案标签或 extra option。

## 6. 训练命令

只有 sample completion debug 通过后，才启动训练：

```bash
TRAIN_LOG=logs/grpo_cmb_cot_mixed_pilot_v2_$(date +%Y%m%d-%H%M%S).log
nohup bash scripts/run_grpo_pilot_v2_cmb_cot_mixed.sh > "$TRAIN_LOG" 2>&1 &
echo $! > logs/grpo_cmb_cot_mixed_pilot_v2.pid
echo "$TRAIN_LOG"
```

如果 `max_completion_length=64` 仍然导致 clipped_ratio 很高，再跑 48-token 备用配置，输出目录独立为 `_v2_48`：

```bash
TRAIN_LOG=logs/grpo_cmb_cot_mixed_pilot_v2_48_$(date +%Y%m%d-%H%M%S).log
nohup bash scripts/run_grpo_pilot_v2_cmb_cot_mixed_48.sh > "$TRAIN_LOG" 2>&1 &
echo $! > logs/grpo_cmb_cot_mixed_pilot_v2_48.pid
echo "$TRAIN_LOG"
```

## 7. 训练与评估结果

v2 已完成 120 step 小规模训练，输出目录：

- `outputs/grpo/cmb_cot_mixed_grpo_pilot_v2`

训练后段日志显示：

| 指标 | 观察 |
|---|---|
| train_runtime | 0:06:58.87 |
| train_samples_per_second | 2.292 |
| train_steps_per_second | 0.286 |
| KL | 约 0.0018-0.0024 |
| reward mean | 后段约 0.6788-1.271 |
| reward std | 后段约 1.669-1.987 |
| completions/mean_length | 约 64 |
| completions/clipped_ratio | 0.975-1.0 |

CMExam-test-multi 结果：

| 模型 | exact_match |
|---|---:|
| CoT/Mixed SFT | 0.6078 |
| CoT/Mixed + GRPO v1 | 0.6127 |
| CoT/Mixed + GRPO v2 | 0.6078 |

v2 没有超过 v1，也没有超过 CoT/Mixed SFT 基线。由于 `clipped_ratio` 仍然几乎为 1.0，本轮不能证明 answer-boundary GRPO 已经解决 extra option 问题。

## 8. 待补充 Sample-level 诊断

v2 训练完成后至少评估：

- CoT/Mixed SFT: `outputs/sft/cmb_cot_mixed`
- CoT/Mixed + GRPO v1: `outputs/grpo/cmb_cot_mixed_grpo_pilot`
- CoT/Mixed + GRPO v2: `outputs/grpo/cmb_cot_mixed_grpo_pilot_v2`

重点指标：

- EM
- extra_option_rate
- missed_gold_rate
- precision / recall / F1
- answer_extraction_success_rate
- invalid_answer_rate
- format_compliance_rate
- pred length distribution
- gold answer length bucket performance
- completion mean length
- clipped_ratio
- reward mean / reward std / KL

当前已知 exact_match，但仍需补齐 sample-level 指标：

- extra_option_rate
- missed_gold_rate
- option_precision
- option_recall
- option_f1
- answer_extraction_success_rate
- invalid_answer_rate
- pred length distribution

只有这些指标出来后，才能判断 v2 是“extra 降了但 missing 上升”还是“extra 根本没降”。

## 9. 下一步建议

不建议继续盲目增加 step。优先做：

1. 对 v2 评估样本跑 sample-level diagnostics，确认 extra/missing/precision/recall。
2. 如果 extra 未降，先修 generation 终止：stop/EOS、严格 answer-only prompt、必要时更短 `max_completion_length`。
3. 如果 extra 降但 missed 大涨，说明 reward 太保守，需要提高 recall 或 F1 权重。
4. 48-token 备用配置只应作为验证 clipped_ratio 的实验，不应覆盖 v2 64-token 输出。

v2 成功标准优先看 extra_option_rate 是否低于 0.2549、precision 是否高于 0.9007、clipped_ratio 是否显著低于 v1，同时 missed_gold_rate 不应大幅高于 0.2059。
