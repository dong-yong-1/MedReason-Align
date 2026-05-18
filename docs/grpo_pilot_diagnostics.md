# GRPO Pilot 诊断报告

## 1. 实验目的

本 pilot 不是大规模刷分实验，目标是验证 GRPO 是否能修正 CoT/Mixed SFT 在 CMExam-test-multi 上的多选边界变松问题，尤其关注 `extra_option_rate` 是否下降，同时观察 `missed_gold_rate` 是否明显上升。

## 2. 数据构造方式

GRPO pilot 数据由 `scripts/build_grpo_pilot_data.py` 构造。输入优先级为：

1. `data/processed/cmb_clean/cmb_train_vector_dedup_bucket.jsonl`
2. `data/sft/cmb_cot_mixed/cmb_sft_cot_mixed.jsonl`
3. `data/sft/cmb_optimized_direct/cmb_sft_optimized_direct.jsonl`

输出：

- `data/grpo/cmb_grpo_pilot_train.jsonl`
- `data/grpo/cmb_grpo_pilot_summary.md`

数据只保留 prompt、gold answer、question、options、valid_options 和元数据，不包含 SFT assistant 输出，不包含 teacher analysis，也不会把 gold answer 放入 prompt。

## 3. Reward 设计

医学选择题 reward 位于 `scripts/grpo_med_choice_rewards.py`，接入 `training/grpo_training.py --reward_type med_choice`。

当前线性 reward：

```text
reward = format_reward
       + exact_match_reward
       + 0.5 * option_f1
       - 0.3 * extra_count
       - 0.2 * missing_count
       + invalid_penalty
       + too_long_penalty
```

其中 extra option penalty 略重于 missing option penalty，因为当前主要问题是 CoT/Mixed 更容易多选额外选项。

## 4. 两组 GRPO 初始化 checkpoint

| 实验 | policy/init checkpoint | reference checkpoint | output |
|---|---|---|---|
| CoT/Mixed + GRPO | `outputs/sft/cmb_cot_mixed` | frozen `outputs/sft/cmb_cot_mixed` | `outputs/grpo/cmb_cot_mixed_grpo_pilot` |
| Direct + GRPO | `outputs/sft/cmb_optimized` | frozen `outputs/sft/cmb_optimized` | `outputs/grpo/cmb_optimized_grpo_pilot` |

两组使用相同 GRPO 数据、reward、prompt、超参和训练步数；唯一差异是 SFT 初始化 checkpoint。

## 5. 四组模型评估表

| 模型 | CMExam-test-multi exact_match | answer_extraction_success_rate | invalid_answer_rate | missed_gold_rate | extra_option_rate | option_precision | option_recall | option_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct SFT | 待补充到本报告 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| CoT/Mixed SFT | 0.6078 | 1.0000 | 0.0000 | 0.2059 | 0.2549 | 0.9043 | 0.9187 | 0.9114 |
| Direct + GRPO | 已训练，待补充到本报告 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| CoT/Mixed + GRPO v1 | 0.6127 | 1.0000 | 0.0000 | 0.1961 | 0.2549 | 0.9007 | 0.9216 | 0.9110 |

## 6. extra_option_rate 是否下降

没有下降。CoT/Mixed SFT 与 CoT/Mixed + GRPO v1 的 `extra_option_rate` 均为 0.2549。v1 没有修正 CoT/Mixed 的多选边界变松问题。

## 7. missed_gold_rate 是否上升

没有上升，反而从 0.2059 降到 0.1961。但由于 extra_option_rate 未下降、precision 略降、recall 略升，这更像 reward 仍偏 recall-oriented，而不是成功收紧多选边界。

## 8. exact_match 是否回升

CoT/Mixed + GRPO v1 从 0.6078 小幅升至 0.6127，但 option F1 从 0.9114 略降到 0.9110。这个 EM 小涨不能单独解释为多选边界修复。

## 9. 典型 case

待后续从 sample-level 评估文件中补充：

- GRPO 修正多选的例子
- GRPO 仍然多选的例子
- GRPO 从正确改错的例子

## 10. 下一步建议

- v1 已确认训练和评估链路能闭环，但没有降低 extra_option_rate。
- v1 训练中 completion 经常打满 `max_completion_length=256`，`clipped_ratio` 接近 1.0，因此后续不应继续盲目加步数。
- 已设计 v2 answer-only GRPO：缩短 completion、强化 precision reward 和 extra option penalty，先做答案边界对齐，再考虑把 CoT 加回来。
- 保留 Direct + GRPO 对照，避免把 GRPO 收益误归因于 CoT/Mixed。
