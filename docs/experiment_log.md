# 实验记录模板

> 建议每天只记录一版主实验，避免日志太散。

## 基础信息

- 项目名：`MedReason-Align`
- 当前阶段：Direct SFT 消融后，进行 Direct + CoT mixed SFT 对照评估
- 当前数据版本：`data/sft/cmb_cot_mixed/`，5000 条 train，Direct 4014 条 + CoT 986 条，CoT 占比 19.72%
- 当前模型版本：Qwen2.5-7B-Instruct + LoRA，服务器输出目录 `outputs/sft/cmb_cot_mixed`
- 当前评测集版本：CMExam-test single/multi；CMB-val 仅作为 SFT validation sanity check

## 每日记录

| 日期 | 做了什么 | 产出文件 | 主要结果 | 发现的问题 | 下一步 |
|---|---|---|---|---|---|
| 2026-05-15 | 在服务器完成 `cmb_cot_mixed` LoRA SFT，并在 CMExam-test single/multi 上评估 | `outputs/sft/cmb_cot_mixed/`; `outputs/eval/cmb_cot_mixed_cmexam/outputs__sft__cmb_cot_mixed/results_2026-05-15T17-16-34.011530.json`; `logs/train_cmb_cot_mixed_clean_20260515-154924.log`; `logs/eval_cmb_cot_mixed_cmexam_20260515-171026.log` | SFT：3 epoch，train_loss 0.5619，eval_loss 0.2342，ppl 1.2639。CMExam-test：single acc 0.8408，multi exact_match 0.6078 | mixed 结果低于此前文档记录的 quality/optimized direct 外部结果；服务器当前只找到 mixed 评估 JSON，未找到 direct/quality/optimized 同环境评估 JSON | 先补齐同环境 Direct baseline 复评，再比较 CoT mixed 是否真实有效；检查 CoT 比例、评估 prompt 和输出格式是否影响答案抽取 |
| 2026-05-15 | 在 AutoDL 同环境用 `--log_samples` 复评 Optimized Direct 与 CoT/Mixed，并拉回 sample-level 输出做诊断 | `autodl_outputs/eval_samples/`; `data/analysis/cot_mixed_diagnostics_single.json`; `data/analysis/cot_mixed_diagnostics_multi.json`; `docs/cot_mixed_diagnostics.md` | Optimized Direct：single 0.8503，multi 0.6324；CoT/Mixed：single 0.8403，multi 0.6078。CoT/Mixed 低 0.0100 / 0.0245 | single 是 loglikelihood，不能分析生成格式；multi 中 CoT/Mixed 抽取成功率 1.0000、max output 5，下降更像 extra option rate 升高 | 补 C-Eval 和 CMB-val；做 CoT ratio ablation；对多选加入漏选/多选分解指标和 reward |
| 2026-05-17 | 跑通 GRPO pilot v1，分别以 `cmb_cot_mixed` 和 `cmb_optimized` SFT checkpoint 初始化；重点分析 CoT/Mixed 的多选 extra option 问题 | `outputs/grpo/cmb_cot_mixed_grpo_pilot`; `outputs/grpo/cmb_optimized_grpo_pilot`; `docs/grpo_pilot_diagnostics.md`; `scripts/grpo_med_choice_rewards.py`; `training/grpo_training.py` | CoT/Mixed SFT multi EM 0.6078，CoT/Mixed + GRPO v1 multi EM 0.6127；extra_option_rate 0.2549 -> 0.2549；missed_gold_rate 0.2059 -> 0.1961；precision 0.9043 -> 0.9007；recall 0.9187 -> 0.9216 | v1 pipeline 跑通，但没有降低 extra option；训练中 `completions/mean_length` 经常为 256，`clipped_ratio` 接近 1.0，reward/prompt 仍偏 recall-oriented 且有截断噪声 | 做 answer-only GRPO v2：缩短 max_completion_length，强化 precision/extra penalty，先做 reward sanity check 和训练前 completion debug |
| 2026-05-18 | 完成 Answer-boundary GRPO v2 小规模实验，只跑 CoT/Mixed 初始化；目标是答案边界对齐而不是长 CoT 训练 | `outputs/grpo/cmb_cot_mixed_grpo_pilot_v2`; `docs/grpo_answer_boundary_v2_diagnostics.md`; `scripts/grpo_med_choice_rewards_v2.py`; `scripts/build_grpo_pilot_v2_data.py`; `scripts/reward_sanity_check_v2.py`; `scripts/sample_grpo_completions_debug_v2.py` | reward sanity check PASS；服务器 v2 prompt pool 1899 条。训练 120 step 完成，train_runtime 0:06:58，KL 约 0.0018-0.0024，reward mean 约 0.68-1.27。CMExam-test-multi strict_letters exact_match = 0.6078 | `max_completion_length=64` 后 `clipped_ratio` 仍为 0.975-1.0，说明 answer-only prompt 仍未让模型自然停止；EM 回到 SFT 基线，未超过 v1 0.6127 | 不建议继续盲目加步数。先做 v2 sample-level diagnostics：extra/missed/precision/recall/F1；若 extra 未降，优先修 stop/EOS/格式约束或用更短 answer-only generation，再考虑 48-token 配置 |

## 实验对比

| 实验名 | 数据版本 | 模型 | 训练方法 | 公共 benchmark | 内部 holdout | 备注 |
|---|---|---|---|---|---|---|
| Base | 无 SFT | Qwen2.5-7B-Instruct | 原始 instruct 模型 | 文档记录：CMExam-test single acc 0.8295，multi exact_match 0.5245；服务器另有 C-Eval valid acc 0.7949 | 未跑 | CMExam base 结果来自项目进度文档，当前服务器只看到 C-Eval base JSON |
| Direct SFT baseline | `data/sft/cmb_baseline_direct/` 或历史 baseline direct | Qwen2.5-7B-Instruct + LoRA | Direct-answer SFT | 文档记录：CMExam-test single acc 0.8438，multi exact_match 0.6324 | 未跑 | 当前服务器未找到该实验的同环境评估 JSON，建议复跑确认 |
| Direct SFT quality-only | `data/sft/cmb_quality_direct/` | Qwen2.5-7B-Instruct + LoRA | Direct-answer SFT | 文档记录：CMExam-test single acc 0.8513，multi exact_match 0.6373 | 未跑 | 当前最重要的 direct 对照，需和 mixed 在同环境复评 |
| Direct SFT optimized | `data/sft/cmb_optimized_direct/` | Qwen2.5-7B-Instruct + LoRA | Direct-answer SFT | 文档记录：CMExam-test single acc 0.8521，multi exact_match 0.6373 | 未跑 | CMB-val 较差但外部泛化较好，仍需同环境保留 |
| Direct + CoT mixed SFT | `data/sft/cmb_cot_mixed/` | Qwen2.5-7B-Instruct + LoRA | 80.28% Direct + 19.72% DeepSeek teacher CoT | 已完成：CMExam-test single acc 0.8408，multi exact_match 0.6078 | 未跑 | 服务器结果显示 mixed 暂未超过 direct 历史结果；需做输出格式和样本选择分析 |
| CoT/Mixed + GRPO v1 | `data/grpo/cmb_grpo_pilot_train.jsonl` | Qwen2.5-7B-Instruct + LoRA，init/ref=`outputs/sft/cmb_cot_mixed` | GRPO，医学选择题 reward，max_completion_length=256 | CMExam-test-multi exact_match 0.6127；extra_option_rate 0.2549，precision 0.9007，recall 0.9216 | 未跑 | 相比 CoT/Mixed SFT，EM 小幅 +0.0049，但 extra option 未降；completion 大量截断 |
| CoT/Mixed + GRPO v2 | `data/grpo/cmb_grpo_pilot_v2_train.jsonl` | Qwen2.5-7B-Instruct + LoRA，init/ref=`outputs/sft/cmb_cot_mixed` | Answer-only GRPO，precision/boundary-oriented reward，max_completion_length=64 | CMExam-test-multi exact_match 0.6078 | 未跑 | reward sanity PASS，但 clipped_ratio 仍约 1.0，EM 未超过 v1；需先解决自然停止/格式控制 |

## 失败案例归档

| case_id | 失败类型 | 现象 | 原因推测 | 处理办法 |
|---|---|---|---|---|
|  | 漏诊 / 乱答 / 格式错 / 过度自信 |  |  |  |

## 结论记录

- 当前服务器上，`cmb_cot_mixed` 的训练和 CMExam-test 评估已经完成。
- 这次 mixed 实验暂时没有证明 CoT mixed 优于 Direct SFT；相对历史 direct 结果，CMExam single 和 multi 都偏低。
- 已补齐 `cmb_optimized` 与 `cmb_cot_mixed` 的同环境 sample-level 复评。当前证据显示，multi 下降主要不是答案抽取失败、输出过长或 `max_gen_toks` 截断，而是 CoT/Mixed 更容易输出额外选项。
- GRPO v1 跑通了训练/评估闭环，但未降低 CoT/Mixed 的 extra_option_rate；v2 改为 answer-only 和 precision/boundary-oriented reward 后，CMExam-test-multi EM 仍为 0.6078，且训练中 `clipped_ratio` 仍接近 1.0。
- 当前 GRPO 结论：reward 方向可以继续研究，但现阶段不应把 v1/v2 解释为已修复多选边界；下一步优先解决 completion 自然停止、严格 answer-only 格式和 sample-level extra/missing 诊断。
