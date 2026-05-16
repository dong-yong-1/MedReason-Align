# CoT/Mixed 诊断报告

## 1. 背景

当前主结果显示，CoT/Mixed 相比 Base 在 CMExam-test 上仍有提升，但低于 Optimized Direct SFT。本报告基于本地代码、训练数据，以及从 AutoDL 无卡模式拉回的 lm-eval sample-level 输出进行诊断。

需要特别区分两类评测：CMExam-test-single 是 lm-eval `multiple_choice` / loglikelihood 评测，sample 文件不包含模型生成文本；CMExam-test-multi 是 `generate_until` 评测，sample 文件包含模型生成的短答案。

## 2. 当前主结果表

| 模型 | CMB-val | CMExam-test-single | CMExam-test-multi | C-Eval-med |
|---|---:|---:|---:|---:|
| Base Qwen2.5-7B-Instruct | 0.8042 | 0.8295 | 0.5245 | 0.8230 |
| Optimized Direct SFT | 0.8000 | 0.8521 | 0.6373 | 待跑 |
| CoT SFT / Mixed | 待跑 | 0.8408 | 0.6078 | 待跑 |

## 3. 本地文件可用性

已找到并用于本次诊断：

- `data/eval/lm_eval/cmexam_test_single.jsonl`：CMExam-test-single gold/task，6606 条。
- `data/eval/lm_eval/cmexam_test_multi.jsonl`：CMExam-test-multi gold/task，204 条。
- `autodl_outputs/eval_samples/cmb_optimized_cmexam/outputs__sft__cmb_optimized/samples_cmexam_test_single_2026-05-15T22-39-59.775161.jsonl`：Optimized Direct single sample。
- `autodl_outputs/eval_samples/cmb_optimized_cmexam/outputs__sft__cmb_optimized/samples_cmexam_test_multi_2026-05-15T22-39-59.775161.jsonl`：Optimized Direct multi sample。
- `autodl_outputs/eval_samples/cmb_cot_mixed_cmexam/outputs__sft__cmb_cot_mixed/samples_cmexam_test_single_2026-05-15T22-35-27.510760.jsonl`：CoT/Mixed single sample。
- `autodl_outputs/eval_samples/cmb_cot_mixed_cmexam/outputs__sft__cmb_cot_mixed/samples_cmexam_test_multi_2026-05-15T22-35-27.510760.jsonl`：CoT/Mixed multi sample。
- `autodl_outputs/eval_samples/*/results_*.json`：lm-eval 聚合结果。
- `autodl_outputs/logs/eval_cmb_optimized_cmexam_samples_retry_20260515-223528.log`：Direct 评估日志。
- `autodl_outputs/logs/eval_cmb_cot_mixed_cmexam_samples_retry_20260515-223055.log`：CoT/Mixed 评估日志。
- `autodl_outputs/logs/train_cmb_cot_mixed_5090_20260515-215722.log`：CoT/Mixed 训练日志。
- `autodl_outputs/logs/sft_truncation_1024_20260515-222916.log`：SFT max_length 截断诊断日志。
- `data/sft/cmb_optimized_direct/cmb_sft_optimized_direct.jsonl`：Direct SFT 训练数据，5000 条。
- `data/sft/cmb_cot_mixed/cmb_sft_cot_mixed.jsonl`：CoT/Mixed 训练数据，5000 条。
- `tasks/cmexam_choice/cmexam_test_single.yaml`、`tasks/cmexam_choice/cmexam_test_multi.yaml`：lm-eval task。

仍缺少：

- C-Eval-med 上 Direct 和 CoT/Mixed 的评估结果。
- CoT/Mixed 在 CMB-val 上的最终外部评估结果。
- 如果要分析 single-choice 的“生成格式/输出长度/答案标签”，需要另跑生成式 single 评测；当前 loglikelihood sample 不包含生成文本。

## 4. 诊断结论摘要

- 本轮同环境复评结果：Optimized Direct SFT single `0.8503`、multi `0.6324`；CoT/Mixed single `0.8403`、multi `0.6078`。CoT/Mixed 分别低 `0.0100` 和 `0.0245`。
- single 是 loglikelihood 评测，不能据此诊断“答案：/分析：”格式、生成输出长度或截断；只能恢复选项预测并做准确率、分桶和错误重叠。
- multi 是真实生成短答案，CoT/Mixed 并没有明显答案抽取问题：抽取成功率 `1.0000`，非法答案率 `0.0000`，最长输出 5 个字符。
- multi 下降更像是多选边界变松：CoT/Mixed 的 extra option rate `0.2549`，高于 Direct 的 `0.1765`；但漏选率反而更低，CoT/Mixed `0.2059`，Direct `0.2549`。
- max_length 不是当前训练数据的主要问题：`max_length=1024` 下 CoT/Mixed 和 Direct 训练数据均无 source/target truncation，`answer_label_lost=0`，`gold_lost=0`。
- 训练数据存在输出分布冲突风险：CoT/Mixed 为 4014 条 Direct + 986 条 teacher CoT，CoT 占比 `19.72%`；训练输出同时包含极短 `答案：X` 和较长 `分析：...答案：X`。

## 5. 答案抽取与格式合规

### CMExam-test-single

single 为 loglikelihood 选项打分，sample 中没有生成文本。本表中的抽取成功率表示脚本能从 lm-eval 分数恢复预测选项，不代表生成文本格式合规。

| 模型 | total | accuracy | lm_eval_choice_prediction_rate | generated_text_available_rate |
|---|---:|---:|---:|---:|
| Direct SFT | 6606 | 0.8503 | 1.0000 | 0.0000 |
| CoT/Mixed | 6606 | 0.8403 | 1.0000 | 0.0000 |

### CMExam-test-multi

| 模型 | total | exact_match | answer_extraction_success_rate | invalid_answer_rate | format_compliance_rate | conflicting_answer_rate |
|---|---:|---:|---:|---:|---:|---:|
| Direct SFT | 204 | 0.6324 | 0.9951 | 0.0049 | 0.9951 | 0.0000 |
| CoT/Mixed | 204 | 0.6078 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

结论：在 multi 任务上，CoT/Mixed 不是因为答案抽取失败而低于 Direct；它的抽取成功率反而略高。

## 6. 输出长度与截断风险

single 没有生成文本，不能分析输出长度。

multi 的输出均为短答案：

| 模型 | avg_output_length | p50 | p90 | p95 | max | empty_output_rate |
|---|---:|---:|---:|---:|---:|---:|
| Direct SFT | 3.250 | 3 | 5 | 5 | 5 | 0.0000 |
| CoT/Mixed | 3.431 | 3 | 5 | 5 | 5 | 0.0000 |

评测日志显示 multi 使用 `generate_until`，`until: ["\n"]`、`max_gen_toks: 16`、`do_sample: False`。本次 multi 最长输出 5 个字符，因此没有证据支持“评测时答案被 max_gen_toks 截断”。

训练侧 max_len 排查也显示：`max_length=1024` 下 CoT/Mixed 最大 total tokens 为 751，Direct 最大 total tokens 为 498，两者 `source_trunc=0`、`target_trunc=0`、`answer_label_lost=0`、`gold_lost=0`。

## 7. 分题型表现

single：

- Direct：5617 / 6606，accuracy `0.8503`。
- CoT/Mixed：5551 / 6606，accuracy `0.8403`。
- 错误重叠：both_correct 5301，both_wrong 739，Direct 对 CoT 错 316，Direct 错 CoT 对 250。

multi：

- Direct：129 / 204，exact match `0.6324`。
- CoT/Mixed：124 / 204，exact match `0.6078`。
- 错误重叠：both_correct 112，both_wrong 63，Direct 对 CoT 错 17，Direct 错 CoT 对 12。

multi 按 gold 答案长度：

| gold 长度 | 样本数 | Direct EM | CoT/Mixed EM |
|---:|---:|---:|---:|
| 2 | 40 | 0.6500 | 0.6500 |
| 3 | 71 | 0.6338 | 0.5775 |
| 4 | 69 | 0.5942 | 0.5217 |
| 5 | 24 | 0.7083 | 0.8750 |

CoT/Mixed 在 5 选全选题上更好，但在 3、4 个答案的多选题上下降明显。

multi 漏选/多选：

| 模型 | missed_gold_rate | extra_option_rate | same_answer_length_rate | pred length distribution |
|---|---:|---:|---:|---|
| Direct SFT | 0.2549 | 0.1765 | 0.6716 | 0:1, 1:7, 2:44, 3:69, 4:57, 5:26 |
| CoT/Mixed | 0.2059 | 0.2549 | 0.6520 | 1:4, 2:37, 3:71, 4:51, 5:41 |

结论：CoT/Mixed 倾向于输出更多选项，减少漏选，但增加额外错误选项，exact match 因此受损。

## 8. 分 CoT-worthiness / 病例丰富度表现

single 分桶：

| 子集 | 样本数 | Direct | CoT/Mixed |
|---|---:|---:|---:|
| short | 4840 | 0.8496 | 0.8421 |
| medium | 1097 | 0.8469 | 0.8350 |
| long | 669 | 0.8610 | 0.8356 |
| non_case | 5537 | 0.8432 | 0.8344 |
| case_rich | 1069 | 0.8868 | 0.8709 |
| low CoT-worthiness | 5397 | 0.8442 | 0.8351 |
| medium CoT-worthiness | 576 | 0.8594 | 0.8611 |
| high CoT-worthiness | 633 | 0.8942 | 0.8657 |

multi 分桶样本较少，结论需谨慎：

| 子集 | 样本数 | Direct | CoT/Mixed |
|---|---:|---:|---:|
| short | 180 | 0.6389 | 0.6222 |
| medium | 23 | 0.5652 | 0.4783 |
| low CoT-worthiness | 201 | 0.6368 | 0.6119 |
| medium CoT-worthiness | 3 | 0.3333 | 0.3333 |
| reasoning_like | 191 | 0.6492 | 0.6126 |
| recall_like | 13 | 0.3846 | 0.5385 |

基于当前轻量规则，没有看到 CoT/Mixed 在 high CoT-worthiness 或 case-rich 子集上稳定超过 Direct。single 的 high CoT-worthiness 反而 Direct 更高；multi 的高 CoT-worthiness 样本基本没有，不能下结论。

## 9. 错误重叠分析

single：

- both_correct_count: 5301
- both_wrong_count: 739
- direct_correct_cot_wrong_count: 316
- direct_wrong_cot_correct_count: 250

multi：

- both_correct_count: 112
- both_wrong_count: 63
- direct_correct_cot_wrong_count: 17
- direct_wrong_cot_correct_count: 12

典型 multi 案例：

- Direct 对、CoT/Mixed 错：`cmexam_test_178`，gold `ABCD`，Direct `ABCD`，CoT/Mixed `ABCDE`。错误类型：CoT/Mixed 多选了 E。
- CoT/Mixed 对、Direct 错：`cmexam_test_987`，gold `ABCDE`，Direct `CE`，CoT/Mixed `ABCDE`。错误类型：Direct 漏选。
- 两者都错：`cmexam_test_200`，gold `ABDE`，Direct `BE`，CoT/Mixed `BCDE`。错误类型：Direct 漏选，CoT/Mixed 同时漏 A、多 C。

典型 single 案例只能展示恢复出的选项预测，不能展示模型生成文本：

- Direct 对、CoT/Mixed 错：`cmexam_test_77`，gold `A`，Direct `A`，CoT/Mixed `E`。
- CoT/Mixed 对、Direct 错：`cmexam_test_39`，gold `D`，Direct `B`，CoT/Mixed `D`。
- 两者都错：`cmexam_test_8`，gold `C`，Direct `B`，CoT/Mixed `B`。

## 10. CoT 数据与训练分布分析

Direct 训练数据：

- 路径：`data/sft/cmb_optimized_direct/cmb_sft_optimized_direct.jsonl`
- total: 5000
- format: direct_answer 5000
- 题型：单选 4759，多选 241
- contains_answer_label_rate: 1.0000
- contains_analysis_label_rate: 0.0000
- assistant 平均长度：4.119 字符，p95 4，max 8

CoT/Mixed 训练数据：

- 路径：`data/sft/cmb_cot_mixed/cmb_sft_cot_mixed.jsonl`
- total: 5000
- format: direct_answer 4014，cot_teacher_deepseek 986
- CoT 占比：19.72%
- 题型：单选 4083，多选 916，C 型 1
- contains_answer_label_rate: 1.0000
- contains_analysis_label_rate: 0.1972
- assistant 平均长度：30.551 字符，p50 4，p90 135，p95 157，max 248
- long_rationale_count: 7

训练日志显示 CoT/Mixed 已完成 3 epoch，`train_loss=0.5613`，`eval_loss=0.2354`，`perplexity=1.2654`。结合训练数据格式，当前没有证据表明“分析部分没有参与训练损失”；相反，CoT 样本 assistant 文本中包含 `分析：...答案：...`，训练脚本按 assistant 输出计算 loss 时会覆盖这部分。

## 11. 评测一致性检查

Direct 和 CoT/Mixed 这次同环境评估使用相同 task：

- `tasks/cmexam_choice_clean/cmexam_test_single.yaml`
- `tasks/cmexam_choice_clean/cmexam_test_multi.yaml`

评估日志共同点：

- base model: `/root/autodl-tmp/models/Qwen/Qwen/Qwen2___5-7B-Instruct`
- dtype: `bfloat16`
- batch_size: 1
- num_fewshot: None / 0-shot
- random seed: 0，numpy/torch/fewshot seed: 1234
- multi gen kwargs: `until: ["\n"]`、`max_gen_toks: 16`、`do_sample: False`

差异：

- Direct adapter: `outputs/sft/cmb_optimized`
- CoT/Mixed adapter: `outputs/sft/cmb_cot_mixed`

日志中都出现 lm-eval 警告：Qwen instruct/chat variant 未应用 chat template。这是两者共同设置，不解释 CoT/Mixed 相对 Direct 的差距，但后续若改 prompt/chat template，需要两者一起重评。

## 12. 结论

不能简单说“CoT 没用”。基于当前本地文件和 AutoDL sample 输出，更谨慎的结论是：

当前 Mixed 策略没有超过 Optimized Direct SFT，主要不是由 CMExam multi 的答案抽取失败、输出过长或 max_gen_toks 截断造成。更有证据支持的原因是：CoT/Mixed 改变了多选题的选择边界，倾向输出更多选项，降低漏选但增加额外选项；同时 Direct/CoT 混合训练让输出分布从极短答案扩展到部分长 rationale，可能带来目标分布冲突。single-choice 当前是 loglikelihood 评测，不能验证生成格式问题。

仍属于推测、需要后续实验确认的原因包括：teacher rationale 噪声、CoT 样本更难导致训练分布偏移、测试集中大量 recall 型题不需要显式推理、评测指标只奖励最终答案不奖励推理质量。

## 13. 下一步实验建议

- 补跑 C-Eval Direct 和 CoT/Mixed。
- 补跑 CoT/Mixed CMB-val。
- 对 high CoT-worthiness 子集单独构造更可靠的评估集，尤其补足 multi 中的高 CoT-worthiness 样本。
- 做 Direct-only / CoT-only / Mixed ratio ablation，优先尝试 10% / 20% / 40% CoT mixing ratio。
- 缩短 teacher rationale，减少长 rationale 对短答案任务的分布扰动。
- 为 single-choice 额外跑生成式评估，统一 prompt 为“先分析，再输出答案：X”，再检查格式合规与答案抽取。
- 增强答案抽取器，并单独报告 answer extraction success rate、format compliance rate、extra option rate、missed gold rate。
- 在 GRPO 中加入 answer reward、format reward，以及多选题的漏选/多选分解 reward。
