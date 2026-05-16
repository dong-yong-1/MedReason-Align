# CoT/Mixed 诊断报告

## 1. 背景

当前主结果显示，CoT/Mixed 相比 Base 在 CMExam-test 上仍有提升，但低于 Optimized Direct SFT。本报告区分本地可直接验证的事实、缺失的 AutoDL 输入，以及后续拿到原始预测后可运行的诊断项。

## 2. 当前主结果表

| 模型 | CMB-val | CMExam-test-single | CMExam-test-multi | C-Eval-med |
|---|---:|---:|---:|---:|
| Base Qwen2.5-7B-Instruct | 0.8042 | 0.8295 | 0.5245 | 0.8230 |
| Optimized Direct SFT | 0.8000 | 0.8521 | 0.6373 | 待跑 |
| CoT SFT / Mixed | 待跑 | 0.8408 | 0.6078 | 待跑 |

## 3. 本地文件可用性

- `data/eval/lm_eval/cmexam_test_single.jsonl`：gold/task；可用于诊断：是；说明：found locally; 6606 lines
- `autodl_outputs/eval_samples/cmb_optimized_cmexam/outputs__sft__cmb_optimized/samples_cmexam_test_single_2026-05-15T22-39-59.775161.jsonl`：direct prediction；可用于诊断：是；说明：found locally; 6606 lines
- `autodl_outputs/eval_samples/cmb_cot_mixed_cmexam/outputs__sft__cmb_cot_mixed/samples_cmexam_test_single_2026-05-15T22-35-27.510760.jsonl`：cot/mixed prediction；可用于诊断：是；说明：found locally; 6606 lines
- `data/sft/cmb_optimized_direct/cmb_sft_optimized_direct.jsonl`：direct train；可用于诊断：是；说明：found locally; 5000 lines
- `data/sft/cmb_cot_mixed/cmb_sft_cot_mixed.jsonl`：cot/mixed train；可用于诊断：是；说明：found locally; 5000 lines
- `tasks/cmexam_choice/cmexam_test_single.yaml`：lm-eval task；可用于诊断：是；说明：found locally; 13 lines
- `tasks/cmexam_choice/cmexam_test_multi.yaml`：lm-eval task；可用于诊断：是；说明：found locally; 23 lines
- `data/eval/lm_eval/cmexam_test_multi.jsonl`：CMExam-test multi gold/task；可用于诊断：是；说明：found locally; 204 lines
- `data/eval/lm_eval/cmexam_valid_single.jsonl`：CMExam-valid single gold/task；可用于诊断：是；说明：found locally; 6600 lines
- `data/eval/lm_eval/cmexam_valid_multi.jsonl`：CMExam-valid multi gold/task；可用于诊断：是；说明：found locally; 210 lines
- `data/raw/hf/CMB/CMB-Exam/CMB-val/CMB-val-merge.json`：CMB-val raw；可用于诊断：是；说明：found locally; 4483 lines
- `data/eval_internal/holdout_v1.jsonl`：internal holdout；可用于诊断：是；说明：found locally; 0 lines
- `data/raw/hf/ceval-exam/medical_jsonl/basic_medicine_dev.jsonl`：C-Eval medical raw；可用于诊断：是；说明：found locally; 5 lines
- `data/raw/hf/ceval-exam/medical_jsonl/basic_medicine_test.jsonl`：C-Eval medical raw；可用于诊断：是；说明：found locally; 175 lines
- `data/raw/hf/ceval-exam/medical_jsonl/basic_medicine_val.jsonl`：C-Eval medical raw；可用于诊断：是；说明：found locally; 19 lines
- `data/raw/hf/ceval-exam/medical_jsonl/clinical_medicine_dev.jsonl`：C-Eval medical raw；可用于诊断：是；说明：found locally; 5 lines
- `data/raw/hf/ceval-exam/medical_jsonl/clinical_medicine_test.jsonl`：C-Eval medical raw；可用于诊断：是；说明：found locally; 200 lines
- `data/raw/hf/ceval-exam/medical_jsonl/clinical_medicine_val.jsonl`：C-Eval medical raw；可用于诊断：是；说明：found locally; 22 lines
- `data/raw/hf/ceval-exam/medical_jsonl/physician_dev.jsonl`：C-Eval medical raw；可用于诊断：是；说明：found locally; 5 lines
- `data/raw/hf/ceval-exam/medical_jsonl/physician_test.jsonl`：C-Eval medical raw；可用于诊断：是；说明：found locally; 443 lines
- `data/raw/hf/ceval-exam/medical_jsonl/physician_val.jsonl`：C-Eval medical raw；可用于诊断：是；说明：found locally; 49 lines
- `data/raw/hf/ceval-exam/medical_jsonl/veterinary_medicine_dev.jsonl`：C-Eval medical raw；可用于诊断：是；说明：found locally; 5 lines
- `data/raw/hf/ceval-exam/medical_jsonl/veterinary_medicine_val.jsonl`：C-Eval medical raw；可用于诊断：是；说明：found locally; 23 lines

## 4. 诊断结论摘要

- 已基于 Direct 与 CoT/Mixed 原始预测输出完成答案抽取、格式、长度和错误重叠诊断。
- CoT/Mixed 相对 Direct 的 accuracy/exact-match 差值为 -0.0100。
- 该数据集来自 lm-eval single-choice loglikelihood 样本，不能用于分析模型生成文本中的“答案：/分析：”格式和输出长度。
- 错误重叠：both_correct=5301, both_wrong=739, Direct对CoT错=316, Direct错CoT对=250。

## 5. 答案抽取与格式合规

| 模型 | accuracy_or_exact_match | lm_eval_choice_prediction_rate | generated_text_available_rate | answer_extraction_success_rate | invalid_answer_rate | format_compliance_rate | contains_answer_label_rate | contains_analysis_label_rate | conflicting_answer_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct SFT | 0.8503 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| CoT/Mixed | 0.8403 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

## 6. 输出长度与截断风险

| 模型 | avg_output_length | p50_output_length | p90_output_length | p95_output_length | max_output_length | empty_output_rate | overlong_output_rate_256_chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct SFT | 0 | 0.0000 | 0.0000 | 0.0000 | 0 | 1.0000 | 0.0000 |
| CoT/Mixed | 0 | 0.0000 | 0.0000 | 0.0000 | 0 | 1.0000 | 0.0000 |

## 7. 分题型表现

### Direct SFT

{
  "single": {
    "total": 6606,
    "correct": 5617,
    "accuracy_or_exact_match": 0.850288,
    "extraction_success_rate": 1.0
  }
}

{
  "total": 0,
  "correct": 0,
  "accuracy_or_exact_match": 0.0,
  "extraction_success_rate": 0.0
}

{
  "total": 0,
  "missed_gold_rate": 0.0,
  "extra_option_rate": 0.0,
  "same_answer_length_rate": 0.0,
  "pred_answer_length_distribution": {},
  "gold_answer_length_distribution": {}
}

### CoT/Mixed

{
  "single": {
    "total": 6606,
    "correct": 5551,
    "accuracy_or_exact_match": 0.840297,
    "extraction_success_rate": 1.0
  }
}

{
  "total": 0,
  "correct": 0,
  "accuracy_or_exact_match": 0.0,
  "extraction_success_rate": 0.0
}

{
  "total": 0,
  "missed_gold_rate": 0.0,
  "extra_option_rate": 0.0,
  "same_answer_length_rate": 0.0,
  "pred_answer_length_distribution": {},
  "gold_answer_length_distribution": {}
}

## 8. 分 CoT-worthiness / 病例丰富度表现

### Direct SFT
- length_bucket_metrics: `{"medium": {"total": 1097, "correct": 929, "accuracy_or_exact_match": 0.846855, "extraction_success_rate": 1.0}, "short": {"total": 4840, "correct": 4112, "accuracy_or_exact_match": 0.849587, "extraction_success_rate": 1.0}, "long": {"total": 669, "correct": 576, "accuracy_or_exact_match": 0.860987, "extraction_success_rate": 1.0}}`
- case_richness_bucket_metrics: `{"non_case": {"total": 5537, "correct": 4669, "accuracy_or_exact_match": 0.843236, "extraction_success_rate": 1.0}, "case_rich": {"total": 1069, "correct": 948, "accuracy_or_exact_match": 0.88681, "extraction_success_rate": 1.0}}`
- cot_worthiness_bucket_metrics: `{"low": {"total": 5397, "correct": 4556, "accuracy_or_exact_match": 0.844173, "extraction_success_rate": 1.0}, "high": {"total": 633, "correct": 566, "accuracy_or_exact_match": 0.894155, "extraction_success_rate": 1.0}, "medium": {"total": 576, "correct": 495, "accuracy_or_exact_match": 0.859375, "extraction_success_rate": 1.0}}`
- recall_bucket_metrics: `{"reasoning_like": {"total": 6049, "correct": 5114, "accuracy_or_exact_match": 0.845429, "extraction_success_rate": 1.0}, "recall_like": {"total": 557, "correct": 503, "accuracy_or_exact_match": 0.903052, "extraction_success_rate": 1.0}}`
### CoT/Mixed
- length_bucket_metrics: `{"medium": {"total": 1097, "correct": 916, "accuracy_or_exact_match": 0.835005, "extraction_success_rate": 1.0}, "short": {"total": 4840, "correct": 4076, "accuracy_or_exact_match": 0.842149, "extraction_success_rate": 1.0}, "long": {"total": 669, "correct": 559, "accuracy_or_exact_match": 0.835575, "extraction_success_rate": 1.0}}`
- case_richness_bucket_metrics: `{"non_case": {"total": 5537, "correct": 4620, "accuracy_or_exact_match": 0.834387, "extraction_success_rate": 1.0}, "case_rich": {"total": 1069, "correct": 931, "accuracy_or_exact_match": 0.870907, "extraction_success_rate": 1.0}}`
- cot_worthiness_bucket_metrics: `{"low": {"total": 5397, "correct": 4507, "accuracy_or_exact_match": 0.835094, "extraction_success_rate": 1.0}, "high": {"total": 633, "correct": 548, "accuracy_or_exact_match": 0.865719, "extraction_success_rate": 1.0}, "medium": {"total": 576, "correct": 496, "accuracy_or_exact_match": 0.861111, "extraction_success_rate": 1.0}}`
- recall_bucket_metrics: `{"reasoning_like": {"total": 6049, "correct": 5051, "accuracy_or_exact_match": 0.835014, "extraction_success_rate": 1.0}, "recall_like": {"total": 557, "correct": 500, "accuracy_or_exact_match": 0.897666, "extraction_success_rate": 1.0}}`

## 9. 错误重叠分析

{
  "both_correct_count": 5301,
  "both_wrong_count": 739,
  "direct_correct_cot_wrong_count": 316,
  "direct_wrong_cot_correct_count": 250
}

## 10. CoT 数据与训练分布分析

### direct_train
```json
{
  "path": "data/sft/cmb_optimized_direct/cmb_sft_optimized_direct.jsonl",
  "total": 5000,
  "format_counts": {
    "direct_answer": 5000
  },
  "question_type_counts": {
    "单项选择题": 4759,
    "多项选择题": 241
  },
  "contains_answer_label_rate": 1.0,
  "contains_analysis_label_rate": 0.0,
  "avg_assistant_length": 4.119,
  "p50_assistant_length": 4.0,
  "p90_assistant_length": 4.0,
  "p95_assistant_length": 4.0,
  "max_assistant_length": 8,
  "short_rationale_count": 0,
  "long_rationale_count": 0,
  "top_rationale_prefixes": []
}
```
### cot_or_mixed_train
```json
{
  "path": "data/sft/cmb_cot_mixed/cmb_sft_cot_mixed.jsonl",
  "total": 5000,
  "format_counts": {
    "direct_answer": 4014,
    "cot_teacher_deepseek": 986
  },
  "question_type_counts": {
    "单项选择题": 4083,
    "多项选择题": 916,
    "C型选择题": 1
  },
  "contains_answer_label_rate": 1.0,
  "contains_analysis_label_rate": 0.1972,
  "avg_assistant_length": 30.551,
  "p50_assistant_length": 4.0,
  "p90_assistant_length": 135.0,
  "p95_assistant_length": 157.0,
  "max_assistant_length": 248,
  "short_rationale_count": 0,
  "long_rationale_count": 7,
  "top_rationale_prefixes": [
    [
      "患者心前区疼痛在休息或清",
      4
    ],
    [
      "患者长期上腹痛、进食缓解",
      3
    ],
    [
      "心电图显示提前出现的宽大",
      3
    ],
    [
      "患者有铁锈钉刺伤史，出现",
      3
    ],
    [
      "患者血压160~179/",
      2
    ],
    [
      "慢性锰中毒主要损害锥体外",
      2
    ],
    [
      "患者年轻男性，运动扭伤后",
      2
    ],
    [
      "急诊胃镜和肠镜可直接观察",
      2
    ],
    [
      "患者血压150/95mm",
      2
    ],
    [
      "患者外阴口腔反复溃疡、结",
      2
    ]
  ]
}
```

## 11. 评测一致性检查

- 本地 `tasks/cmexam_choice/cmexam_test_single.yaml` 使用 `multiple_choice`，prompt 明确要求“直接给出答案选项，不需要解释”。
- 本地 `tasks/cmexam_choice/cmexam_test_multi.yaml` 使用 `generate_until`，`max_gen_toks: 16`、`do_sample: false`，regex 只抽取 `[A-E]{1,5}`。
- 本地 task YAML 的 `data_files` 是 AutoDL 绝对路径 `/root/autodl-tmp/MedicalGPT/...`，在本地直接运行 lm-eval 前需要改为本地路径或在 AutoDL 上运行。
- Direct 与 CoT/Mixed 是否完全使用相同 eval command、checkpoint/adaptor 加载方式，目前本地缺少 AutoDL 评测日志，待拉回日志确认。

## 12. 结论

当前 Mixed 策略没有超过 Direct SFT，不能简单解释为“CoT 没用”。基于本地训练数据和 task 配置，较谨慎的判断是：简单混入 teacher rationale 可能引入输出分布冲突；CMExam task 又明确要求直接输出答案，评测指标只看最终答案，不直接奖励推理质量。其他可能原因包括输出变长、格式抽取风险、teacher rationale 噪声、测试集中 recall 题较多、CoT 样本更难等，这些需要 AutoDL 原始预测输出才能实证。

## 13. 下一步实验建议

- 从 AutoDL 拉取 Direct 与 CoT/Mixed 在 CMExam-test single/multi 上的原始预测输出。
- 补跑 C-Eval Direct 和 CoT/Mixed。
- 补跑 CoT/Mixed CMB-val。
- 对 high CoT-worthiness 子集单独评估。
- 做 Direct-only / CoT-only / Mixed ratio ablation，尝试 10% / 20% / 40% CoT mixing ratio。
- 缩短 teacher rationale，并检查 rationale 模板化和噪声。
- 统一 eval prompt，或明确评估“先分析，再输出答案：X”的生成式流程。
- 增强答案抽取器，并单独报告 answer extraction success rate 和 format compliance rate。
- 在 GRPO 中加入 answer reward 和 format reward。

## 14. 需要从 AutoDL 拉取的文件

### 必须拉取

1. Direct SFT CMExam-test single 原始预测输出
   - 可能路径：`/root/autodl-tmp/MedicalGPT/outputs/eval/*direct*cmexam*`、`/root/autodl-tmp/MedicalGPT/lm_eval_outputs/*direct*`
   - 用途：答案抽取、格式、长度、错误案例诊断
2. Direct SFT CMExam-test multi 原始预测输出
   - 可能路径：同上，需包含 per-sample generated text
   - 用途：多选 exact match、漏选/多选分析
3. CoT/Mixed CMExam-test single 原始预测输出
   - 可能路径：`/root/autodl-tmp/MedicalGPT/outputs/eval/cmb_cot_mixed_cmexam/` 或 lm-eval samples 文件
   - 用途：与 Direct 对齐做错误重叠
4. CoT/Mixed CMExam-test multi 原始预测输出
   - 可能路径：同上
   - 用途：分析 CoT/Mixed 多选下降原因
5. 对应 gold / task jsonl 文件
   - 可能路径：`/root/autodl-tmp/MedicalGPT/data/eval/lm_eval/cmexam_test_*.jsonl`
   - 用途：保证预测顺序、id 和 gold 一致

### 建议拉取

1. lm-eval 完整输出目录，尤其 samples / predictions / requests 文件。
2. generation config / eval command 日志。
3. 训练日志。
4. checkpoint config 和 adapter config。
5. CoT/Mixed 训练数据。
6. Direct SFT 训练数据。

示例命令模板：

```bash
rsync -avP -e 'ssh -p <port>' root@<autodl_host>:/root/autodl-tmp/MedicalGPT/outputs/eval/ ./autodl_outputs/eval/
rsync -avP -e 'ssh -p <port>' root@<autodl_host>:/root/autodl-tmp/MedicalGPT/logs/ ./autodl_outputs/logs/
rsync -avP -e 'ssh -p <port>' root@<autodl_host>:/root/autodl-tmp/MedicalGPT/data/eval/lm_eval/ ./data/eval/lm_eval/
```
