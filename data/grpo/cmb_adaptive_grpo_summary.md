# Adaptive GRPO 数据构造报告

## 输入与输出

- input_path: `data/processed/cmb_clean/cmb_cot_candidates_scored.jsonl`
- output_jsonl: `data/grpo/cmb_adaptive_grpo_train.jsonl`
- full_distribution_json: `data/grpo/cmb_adaptive_grpo_full_distribution.json`
- direct_review_jsonl: `data/grpo/cmb_adaptive_grpo_direct_review_50.jsonl`
- cot_review_jsonl: `data/grpo/cmb_adaptive_grpo_cot_review_50.jsonl`
- 注意：输入来自 clean pool 的全量 `cot_candidate_score` 打分文件，不使用 `cmb_cot_mixed`，输出不包含 teacher CoT 文本。

## Difficulty 规则

- easy: `cot_candidate_score <= -0.1`
- medium: `-0.1 < cot_candidate_score <= 0.9`
- hard: `cot_candidate_score > 0.9`

## 全量 Clean Pool 分布

- total: 233188

| difficulty | count |
|---|---:|
| easy | 77978 |
| medium | 84361 |
| hard | 70849 |

### 全量分数统计

```json
{
  "min": -1.6,
  "p25": -0.1,
  "p50": 0.3,
  "p75": 1.2,
  "p90": 1.9,
  "max": 5.1
}
```

## 训练集 1:1:1 分层采样

- total: 4500

| difficulty | selected |
|---|---:|
| easy | 1500 |
| medium | 1500 |
| hard | 1500 |

### 训练集题型分布

```json
{
  "multi": 509,
  "single": 3991
}
```

### 训练集答案长度分布

```json
{
  "3": 163,
  "1": 3991,
  "5": 75,
  "4": 168,
  "2": 102,
  "6": 1
}
```

### 跳过样本

```json
{}
```

## 人工检查抽样

- direct/easy review: `data/grpo/cmb_adaptive_grpo_direct_review_50.jsonl`
- cot/hard review: `data/grpo/cmb_adaptive_grpo_cot_review_50.jsonl`
- direct review 从 easy 中优先抽单选题，用于检查低 CoT 价值样本是否确实更适合直接答题。
- cot review 从 hard 中优先抽多选题，用于检查高 CoT 价值样本是否确实更需要推理/逐项分析。
