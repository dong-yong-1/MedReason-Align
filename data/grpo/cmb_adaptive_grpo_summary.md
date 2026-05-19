# Adaptive GRPO Data Summary

## Inputs And Outputs

- input: `data/processed/cmb_clean/cmb_cot_candidates_scored.jsonl`
- train_output: `data/grpo/cmb_adaptive_grpo_train.jsonl`
- full_distribution_json: `data/grpo/cmb_adaptive_grpo_full_distribution.json`
- direct_review_jsonl: `data/grpo/cmb_adaptive_grpo_direct_review_50.jsonl`
- brief_review_jsonl: `data/grpo/cmb_adaptive_grpo_brief_review_50.jsonl`
- cot_review_jsonl: `data/grpo/cmb_adaptive_grpo_cot_review_50.jsonl`

## Classification Rules

- `reasoning_score = case_info_score + option_confusion_score - definition_penalty - low_reasoning_penalty`
- `definition_level == strong` -> direct
- `reasoning_score <= -0.3` -> direct
- `reasoning_score > 0.5` -> cot, except multi-choice concept/recall rows without real case context are brief
- remaining rows -> brief

## Full Distribution

- total: 232106

### By Difficulty

| key | count |
|---|---:|
| direct | 45927 |
| brief | 111764 |
| cot | 74415 |

### By Question Type

| key | count |
|---|---:|
| multi | 23504 |
| single | 208602 |

### Full Reasoning Score Quantiles

```json
{
  "min": -1.6,
  "p25": -0.1,
  "p50": 0.3,
  "p75": 0.9,
  "p90": 1.4,
  "max": 3.1
}
```

## Train Sampling

- target_per_difficulty: 1500
- total: 4500

### Train By Difficulty

| key | count |
|---|---:|
| direct | 1500 |
| brief | 1500 |
| cot | 1500 |

### Train By Question Type

| key | count |
|---|---:|
| multi | 453 |
| single | 4047 |

### Train Reasoning Score Quantiles

```json
{
  "min": -1.5,
  "p25": -0.5,
  "p50": 0.3,
  "p75": 0.9,
  "p90": 1.4,
  "max": 3.1
}
```

## Review Sampling

- review_count_per_difficulty: 50
- direct review: random single-choice rows from direct; falls back to all direct rows if needed.
- brief review: random rows from brief.
- cot review: random multi-choice rows from cot; falls back to all cot rows if needed.

## Skip Counts

```json
{}
```
