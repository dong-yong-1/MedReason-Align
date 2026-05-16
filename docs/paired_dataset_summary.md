# Paired Dataset Summary

## paired_direct_5000

- output_dir: `data/sft/paired_direct_5000`
- train_file: `data/sft/paired_direct_5000/paired_direct_5000.jsonl`

```json
{
  "total": 5000,
  "paired_format_counts": {
    "direct": 5000
  },
  "original_sft_format_counts": {
    "direct_answer": 4014,
    "cot_teacher_deepseek": 986
  },
  "question_type_counts": {
    "单项选择题": 4083,
    "多项选择题": 916,
    "C型选择题": 1
  },
  "answer_len_distribution": {
    "1": 4084,
    "2": 240,
    "3": 266,
    "4": 272,
    "5": 117,
    "6": 21
  }
}
```

## paired_mixed_5000

- output_dir: `data/sft/paired_mixed_5000`
- train_file: `data/sft/paired_mixed_5000/paired_mixed_5000.jsonl`

```json
{
  "total": 5000,
  "paired_format_counts": {
    "direct": 4014,
    "cot": 986
  },
  "original_sft_format_counts": {
    "direct_answer": 4014,
    "cot_teacher_deepseek": 986
  },
  "question_type_counts": {
    "单项选择题": 4083,
    "多项选择题": 916,
    "C型选择题": 1
  },
  "answer_len_distribution": {
    "1": 4084,
    "2": 240,
    "3": 266,
    "4": 272,
    "5": 117,
    "6": 21
  }
}
```

## paired_cot_986_direct

- output_dir: `data/sft/paired_cot_986_direct`
- train_file: `data/sft/paired_cot_986_direct/paired_cot_986_direct.jsonl`

```json
{
  "total": 986,
  "paired_format_counts": {
    "direct": 986
  },
  "original_sft_format_counts": {
    "cot_teacher_deepseek": 986
  },
  "question_type_counts": {
    "多项选择题": 678,
    "单项选择题": 308
  },
  "answer_len_distribution": {
    "1": 308,
    "2": 196,
    "3": 190,
    "4": 189,
    "5": 82,
    "6": 21
  }
}
```

## paired_cot_986_cot

- output_dir: `data/sft/paired_cot_986_cot`
- train_file: `data/sft/paired_cot_986_cot/paired_cot_986_cot.jsonl`

```json
{
  "total": 986,
  "paired_format_counts": {
    "cot": 986
  },
  "original_sft_format_counts": {
    "cot_teacher_deepseek": 986
  },
  "question_type_counts": {
    "多项选择题": 678,
    "单项选择题": 308
  },
  "answer_len_distribution": {
    "1": 308,
    "2": 196,
    "3": 190,
    "4": 189,
    "5": 82,
    "6": 21
  }
}
```
