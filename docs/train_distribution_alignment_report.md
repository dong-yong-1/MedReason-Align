# Train Distribution Alignment Report

## 1. 结论

- 当前 Direct 和 CoT/Mixed 是否严格公平对照：否。
- question+options 完全相同题目数量：160 / Direct 5000 / CoT-Mixed 5000。
- 按题目 hash 计，Direct-only：4644；CoT/Mixed-only：4820；共有：160。
- 按行计，Direct 中有共有题目的行数：170；CoT/Mixed 中有共有题目的行数：162。

如果题目集合不一致，当前结果不能严格归因于 CoT 输出格式，可能同时受到训练数据分布变化影响。

## 2. 样本级重叠

{
  "direct_total": 5000,
  "cot_mixed_total": 5000,
  "sample_id_shared_count": 121,
  "sample_id_overlap_rate_vs_direct": 0.0242,
  "sample_id_overlap_rate_vs_cot_mixed": 0.0242,
  "sample_id_direct_only_count": 4879,
  "sample_id_cot_mixed_only_count": 4879,
  "question_options_hash_shared_count": 160,
  "question_options_hash_overlap_rate_vs_direct": 0.033306,
  "question_options_hash_overlap_rate_vs_cot_mixed": 0.032129,
  "exact_same_question_count": 160,
  "question_options_hash_direct_only_count": 4644,
  "question_options_hash_cot_mixed_only_count": 4820,
  "question_options_hash_shared_count_unique": 160,
  "direct_rows_with_shared_question_count": 170,
  "cot_mixed_rows_with_shared_question_count": 162,
  "direct_rows_without_shared_question_count": 4830,
  "cot_mixed_rows_without_shared_question_count": 4838
}

## 3. Direct 分布

{
  "total": 5000,
  "single_count": 4759,
  "multi_count": 241,
  "multi_ratio": 0.0482,
  "gold_answer_len_distribution": {
    "1": 4760,
    "2": 51,
    "3": 66,
    "4": 78,
    "5": 45
  },
  "avg_gold_answer_len": 1.1194,
  "question_length": {
    "avg": 39.6704,
    "p50": 21,
    "p90": 100,
    "min": 3,
    "max": 489
  },
  "case_rich_ratio": 0.182,
  "option_confusing_ratio": 0.2382,
  "high_cot_worthiness_ratio": 0.0118,
  "exam_type_distribution": {
    "医师考试": 3198,
    "专业知识考试": 724,
    "医技考试": 433,
    "药师考试": 302,
    "护理考试": 251,
    "医学考研": 92
  },
  "exam_class_distribution": {
    "unknown": 5000
  },
  "exam_subject_top20": {
    "临床执业医师": 279,
    "内科主治医师": 179,
    "外科主治医师": 149,
    "儿科主治医师": 146,
    "执业西药师": 142,
    "临床执业助理医师": 141,
    "公共卫生执业医师": 136,
    "妇产科学副主任、主任医师职称考试": 101,
    "中西医结合执业医师": 94,
    "超声波医学主治医师": 92,
    "妇产科主治医师": 89,
    "全科主治医师": 87,
    "公共卫生执业助理医师": 82,
    "儿科学": 78,
    "放射科主治医师": 75,
    "主管护师资格考试": 71,
    "预防医学主治医师": 67,
    "护师资格考试": 66,
    "中医执业医师": 65,
    "中西医结合执业助理医师": 64
  },
  "difficulty_distribution": {
    "medium": 2500,
    "easy": 1564,
    "hard": 936
  },
  "quality_score": {
    "avg": 0.6733,
    "p50": 0.65,
    "p90": 0.95,
    "min": 0.5,
    "max": 0.95
  },
  "sft_format_distribution": {
    "direct_answer": 5000
  }
}

## 4. CoT/Mixed 分布

{
  "total": 5000,
  "single_count": 4084,
  "multi_count": 916,
  "multi_ratio": 0.1832,
  "gold_answer_len_distribution": {
    "1": 4084,
    "2": 240,
    "3": 266,
    "4": 272,
    "5": 117,
    "6": 21
  },
  "avg_gold_answer_len": 1.4322,
  "question_length": {
    "avg": 38.846,
    "p50": 22,
    "p90": 91,
    "min": 2,
    "max": 563
  },
  "case_rich_ratio": 0.2434,
  "option_confusing_ratio": 0.216,
  "high_cot_worthiness_ratio": 0.1722,
  "exam_type_distribution": {
    "医师考试": 3186,
    "专业知识考试": 818,
    "医技考试": 329,
    "药师考试": 260,
    "护理考试": 231,
    "医学考研": 176
  },
  "exam_class_distribution": {
    "unknown": 4014,
    "中级职称": 409,
    "高级职称": 127,
    "临床医学": 98,
    "规培结业": 91,
    "西医综合": 46,
    "基础医学": 43,
    "执业医师": 29,
    "主管技师": 25,
    "执业助理医师": 24,
    "中医学与中药学": 21,
    "中医综合": 20,
    "执业西药师": 19,
    "高级护师": 13,
    "预防医学与公共卫生学": 4,
    "护士执业资格": 4,
    "执业中药师": 3,
    "主管护师": 3,
    "护理学": 2,
    "初级药士": 1,
    "初级中药士": 1,
    "初级药师": 1,
    "医技师": 1,
    "医技士": 1
  },
  "exam_subject_top20": {
    "临床执业医师": 231,
    "内科主治医师": 214,
    "外科主治医师": 192,
    "儿科主治医师": 158,
    "临床执业助理医师": 129,
    "执业西药师": 106,
    "中西医结合执业医师": 103,
    "考研西医综合": 102,
    "公共卫生执业医师": 94,
    "儿科学": 93,
    "妇产科学副主任、主任医师职称考试": 86,
    "全科主治医师": 80,
    "中医执业医师": 78,
    "妇产科主治医师": 74,
    "中西医结合执业助理医师": 73,
    "心内科高级职称": 68,
    "放射科主治医师": 66,
    "中医执业助理医师": 66,
    "中医内科主治医师": 63,
    "超声波医学主治医师": 59
  },
  "difficulty_distribution": {
    "medium": 3922,
    "unknown": 986,
    "easy": 74,
    "hard": 18
  },
  "quality_score": {
    "avg": 0.653974,
    "p50": 0.6,
    "p90": 0.85,
    "min": 0.5,
    "max": 0.95
  },
  "sft_format_distribution": {
    "direct_answer": 4014,
    "cot_teacher_deepseek": 986
  }
}

## 5. 解释

若 Direct 与 CoT/Mixed 的题目集合、题型比例、多选比例、答案长度或难度分布不一致，则旧实验同时混入了数据分布变化和输出格式变化两个因素。
后续应使用 paired_direct_5000 与 paired_mixed_5000 做同题集合对照，再判断 CoT/Mixed 格式本身是否导致多选边界变化。
