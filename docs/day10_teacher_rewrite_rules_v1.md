# Day 10 Teacher Rewrite Rules v1

目标：把 Day 9 选出的候选样本改写成 `schema_v1` 的 SFT 样本。

这一步可以使用 DeepSeek V4 API，但规则上要保持保守：宁可拒绝样本，也不要为了凑字段编造病例事实。

## 1. 输入来源

Day 10 默认读取：

- `data/medreason/day9_selected_for_rewrite_sample50k_v1.jsonl`

候选样本包含三类训练候选：

- `structured_block_candidate`
- `exam_rewrite_candidate`
- `rewrite_candidate`

`benchmark_only` 不进入 Day 10。

## 2. route 处理策略

### structured_block_candidate

来源：`CMB-Clin`

处理方式：

- 优先保留
- 使用 `case_text` 作为病例输入
- 使用 `raw_blocks.diagnosis`、`raw_blocks.differential`、`raw_blocks.treatment_or_action` 作为依据材料
- teacher 只负责整理为 schema，不应新增病例事实

### exam_rewrite_candidate

来源：`CMB-Exam/train`

处理方式：

- 只接受病例题或临床场景题
- 如果题干只是纯知识问答，应该 reject
- 输出中不能保留 A/B/C/D 选项格式
- 可以把正确选项对应的疾病或处理方向整理为诊断方向，但不能把题目改成“考试解析”

### rewrite_candidate

来源：`HuatuoGPT-sft-data-v1` 或 `shibing624/medical`

处理方式：

- 只有当原始问题包含足够病例信息时才 accept
- 如果只是“某病是什么”“某药怎么吃”“某病简介”，应该 reject
- 如果病例信息不足以支持至少 2 条诊断依据和 1 条鉴别诊断，应该 reject

## 3. teacher 输出格式

teacher 必须只输出合法 JSON 对象。

接受样本：

```json
{
  "decision": "accept",
  "reject_reason": "",
  "case_text": "...",
  "schema_target": {
    "primary_diagnosis": "...",
    "diagnostic_basis": ["...", "..."],
    "differential_diagnoses": ["..."],
    "recommended_actions": ["..."],
    "risk_flags": ["..."]
  },
  "quality_tags": ["case_like", "schema_complete"]
}
```

拒绝样本：

```json
{
  "decision": "reject",
  "reject_reason": "病例信息不足，无法支持结构化诊断推理",
  "case_text": "",
  "schema_target": null,
  "quality_tags": ["insufficient_case_context"]
}
```

## 4. accept 硬性规则

接受样本必须满足：

- `case_text` 是病例输入，不是纯知识题或药品说明
- `schema_target.primary_diagnosis` 是非空字符串
- `diagnostic_basis` 至少 2 条
- `differential_diagnoses` 至少 1 条
- `recommended_actions` 至少 1 条
- `risk_flags` 至少 1 条
- 输出必须是合法 JSON

## 5. 禁止行为

teacher 不得：

- 编造原文没有的年龄、性别、检查结果、既往史
- 把不完整病例写成“确诊”
- 输出具体强处方或危险剂量建议
- 只写“注意观察”作为风险提示
- 把考试选项格式保留进最终 `case_text`
- 把 benchmark 数据改写进训练集

## 6. 语言风格

- 诊断表达使用“可能性大”“首先考虑”“倾向于”等稳健措辞
- 建议以检查、就诊、进一步评估、风险升级为主
- 风险提示必须说明何时需要急诊或线下就医

## 7. pilot 配比

第一轮 pilot 默认生成 114 条输入：

- `structured_block_candidate`: 74 条
- `exam_rewrite_candidate`: 20 条
- `rewrite_candidate`: 20 条

pilot 的目标是验证 prompt、API、JSON 合法率、accept rate 和 hallucination 风险，不是追求规模。
