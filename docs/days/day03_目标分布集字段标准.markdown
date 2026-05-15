# Day 3：target-dev-distribution set 字段标准

目标：
把 `target-dev-distribution set` 从“抽象配比”落成“每条样本都怎么记录”的统一标准。

注意：
这份标准不是替代 `schema v1`，而是给每条样本加一层数据管理字段，用来控分布、筛数据、做误差分析。

---

## 1. 这份标准解决什么问题

如果没有统一字段，后面会出现 4 个问题：

- 不知道一条样本该不该收
- 不知道一条样本属于哪个分布桶
- 不知道模型在哪类病例上提升
- 不知道错误到底来自诊断、风险判断，还是输入风格

所以 `target-dev-distribution set` 的字段标准，本质上是项目的数据管理协议。

---

## 2. 总体结构

每条样本统一使用 3 层结构：

```json
{
  "sample_id": "dev_cardio_001",
  "split": "target_dev",
  "case_text": "患者，男，61岁。主诉：胸闷胸痛2小时。活动后胸骨后压榨样疼痛，可放射至左肩，伴出汗、恶心。既往高血压10年，吸烟30年。",
  "schema_target": {
    "primary_diagnosis": "急性冠脉综合征可能性大",
    "diagnostic_basis": [
      "活动后胸骨后压榨样疼痛，符合缺血性胸痛特征",
      "疼痛放射至左肩，伴出汗、恶心，支持急性冠脉事件可能",
      "存在高血压和长期吸烟等心血管危险因素"
    ],
    "differential_diagnoses": [
      "主动脉夹层",
      "肺栓塞",
      "胃食管反流病"
    ],
    "recommended_actions": [
      "建议立即就医，尽快完成心电图和心肌损伤标志物检查",
      "必要时进一步心内科急诊评估"
    ],
    "risk_flags": [
      "若胸痛持续不缓解、出现呼吸困难或意识异常，应立即急诊处理"
    ]
  },
  "meta": {
    "primary_system": "cardiovascular",
    "secondary_systems": ["respiratory"],
    "risk_level": "emergent",
    "difficulty_level": "common_confusable",
    "input_style": "semi_structured_case",
    "completeness_level": "mostly_complete",
    "reasoning_type": ["diagnosis", "differential", "triage"],
    "age_group": "older_adult",
    "sex": "male",
    "visit_setting": "ed_like",
    "gold_source_type": "manual_rewrite_from_public_case",
    "quality_tier": "gold",
    "review_status": "reviewed",
    "annotator": "dongyong_v1",
    "version": "v1.0"
  }
}
```

三层职责如下：

- 顶层字段：样本身份与分区信息
- `schema_target`：模型真正要学会输出的目标答案
- `meta`：分布控制、质量控制和误差分析标签

---

## 3. 顶层必备字段

### 3.1 `sample_id`

作用：

- 唯一标识一条样本
- 方便复查、回滚、统计和错误定位

建议命名：

- `dev_cardio_001`
- `dev_resp_014`
- `dev_mixed_023`

要求：

- 全局唯一
- 尽量包含主方向信息

### 3.2 `split`

可选值：

- `target_dev`
- `holdout_eval`
- `train_candidate`

作用：

- 明确样本属于开发分布集、最终评测集还是训练候选池
- 为后续统一脚本处理预留接口

### 3.3 `case_text`

作用：

- 模型输入文本

要求：

- 尽量保留真实病例表述
- 不提前写入结论
- 可以是患者自然描述、半结构化病历摘要、题干式病例

### 3.4 `schema_target`

作用：

- 使用 `schema v1` 记录标准答案

内部字段固定为：

- `primary_diagnosis`
- `diagnostic_basis`
- `differential_diagnoses`
- `recommended_actions`
- `risk_flags`

---

## 4. meta 核心字段

以下字段为第一版必须标注的核心标签。

### 4.1 `primary_system`

定义：

- 这条病例的主导系统方向

固定枚举：

```json
[
  "respiratory",
  "cardiovascular",
  "gastrointestinal",
  "neurology",
  "infection",
  "endocrine_metabolic",
  "renal_urologic",
  "hematology_rheumatology",
  "drug_related",
  "general_internal_medicine"
]
```

要求：

- 每条样本只能有一个主方向
- 即使是混合病例，也必须指定一个主导系统

### 4.2 `risk_level`

定义：

- 病例当前风险等级

固定枚举：

```json
["routine", "urgent", "emergent"]
```

判断标准：

- `routine`：常规门诊型，可短期随访
- `urgent`：需要尽快线下就医或当天处理
- `emergent`：有明确红旗，需要急诊或立即升级处理

### 4.3 `difficulty_level`

定义：

- 病例推理难度标签

固定枚举：

```json
[
  "typical_single",
  "common_confusable",
  "atypical_presentation",
  "noisy_or_incomplete"
]
```

判断标准：

- `typical_single`：典型表现，首要诊断相对清晰
- `common_confusable`：常见混淆场景，鉴别诊断重要
- `atypical_presentation`：表现不典型，容易误导
- `noisy_or_incomplete`：信息缺失、噪声多或合并症干扰明显

### 4.4 `input_style`

定义：

- 输入文本的风格来源

固定枚举：

```json
[
  "patient_narrative",
  "semi_structured_case",
  "exam_case"
]
```

判断标准：

- `patient_narrative`：患者或家属自然描述
- `semi_structured_case`：病史、体征、检查等半结构化摘要
- `exam_case`：医学考试题干或标准化病例题

### 4.5 `completeness_level`

定义：

- 病例关键信息完整程度

固定枚举：

```json
[
  "complete",
  "mostly_complete",
  "partially_missing",
  "critically_missing"
]
```

判断标准：

- `complete`：关键信息完整，可较稳定推理
- `mostly_complete`：有轻度缺失，但不影响主判断
- `partially_missing`：缺部分关键项，需要更保守表述
- `critically_missing`：关键信息明显不足，只能给方向性判断

### 4.6 `reasoning_type`

定义：

- 这条样本重点考查的推理能力，可多选

建议枚举：

```json
[
  "diagnosis",
  "differential",
  "triage",
  "test_recommendation",
  "risk_detection",
  "comorbidity_reasoning",
  "medication_related_reasoning"
]
```

说明：

- 一条病例往往不止一种推理任务
- 建议使用数组

### 4.7 `quality_tier`

定义：

- 样本质量等级

固定枚举：

```json
["gold", "silver", "bronze"]
```

判断标准：

- `gold`：人工精修，字段完整，适合 dev / eval
- `silver`：主要字段可靠，适合 train / dev
- `bronze`：弱标注或自动转换，只适合候选池

约束：

- `target_dev` 中样本原则上至少为 `silver`
- 核心分析样本优先使用 `gold`

### 4.8 `review_status`

定义：

- 当前审核状态

固定枚举：

```json
["draft", "reviewed", "locked"]
```

判断标准：

- `draft`：初稿，尚未复查
- `reviewed`：人工检查过一轮
- `locked`：作为稳定基准样本锁定

---

## 5. meta 增强字段

以下字段为增强信息，建议尽量保留，但第一批样本可后补。

### 5.1 `secondary_systems`

作用：

- 标记次要相关系统
- 适合混合病例

类型：

- `string[]`

### 5.2 `age_group`

建议枚举：

- `adolescent`
- `young_adult`
- `middle_adult`
- `older_adult`
- `elderly`

作用：

- 辅助分析不同年龄段病例表现

### 5.3 `sex`

建议枚举：

- `male`
- `female`
- `unknown`

### 5.4 `visit_setting`

建议枚举：

- `outpatient_like`
- `ed_like`
- `inpatient_like`
- `unknown`

作用：

- 区分更像门诊、急诊还是住院病例

### 5.5 `gold_source_type`

建议枚举：

- `manual_original`
- `manual_rewrite_from_public_case`
- `converted_from_mcq`
- `converted_from_dialogue`
- `synthetic_clinical_case`

作用：

- 追踪 gold 的来源方式
- 方便后续做数据质量审查

### 5.6 `annotator`

作用：

- 记录谁构造或修订了样本

### 5.7 `version`

作用：

- 标记样本版本，方便后续重写与回滚

---

## 6. 第一版必填字段与可后补字段

### 必填字段

- `sample_id`
- `split`
- `case_text`
- `schema_target`
- `meta.primary_system`
- `meta.risk_level`
- `meta.difficulty_level`
- `meta.input_style`
- `meta.completeness_level`
- `meta.reasoning_type`
- `meta.quality_tier`
- `meta.review_status`

### 可后补字段

- `meta.secondary_systems`
- `meta.age_group`
- `meta.sex`
- `meta.visit_setting`
- `meta.gold_source_type`
- `meta.annotator`
- `meta.version`

---

## 7. 使用规则

这套字段标准后续主要用于 4 件事：

### 7.1 控分布

- 统计 `primary_system` 是否覆盖均衡
- 统计 `risk_level` 是否包含足够红旗病例
- 统计 `input_style` 是否被考试题分布绑架

### 7.2 收样本

- 每收一条样本，先判断能否映射到 `schema_target`
- 再给它打核心标签
- 不能稳定打标签的样本，不进入核心 dev 集

### 7.3 做误差分析

后续评测不只看总分，还要按这些桶拆开：

- 哪个系统方向差
- 哪个风险等级差
- 哪种输入风格差
- 哪种难度最容易出错

### 7.4 支撑实验闭环

- 数据构造按标签补桶
- 训练后按标签看提升
- 错误分析按标签定位问题

---

## 8. 第一版结论

今天正式定下：

- `schema v1` 负责定义模型输出格式
- `target-dev-distribution set` 负责定义样本管理和分布控制格式
- 所有后续 seed 样本、dev 样本、holdout 样本，尽量都兼容这一套结构

一句话总结：

> 这套字段标准不是为了“把数据记详细”，而是为了让项目后面真的能做分布控制、评测拆解和误差分析。
