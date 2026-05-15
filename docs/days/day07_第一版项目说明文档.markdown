# Day 7：第一版项目说明文档

项目名：
`MedReason-Align`

项目定位：
基于 `MedicalGPT` 构建一个面向结构化病例诊断推理的后训练项目。项目重点不是做泛医疗聊天，而是围绕病例输入，稳定输出诊断方向、判断依据、鉴别诊断、下一步建议和风险提示。

---

## 1. 这个项目做什么

本项目的主任务是：

> 结构化病例诊断推理（Clinical Case Reasoning）

输入是一段病例描述，可以来自：

- 患者或家属的自然语言描述
- 半结构化病历摘要
- 医学考试题干
- 症状、病程、既往史、用药史、体征、检查信息的组合

输出统一为 `schema v1` 的 5 个字段：

- `primary_diagnosis`
- `diagnostic_basis`
- `differential_diagnoses`
- `recommended_actions`
- `risk_flags`

第一版任务边界：

- 做病例分析和基础分诊辅助
- 不做泛医疗闲聊
- 不替代医生最终诊断
- 不输出强处方
- 对高风险病例必须给出升级就医提示

---

## 2. 为什么这样做

大模型算法实习项目最容易出现的问题是：

- 只复现训练脚本，没有任务闭环
- 只堆数据，没有解释为什么这些数据有用
- 只报一个分数，没有说明提升来自哪里
- 只做聊天效果，没有严格评测

所以本项目采用的核心思路是：

> 先定义目标病例分布，再构造目标对齐数据，然后通过 SFT 和 DPO 做后训练，最后用公共 benchmark 和自建 holdout 同时评测。

这让项目能回答三个关键问题：

- 模型到底要擅长什么任务？
- 数据为什么能推动这个任务？
- 训练之后是否真的在目标任务上变强？

---

## 3. 方法路线

第一版训练路线：

```text
Base Model
-> Baseline SFT
-> Target-aligned SFT
-> Target-aligned SFT + DPO
-> Evaluation & Error Analysis
```

第一版优先做：

- `SFT`
- `DPO`
- 数据构造
- 严格评测
- 错误分析

第一版暂不优先做：

- `PT`
- `PPO`
- `GRPO`
- 过大的多科室泛化
- 复杂在线诊疗系统

原因：
当前项目最核心的价值是建立“任务定义 -> 数据构造 -> 训练 -> 评测 -> 误差分析”的闭环，而不是先追求训练方法花样。

---

## 4. 数据设计

数据分为 5 层：

| 数据层 | 作用 | 是否参与训练 |
|---|---|---|
| 原始候选池 | 提供大规模原料 | 不一定 |
| target-dev-distribution set | 定义目标任务分布 | 否 |
| SFT 训练集 | 教模型按 schema 做病例推理 | 是 |
| DPO 偏好集 | 让模型偏向更安全、更完整的回答 | 是 |
| final holdout eval set | 最终盲评 | 否 |

第一版候选数据源：

- `FreedomIntelligence/HuatuoGPT-sft-data-v1`
- `shibing624/medical`
- `FreedomIntelligence/CMB`
- 少量 `GBaker/MedQA-USMLE-4-options-hf`

自建数据资产：

- `target-dev-distribution set`
- `final holdout eval set`

---

## 5. target-dev-distribution set

`target-dev-distribution set` 的作用不是训练，而是定义模型应该擅长的目标病例分布。

它用于：

- 指导候选数据筛选
- 指导 teacher 改写
- 指导训练数据配比
- 支撑分桶评测和错误分析

每条样本至少包含：

- `sample_id`
- `split`
- `case_text`
- `schema_target`
- `meta`

其中 `meta` 至少记录：

- `primary_system`
- `risk_level`
- `difficulty_level`
- `input_style`
- `completeness_level`
- `reasoning_type`
- `quality_tier`
- `review_status`

---

## 6. 数据清洗与质量控制

第一版数据清洗规则分为 5 组：

- `R-NORM-*`：标准化前处理
- `R-DUP-*`：去重规则
- `R-LEAK-*`：去泄漏规则
- `R-QUAL-*`：低质过滤规则
- `R-REVIEW-*`：人工复核规则

核心策略：

- 一级去重：`case_text + schema_target` 标准化后完全一致
- 二级去重：embedding 相似度高于阈值
- 三级去重：模板重复候选识别和人工复核
- 泄漏拦截：保护 `target_dev`、`final_holdout` 和公共 benchmark test split
- 低质过滤：剔除结构低质、医学内容低质、推理低质、表达低质和安全边界低质样本

每条样本最终状态统一为：

- `accepted`
- `rejected_duplicate`
- `rejected_leakage`
- `rejected_low_quality`
- `needs_manual_review`

---

## 7. 评测方案

第一版评测分为三层：

### 7.1 公共 benchmark

作用：

- 提供外部参照
- 防止只在自建数据上自我证明

建议使用：

- `CMB-Clin`
- `CMB-Exam`
- 可选 `MedQA` 或 `CMExam`

### 7.2 target_dev 开发评测

作用：

- 开发阶段快速定位问题
- 看模型在哪些桶上变强或变弱

按以下维度拆分：

- `primary_system`
- `risk_level`
- `difficulty_level`
- `input_style`
- `completeness_level`

### 7.3 final_holdout 最终评测

作用：

- 项目最终主报告指标
- 不参与训练、筛选、prompt 调试和 teacher 改写

---

## 8. 评测指标

内容指标：

- 主诊断命中
- 依据完整性
- 鉴别诊断覆盖
- 下一步建议可用性
- 风险提示质量

结构化指标：

- `JSON Valid Rate`
- `Schema Match Rate`
- `Field Completion Rate`

安全指标：

- `Risk Recall`
- `Overconfidence Rate`
- `Unsafe Recommendation Rate`

第一版不只报总分，必须同时报告分桶结果和失败案例。

---

## 9. 预期项目产出

第一版项目完成后，应该能交付：

- 一套病例推理 schema
- 一套目标分布定义
- 一套数据清洗规则
- 一批目标对齐 SFT 数据
- 一批 DPO 偏好数据
- 至少三个模型版本对比
- 公共 benchmark 结果
- 自建 holdout 结果
- 错误分析报告
- before / after 案例展示

---

## 10. Week 1 已完成内容

Week 1 的目标是定任务、定数据、定评测。

目前已经完成：

- 项目定义 v1
- 数据源清单和质量标准
- `schema v1`
- `target-dev-distribution set` 字段标准
- 评测方案草案
- 数据清洗规则 v1
- 第一版项目说明文档

仍需继续落地：

- `target_dev` 的正式配比表
- 第一批 10 条 seed case
- `final_holdout` 的实际样本构建

---

## 11. Week 2 下一步

Week 2 进入数据和 baseline 阶段。

建议顺序：

1. 先做 10 条 seed case，校准样本质量
2. 再做 `target_dev` 的 120 条配比表
3. 整理原始候选池
4. 实现标准化、去重、低质过滤和泄漏检查脚本
5. 构造第一版 SFT 数据
6. 跑通 baseline SFT
7. 做第一次快速评测和错误分析

---

## 12. 一句话总结

`MedReason-Align` 的核心不是复现 `MedicalGPT`，而是把 `MedicalGPT` 包装成一个有明确任务、有数据治理、有训练闭环、有评测证据的结构化医疗病例推理项目。
