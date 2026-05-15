# MedReason-Align

面向中文医学考试推理的 LLM 后训练项目：基于 MedicalGPT 训练框架，对开源基座模型进行 SFT 与后续 RL-style alignment，提升模型在医学选择题、病例推理题中的答案准确率、推理格式稳定性与跨数据集泛化能力。

> 本项目用于医学考试/医学问答推理研究，不用于真实临床诊断、治疗决策或处方建议。

## 项目定位

MedReason-Align 关注的问题不是“做一个泛医疗聊天机器人”，而是：

> 通过高质量医学数据构造、CoT-style SFT 和多目标 RL 对齐，能否提升开源大模型在中文医学考试推理任务上的准确率、推理稳定性与外部 benchmark 泛化能力？

当前主线：

- **数据**：CMB/CME 风格中文医学考试题与医学问答数据
- **基座模型**：Qwen2.5-7B-Instruct 为主
- **训练框架**：MedicalGPT
- **训练路线**：Direct SFT -> CoT-style SFT -> GRPO/RL-style alignment
- **评测路线**：CMB-val 开发集 + CMExam/C-Eval 医学子集外部泛化评测

## 为什么做这个项目

通用大模型在医学题目上常见问题包括：

- 答案格式不稳定，难以自动评测
- 多选题、病例题容易漏选或混淆
- 推理过程与最终答案不一致
- 遇到不确定问题时表达过度自信
- 在同源数据上提升后，外部数据集泛化不一定稳定

因此本项目把重点放在“可验证的医学考试推理能力”上，而不是泛泛地做医疗问答演示。

## 方法路线

### 1. 数据清洗与候选池构建

对 CMB 训练集进行保守清洗：

- 过滤空题干、缺失答案、答案无法对应选项、正确选项为空等确定错误
- 对 `question + options` 做规范化精确去重
- 使用字符 n-gram TF-IDF 在同一考试来源桶内做近重复诊断与去重
- 保留短题干、非答案空选项等可能合法样本，并打标记而不是激进删除

当前清洗结果：

| 阶段 | 数量 |
|---|---:|
| CMB raw train | 269,359 |
| 基础清洗后 | 235,757 |
| 近重复去重后 | 233,188 |

### 2. SFT 数据策略消融

已构造并比较多种 Direct SFT 数据策略：

- 随机采样 baseline
- 质量过滤 quality-only
- 质量过滤 + 难度分层 optimized
- Direct-answer 输出格式统一：`答案：A`

核心观察：单样本质量更高不一定带来同源验证集更好效果，题型/难度分布偏移会影响评测表现；但 quality/optimized 在外部 benchmark 上可能更有泛化收益。

### 3. CoT 候选筛选与 teacher 精筛

为了避免把所有题都包装成长推理，本项目先筛选“值得写 CoT 的题”：

- 第一阶段：规则打分，优先多选题、病例信息丰富、选项混淆度高的题
- 第二阶段：DeepSeek teacher 评估 CoT-worthiness
- 只把适合显式推理的题进入后续 CoT-style SFT 子集

当前 DeepSeek 精筛结果：

| 项目 | 数量 |
|---|---:|
| 第一阶段 top candidates | 3,000 |
| DeepSeek keep | 1,158 |
| 最终 selected | 1,000 |

### 4. SFT 与 RL-style alignment

项目计划中的训练对照：

| 实验 | 输出格式 | 目的 |
|---|---|---|
| Base | 模型原生输出 | 基座医学题能力 |
| Direct SFT | `答案：A` | 学习答题格式与任务分布 |
| CoT SFT | `分析：...\n答案：A` | 学习显式医学题推理过程 |
| CoT + GRPO | `分析 + 答案` | 用多目标 reward 强化正确率、格式、推理结构和安全表达 |

GRPO/RL-style reward 设计方向：

- 答案正确率
- 答案格式合规
- CoT 结构完整
- CoT 与答案一致
- 过度自信惩罚
- 冗长/重复推理惩罚

## 初步实验结果

### CMB-val

CMB-val 样本较少，主要作为开发集和 sanity check。

| 模型 | 单选 Accuracy | 多选 Exact Match |
|---|---:|---:|
| Base Qwen2.5-7B-Instruct | 0.8042 | 0.4359 |
| Baseline LoRA | 0.9542 | 0.8462 |
| Optimized LoRA | 0.8000 | 0.6154 |

观察：

- Baseline SFT 在 CMB-val 上提升明显，说明 SFT 对 CMB 答题格式适配有效。
- Optimized 在 CMB-val 上不如 baseline，后续分布诊断显示其题型/难度分布与 val 偏移较大。

### CMExam-test 外部泛化

CMExam-test 用作外部泛化主评测之一。

| 模型 | 单选 Accuracy | 多选 Exact Match |
|---|---:|---:|
| Base Qwen2.5-7B-Instruct | 0.8295 | 0.5245 |
| Baseline LoRA | 0.8438 | 0.6324 |
| Quality-only LoRA | 0.8513 | 0.6373 |
| Optimized LoRA | 0.8521 | 0.6373 |

观察：

- SFT 后模型在 CMExam 单选、多选上均有提升。
- Quality-only/Optimized 在外部 benchmark 上优于随机 baseline，提示质量过滤可能更有助于泛化。
- 难度分层收益尚不明确，需要继续做更严格的消融。

## 代码结构

```text
MedReason-Align/
├── training/                         # MedicalGPT 训练入口
├── scripts/
│   ├── clean_cmb_candidates.py        # CMB 候选池基础清洗
│   ├── vector_dedup_cmb.py            # 保守近重复去重
│   ├── prepare_cmb_sft.py             # CMB -> SFT 格式转换
│   ├── build_direct_sft.py            # Direct-answer SFT 构造
│   ├── build_cot_candidates.py        # CoT 候选第一阶段规则筛选
│   ├── filter_cot_candidates_with_deepseek.py
│   ├── build_cmb_cot_with_deepseek.py # teacher rationale 生成
│   ├── build_cot_mixed_sft.py         # Direct + CoT 混合 SFT 构造
│   └── eval_medreason.py              # 结构化医学推理评测脚本
├── tasks/cmexam_choice/               # lm-evaluation-harness 任务配置
├── tests/                             # 数据处理与 teacher rewrite 单元测试
├── docs/
│   ├── project_roadmap.md
│   ├── project_progress.md
│   ├── schema_v1.md
│   ├── cot_candidate_rules.md
│   └── day10_teacher_rewrite_rules_v1.md
└── data/                              # 本地数据目录，公开仓库不提交大体量数据
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
pip install pytest
```

运行单元测试：

```bash
python -m pytest -q tests
```

核心脚本语法检查：

```bash
python -m py_compile \
  scripts/prepare_cmb_sft.py \
  scripts/clean_cmb_candidates.py \
  scripts/vector_dedup_cmb.py \
  scripts/build_cot_candidates.py \
  scripts/filter_cot_candidates_with_deepseek.py \
  scripts/build_cot_mixed_sft.py
```

## 数据准备

公开仓库不包含大体量原始数据、生成训练集和 teacher 输出。需要自行准备或生成：

- CMB train/val/test
- CMExam valid/test
- C-Eval 医学子集
- DeepSeek teacher audit/selected 输出

推荐本地目录：

```text
data/raw/hf/CMB/
data/raw/hf/CMExam/
data/processed/
data/sft/
data/teacher/
data/eval/
```

## 典型流程

### 1. 清洗 CMB 候选池

```bash
python scripts/clean_cmb_candidates.py \
  --input data/raw/hf/CMB/CMB-Exam/CMB-train/CMB-train-merge.json \
  --output data/processed/cmb_clean/cmb_train_clean.jsonl \
  --rejected-output data/processed/cmb_clean/cmb_train_rejected.jsonl
```

### 2. 近重复去重

```bash
python scripts/vector_dedup_cmb.py \
  --input data/processed/cmb_clean/cmb_train_clean.jsonl \
  --output data/processed/cmb_clean/cmb_train_vector_dedup_bucket.jsonl \
  --pairs-output data/processed/cmb_clean/cmb_train_vector_pairs_bucket.jsonl \
  --clusters-output data/processed/cmb_clean/cmb_train_vector_clusters_bucket.json
```

### 3. 构造 Direct SFT

```bash
python scripts/prepare_cmb_sft.py \
  --output data/sft/cmb_sft_baseline.jsonl \
  --max-samples 5000
```

### 4. 筛选 CoT 候选题

```bash
python scripts/build_cot_candidates.py \
  --input data/processed/cmb_clean/cmb_train_vector_dedup_bucket.jsonl \
  --output data/processed/cmb_clean/cmb_cot_candidates_scored.jsonl
```

### 5. DeepSeek teacher 精筛

```bash
export DEEPSEEK_API_KEY=...

python scripts/filter_cot_candidates_with_deepseek.py \
  --input data/processed/cmb_clean/cmb_cot_candidates_scored.jsonl \
  --candidate-top-k 3000 \
  --select-top-k 1000 \
  --call-api \
  --resume
```

### 6. SFT 训练

参考：

```bash
bash scripts/run_sft.sh
```

## 当前进度

- [x] CMB 候选池基础清洗
- [x] CMB 近重复去重
- [x] Direct SFT 数据构造
- [x] Baseline / Quality / Optimized SFT 初步训练与评测
- [x] CMExam-test 外部泛化初评
- [x] CoT 候选规则筛选
- [x] DeepSeek CoT-worthiness 精筛
- [ ] CoT-style SFT 主实验
- [ ] GRPO/RL-style 多目标对齐
- [ ] C-Eval 医学子集完整评测
- [ ] 失败案例与校准分析

## 面试版一句话

MedReason-Align 是一个面向中文医学考试推理的 LLM 后训练项目，基于 MedicalGPT 与 Qwen2.5-7B-Instruct，构建医学题数据清洗、质量过滤、CoT 候选筛选、SFT 消融和 RL-style alignment 流程，并通过 CMExam/C-Eval 等外部数据集验证模型医学推理能力的泛化提升。

## 致谢

本项目基于 [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT) 训练框架二次开发，感谢原项目提供的医疗大模型训练底座。
