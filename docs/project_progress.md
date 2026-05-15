# 项目进度交接文档

> 最后更新时间：2026-05-13
> 当前阶段：Direct SFT 消融完成，进入 clean pool 重构、CoT-style SFT 构造与 GRPO 设计阶段

---

## 一、项目基本信息

### 1.1 项目名称
**MedReason-Align** —— 面向中文医学考试推理的 SFT + CoT + GRPO 多目标后训练实验

### 1.2 项目定位
展示端到端 LLM 后训练能力：数据处理 → SFT → CoT 格式学习 → GRPO 多目标对齐 → 跨 benchmark 评测 → 失败分析。

项目不定位为可直接用于临床诊断的医疗模型，而是定位为 **中文医学考试推理场景下的可控后训练实验**。选择医学考试题的原因是：答案可量化、推理过程可观察、适合比较不同数据策略和 reward shaping 对模型行为的影响。

### 1.3 核心技术栈
- 基座模型：Qwen2.5-7B-Instruct（LoRA 微调）
- 训练框架：MedicalGPT
- 评测：CMB-val 开发集 + CMExam-test 外部泛化评测 + C-Eval 医学子集补充
- 方法：SFT → CoT SFT → GRPO

### 1.4 服务器信息
- 模型路径：`/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct`
- 缓存路径：`/root/autodl-tmp/hf-cache`
- 显存：单卡 24GB 或双卡 48GB
- 关键环境变量：
  ```bash
  export HF_HOME=/root/autodl-tmp/hf-cache
  export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  ```

---

## 二、项目目标（面试版本）

### 核心研究问题
> "我想研究 SFT、CoT 和 GRPO 多维度 reward shaping 如何影响中文医学考试模型的答题准确性、推理格式稳定性和安全表达。"

### 项目背景
通用大模型在医疗问答中容易出现两类问题：
- **格式不可控**：答案、解释、选项混杂，难以稳定评测和部署。
- **过度自信**：在不确定时仍使用"一定"、"肯定"等绝对化表达。

医学考试题提供了可控实验环境：标准答案明确，可用准确率量化；同时可以要求模型输出 `分析 + 答案`，观察 CoT 格式和推理行为。SFT 先教模型"怎么答"，GRPO 再通过多维 reward 教模型"偏好什么样的答法"。

### 三层训练目标

1. **SFT：任务格式和基础答题能力**
   - 使用 CMB 构造 SFT 数据。
   - 比较随机采样、质量过滤、质量 + 难度分层三种数据策略。
   - 目标是让模型稳定输出医学考试题答案格式。

2. **CoT：推理过程显式化**
   - 已确认当前训练集主要是 Direct-answer 标签：`答案：A`。
   - CMB-train 无人工 explanation，不能直接声称已有专家 CoT。
   - 后续通过 teacher model 构造 CoT-style 子集，比较 direct answer 与 CoT answer：
     - Direct：`答案：A`
     - CoT：`分析：...\n\n答案：A`
   - 评测时通过答案抽取计算准确率，同时观察格式合规率和推理长度。

3. **GRPO：多目标 reward shaping**
   - 在 SFT checkpoint 基础上继续后训练。
   - Reward 不只包含答案正确，还包含格式、CoT 结构、过度自信惩罚和保守表达奖励。
   - 目标是把项目从"刷准确率"提升为"多目标医学推理对齐实验"。

### 面试能讲的闭环逻辑
1. **数据处理**：分布对齐 + 质量过滤 + 难度分层 —— 每个决策都能回答"为什么这样做"
2. **训练设计**：SFT 教格式，CoT 教显式推理，GRPO 教偏好 —— 三者分工清楚
3. **评测设计**：CMB-val 作为开发集，CMExam-test 作为外部泛化，C-Eval 医学子集作为补充
4. **失败分析**：optimized 数据质量更高但效果更差，说明分布保持比硬过滤更重要
5. **安全表达**：用过度自信率、不确定性表达和格式合规率作为医疗安全表达的 proxy 指标

---

## 三、已完成的工作

### 3.1 数据准备

#### CMB-Exam 数据概况
| 数据集 | 数量 | 用途 |
|---|---|---|
| train | 269,359 | SFT 训练候选池 |
| val | 280 | SFT 验证集 |
| test | 11,200 | 最终评测（无答案） |

字段：`question`、`answer`、`options`、`question_type`、`exam_type`、`exam_class`、`exam_subject`
**注意**：训练集**没有** `explanation` 字段

#### 已生成的 SFT 数据（三份）

| 文件路径 | 内容 | 平均质量分 | 说明 |
|---|---|---|---|
| `data/sft/cmb_sft_baseline.jsonl` | 随机采样 5k | 0.542 | 原始数据，无处理 |
| `data/sft/cmb_sft_quality_only.jsonl` | 质量过滤（≥0.5）5k | 0.652 | 只做质量过滤 |
| `data/sft/cmb_sft_optimized.jsonl` | 质量过滤 + 难度分层 5k | 0.673 | 质量过滤 + 难度对齐 |

每个文件配套有 `_val.jsonl` 验证集（280条）。

#### 质量评分维度（满分1.0）
- 题目长度（30-200字最优）：0.20
- 选项数量（5个最优）：0.20
- 选项内容质量（长度5-30字）：0.20
- 医学术语密度（≥4个术语）：0.20
- 考试类型（医师考试类）：0.15
- 多选题加分：0.05

#### 难度估算（只看题目文本，不用答案）
- 含"鉴别诊断"、"不属于"、"不包括"等关键词 → hard
- 题目超长、医学术语密度高、多选题 → 难度加分
- 总分≥3 → hard；0 → easy；其余 → medium

#### 分布对齐
- 脚本支持但**未执行**（需要 sentence-transformers 库，服务器可能没有）
- 逻辑：用多语言 embedding 模型对齐训练集和测试集的分布
- 激活方式：`--distribution-align --topk 5 --min-similarity 0.3`

#### 外部评测集准备

为避免只在 CMB-val 这个 280 条开发集上得出过强结论，已开始扩展外部 benchmark。

| Benchmark | 本地状态 | 用途 | 备注 |
|---|---|---|---|
| CMB-val | 已评测 | 开发集 / sanity check | 280 条，样本偏小，不作为最终主结论 |
| CMB-test | 本地文件无答案 | 暂不作为本地 accuracy | 需要补官方答案或提交官方评测 |
| CMExam | 已拉取并检查 | 外部泛化主评测 | test 6,811 条，有答案 |
| C-Eval 医学子集 | 已检查并部分转 JSONL | 中文医学补充评测 | test 合计约 1,028 条，规模小于 CMExam |

CMExam 本地文件：
- `data/raw/hf/CMExam/train.json`：54,497 条
- `data/raw/hf/CMExam/valid.json`：6,811 条
- `data/raw/hf/CMExam/test.json`：6,811 条

CMExam 字段：`Question`、`Options`、`Answer`、`Explanation`。虽然扩展名是 `.json`，实际是一行一个 JSON 的 JSONL 格式。

已生成 lm_eval 文件：
- `data/eval/lm_eval/cmexam_test_single.jsonl`：6,606 条
- `data/eval/lm_eval/cmexam_test_multi.jsonl`：204 条
- `data/eval/lm_eval/cmexam_valid_single.jsonl`：6,600 条
- `data/eval/lm_eval/cmexam_valid_multi.jsonl`：210 条

已生成 lm_eval task：
- `tasks/cmexam_choice/cmexam_test_single.yaml`
- `tasks/cmexam_choice/cmexam_test_multi.yaml`

评测定位：
- **CMB-val**：开发集，用来快速看趋势和调试。
- **CMExam-test**：主外部评测，用来验证模型是否只是适配 CMB 格式。
- **C-Eval 医学子集**：补充中文医学科目，不作为唯一主指标。

#### Direct SFT 数据构造（2026-05-07）

目标：
- 构造只输出最终答案的 Direct SFT 数据，用于和 explanation-style / CoT-style 输出做对比。
- 控制变量：保留相同题目、选项、答案、题型、难度、质量分等元信息，只改变 assistant 输出格式。

重要发现：
- 三套 CMB train 文件本身已经基本是 Direct 格式，assistant 为 `答案：X`。
- 三套 CMB val 文件包含 `分析：...\n\n答案：X`，属于 explanation-style 输出。
- 因此 Direct 数据构造的主要作用是统一 train/val 口径，尤其把 val 的 explanation-style 标签改为 direct-answer 标签。

处理脚本：
- `scripts/build_direct_sft.py`

处理逻辑：
1. 读取原始 SFT JSONL。
2. 保留 `human` prompt 不变。
3. 保留 `question`、`options`、`answer`、`question_type`、`exam_type`、`exam_subject`、`difficulty`、`quality_score` 等元信息不变。
4. 将最后一轮 assistant 输出统一替换为：
   ```text
   答案：{answer}
   ```
5. 按每题实际 `options` key 解析答案，支持 `A-E` 和少量 `F` 选项题。
6. 如果 `answer` 字段无法对应选项 key，例如原始数据中混入解释文本，则跳过该样本，避免污染 Direct SFT。
7. 增加 `sft_format: direct_answer` 字段，方便后续追踪数据版本。

已生成数据：

| 数据集 | train 数量 | val 数量 | 备注 |
|---|---:|---:|---|
| `data/sft/cmb_baseline_direct/` | 4,996 | 280 | 跳过 4 条脏答案文本 |
| `data/sft/cmb_quality_direct/` | 5,000 | 280 | 无跳过 |
| `data/sft/cmb_optimized_direct/` | 5,000 | 280 | 无跳过 |

面试表述：
> "为了验证显式推理是否真的有帮助，我构造了 Direct SFT 数据作为对照。这个转换只改变 assistant 输出，把 `分析 + 答案` 统一成 `答案：X`，题目、选项、标签和数据来源全部保持不变。因此 Direct vs CoT 的差异主要来自输出监督格式，而不是样本分布变化。"

#### CoT-style SFT 数据构造（2026-05-12）

背景修正：
- 已确认此前训练的 `cmb_baseline`、`cmb_quality`、`cmb_optimized`、`cmb_quality_direct` 都是 Direct-answer SFT，训练标签为 `答案：X`。
- CMB-train 原始数据没有人工 `explanation` 字段，因此不能声称已有专家 CoT。
- CoT-style 数据需要额外构造，作为 Direct-answer SFT 的对照实验。

Teacher 配置：
- 使用 DeepSeek teacher model。
- `.env` 中配置：
  - `DEEPSEEK_API_KEY`
  - `DEEPSEEK_BASE_URL`
  - `DEEPSEEK_MODEL`
- 脚本只读取环境变量，不在代码或日志中写入 API key。

处理脚本：
- `scripts/build_cmb_cot_with_deepseek.py`

构造原则：
1. 输入仍使用 CMB SFT JSONL，例如 `data/sft/cmb_quality_direct/cmb_sft_quality_only_direct.jsonl`。
2. 保留题目、选项、标准答案、题型、难度、质量分等元信息不变。
3. 将 `question`、`options`、`question_type` 和 gold `answer` 发给 DeepSeek。
4. DeepSeek 只负责生成 `analysis`，不得修改标准答案。
5. Teacher 输出必须是 JSON：
   ```json
   {
     "analysis": "简洁中文解析，不要包含最终答案标签",
     "answer": "C"
   }
   ```
6. 脚本校验 teacher answer 是否等于 gold answer；若答案被改、缺少 analysis 或格式非法，则标记为 invalid，不写入最终 SFT。
7. 最终 assistant 标签统一为：
   ```text
   分析：{teacher_analysis}

   答案：{gold_answer}
   ```
8. 生成过程支持 `--limit`、`--resume`、`--teacher-output`，可断点续跑，并保留 teacher 原始输出用于审计。

Pilot 结果：
- 已对 `cmb_quality_direct` 前 3 条样本调用 DeepSeek。
- 3/3 样本通过校验。
- 输出文件：
  - `data/teacher/cmb_quality_cot_pilot_teacher.jsonl`
  - `data/sft/cmb_quality_cot_pilot/cmb_sft_quality_cot_pilot.jsonl`

面试表述：
> "由于 CMB-train 没有人工 explanation，我没有把原始训练集包装成 CoT。后续我使用 DeepSeek 作为 teacher，在固定标准答案不变的前提下生成解析。脚本会校验 teacher 是否改答案，只有答案一致且解析合法的样本才进入 CoT-style SFT 数据。这样可以把 Direct-answer SFT 和 Teacher-rationale CoT SFT 做严格对照。"

#### 重复率诊断（2026-05-13）

背景：
- 第一版数据处理做了质量过滤和难度分层，但没有做去重。
- 为判断训练集是否存在重复题、以及 CMB-val 是否因为重复题导致评测波动，新增重复率诊断脚本。

诊断脚本：
- `scripts/diagnose_sft_duplicates.py`

诊断方法：
1. `raw_exact`：使用原始 `question + options` JSON 签名做完全重复检测。
2. `normalized_exact`：对题目和选项做 NFKC 归一化、去空白、去标点后再做 hash。
3. `question_only`：只看规范化后的题干文本，用于发现选项轻微变化的重复题；该指标可能高估重复。
4. 额外统计 train-val overlap，检查训练集是否和验证集存在题面重合。

输出文件：
- `data/diagnostics/sft_duplicate_report.md`
- `data/diagnostics/sft_duplicate_report.json`

核心结果：

| Dataset | Rows | Raw dup rows | Norm dup rows | Question-only dup rows |
|---|---:|---:|---:|---:|
| baseline_train | 5000 | 14 (0.28%) | 19 (0.38%) | 25 (0.50%) |
| quality_train | 5000 | 11 (0.22%) | 12 (0.24%) | 19 (0.38%) |
| optimized_train | 5000 | 170 (3.40%) | 202 (4.04%) | 250 (5.00%) |
| val | 280 | 41 (14.64%) | 41 (14.64%) | 45 (16.07%) |

Train-val overlap：

| Pair | Norm overlap | Question-only overlap |
|---|---:|---:|
| baseline_train -> val | 9 (0.18%) | 9 (0.18%) |
| quality_train -> val | 5 (0.10%) | 5 (0.10%) |
| optimized_train -> val | 4 (0.08%) | 4 (0.08%) |

关键结论：
- `quality_train` 的重复率最低，说明质量过滤没有引入明显重复问题。
- `optimized_train` 的重复率明显升高，规范化重复率达到 4.04%，说明当前难度分层采样可能在某些题型/模板上过度集中。
- CMB-val 自身重复率较高（约 14.64%），进一步说明它只适合作为开发集，不适合作为最终主结论。
- train-val overlap 很低，当前评测高分不主要来自完全重复的 train-val 泄露。

后续修正方向：
- v2 数据策略加入 exact dedup：先对 CMB 候选池做规范化 `question + options` hash 去重。
- 再做保守向量去重：使用字符 n-gram TF-IDF + cosine similarity 检测近重复题，并按考试来源字段分桶，避免跨科目误删。
- 对 optimized 采样策略增加"每个 normalized question hash 只保留一条"的约束，避免分层采样放大重复模板。

#### CMB 候选池基础清洗（2026-05-13）

目标：
- 在 SFT 采样前构建 clean candidate pool，避免后续质量过滤/分层采样放大脏样本或重复模板。

清洗脚本：
- `scripts/clean_cmb_candidates.py`

输入输出：
- 输入：`data/raw/hf/CMB/CMB-Exam/CMB-train/CMB-train-merge.json`
- clean pool：`data/processed/cmb_clean/cmb_train_clean.jsonl`
- rejected pool：`data/processed/cmb_clean/cmb_train_rejected.jsonl`
- summary：`data/processed/cmb_clean/cmb_clean_summary.md`

保守清洗规则：
1. 过滤 `question` 为空。
2. 过滤 `answer` 为空。
3. 过滤 `option` 不是 dict 或选项数少于 3。
4. 过滤 `answer` 无法对应 `option` key 的样本，例如 answer 混入解释文本。
5. 过滤正确答案对应选项为空的样本。
6. 过滤明显乱码题干。
7. 过滤规范化 `question + options` 完全重复样本，仅保留第一条。

暂不激进删除：
- 非答案选项为空：保留并打 `empty_non_answer_option_retained` 标记。
- 短题干：保留并打 `short_question_retained` 标记，因为医学考试中存在合法短定义题，如“MAC是”“癌指的是”。
- question-only 重复：只诊断不删除，因为可能存在题干相同但选项不同的合法题。

清洗结果：

| 项目 | 数量 | 比例 |
|---|---:|---:|
| Raw total | 269,359 | 100.00% |
| Accepted | 235,757 | 87.53% |
| Rejected | 33,602 | 12.47% |

拒绝原因：

| Reason | Count |
|---|---:|
| duplicate_normalized_question_options | 33,231 |
| invalid_answer | 302 |
| empty_correct_option | 65 |
| missing_answer | 3 |
| garbled_question | 1 |

保留但标记：

| Flag | Count |
|---|---:|
| empty_non_answer_option_retained | 18,629 |
| short_question_retained | 1,654 |

清洗后复查：
- `clean_train` 的 raw duplicate rows：0
- `clean_train` 的 normalized `question + options` duplicate rows：0
- question-only duplicate rows：9,658（4.10%），暂不删除，作为后续语义/模板重复诊断对象。

面试表述：
> "我把基础清洗放在采样之前，先构建 clean candidate pool。清洗规则尽量保守，只处理确定错误：答案无法对应选项、正确选项为空、字段缺失、明显乱码和规范化完全重复。短题干和非答案空选项不直接删除，而是打标记保留，避免误杀合法医学考试题。"

#### CMB 候选池保守向量去重（2026-05-13）

背景：
- 本地暂未安装 `sentence-transformers`，因此先实现一个轻量向量化去重方案。
- 目标不是做激进语义压缩，而是发现“同题轻微改写、标点差异、题干截断、模板重复”等近重复样本。

去重脚本：
- `scripts/vector_dedup_cmb.py`

输入输出：
- 输入：`data/processed/cmb_clean/cmb_train_clean.jsonl`
- 正式分桶去重输出：`data/processed/cmb_clean/cmb_train_vector_dedup_bucket.jsonl`
- 近重复 pair：`data/processed/cmb_clean/cmb_train_vector_pairs_bucket.jsonl`
- 近重复 cluster：`data/processed/cmb_clean/cmb_train_vector_clusters_bucket.json`

实现逻辑：
1. 对 `question + options` 做 NFKC 归一化、去空白、去标点。
2. 使用字符级 n-gram TF-IDF 向量化，默认 n-gram 为 2 到 5。
3. 将题干重复加权，降低“选项相同但题干不同”的配伍题误删风险。
4. 按 `exam_type + exam_class + exam_subject` 分桶，只在同一考试类型/类别/科目内检索近邻。
5. 使用 cosine similarity 检测近重复，阈值 `0.95`。
6. 默认不合并答案不一致的样本，作为安全阀。
7. 每个近重复簇保留质量分更高的样本；若没有显式质量分，则使用题干长度、选项完整度、答案存在性做启发式评分。

Pilot 诊断：
- 未分桶 pilot（20,000 条）：发现 260 个簇，删除 318 条，但存在配伍题误判风险。
- 加入答案一致约束与题干加权后：20,000 条删除 112 条。
- 加入 `exam_type/exam_class/exam_subject` 分桶后：20,000 条删除 67 条，样例主要是标点、句式、题干截断差异。

正式分桶去重结果：

| 项目 | 数量 |
|---|---:|
| Clean pool 输入 | 235,757 |
| 近重复 pairs | 2,575 |
| 近重复 clusters | 2,555 |
| 删除样本 | 2,569 |
| 保留样本 | 233,188 |
| 分桶数 | 182 |
| 答案不一致 pair | 0 |

Top 重复来源桶：

| Bucket | Pair 数 |
|---|---:|
| 医师考试 / 执业助理医师 / 临床执业助理医师 | 205 |
| 医师考试 / 执业医师 / 临床执业医师 | 204 |
| 药师考试 / 执业西药师 / 执业西药师 | 158 |
| 护理考试 / 主管护师 / 主管护师资格考试 | 132 |
| 医学考研 / 西医综合 / 考研西医综合 | 93 |

样例：
- `典型的食管癌症状特点是( )。` vs `典型的食管癌症状特点是`
- `克罗恩病典型的表现中不包括` vs `克罗恩病典型的表现中不包括( )。`
- 同一处方题的“患者病情简介/处方”格式差异版本。

与 Sentence-BERT 语义去重的关系：
- 当前实现是 **TF-IDF 向量近重复去重**，不是深度语义 embedding 去重。
- 优点是本地可运行、可解释、误删风险低，适合先构建 v2 clean pool。
- 后续若安装 `sentence-transformers`，可以在同一分桶内再做 Sentence-BERT 相似度诊断，但不建议一开始用全局语义阈值直接删除。

面试表述：
> "第一版我只做了质量过滤和难度分层，后来发现 optimized 数据重复率更高。于是我把清洗前置到候选池阶段：先做规范化精确去重，再做保守向量近重复去重。向量去重不是全局乱比，而是按 exam_type、exam_class、exam_subject 分桶，只在同一考试域内比较，并要求答案一致，避免把不同科目的相似模板误删。最终从 23.6 万 clean 样本中额外去掉约 2.6k 条近重复，得到 23.3 万条 v2 clean pool。"

#### CoT 候选题第一阶段规则打分（2026-05-13）

目标：
- 在真正调用 teacher 生成 CoT 之前，先从 `clean + dedup` 候选池里筛出更适合显式推理监督的题。
- 第一阶段不是直接判“绝对难题”，而是构建一个可解释、可复现、低成本的 `cot_candidate_score`。

规则文档：
- `docs/cot_candidate_rules.md`

打分脚本：
- `scripts/build_cot_candidates.py`

输入输出：
- 输入：`data/processed/cmb_clean/cmb_train_vector_dedup_bucket.jsonl`
- 全量打分输出：`data/processed/cmb_clean/cmb_cot_candidates_scored.jsonl`
- summary：`data/processed/cmb_clean/cmb_cot_candidates_summary.md`

打分公式：

```text
cot_candidate_score
= 2.0 * is_multi_choice
+ case_info_score
+ option_confusion_score
- definition_penalty
- low_reasoning_penalty
```

当前实现要点：
1. `is_multi_choice`：多选题额外加分。
2. `case_info_score`：按题干覆盖的病例信息类别计分，类别包括人群、症状体征、检查检验、时间过程。
3. `option_confusion_score`：由两部分组成：
   - 选项长度均衡度：`option_length_cv`
   - 选项两两字符 bigram 平均 Jaccard 相似度
4. `definition_penalty`：
   - 强定义模板：`什么是/什么指/概念/定义/含义`
   - 弱定义模板：`特点/特征`
5. `low_reasoning_penalty`：只看题干中“推理证据不足”的条件组合，不重复使用题型和选项混淆度，避免双重计分。

全量打分结果：

| 项目 | 数量 |
|---|---:|
| 输入题目数 | 233,188 |
| `cot_candidate_score` p50 | 0.3 |
| `cot_candidate_score` p75 | 1.2 |
| `cot_candidate_score` p90 | 1.9 |
| `cot_candidate_score` max | 5.1 |
| 多选题数 | 24,081 |
| 强定义题 | 1,078 |
| 弱定义题 | 8,999 |

Top 排名前列样本特征：
- 多为多选题
- 多为病例题或检查题
- 同时覆盖人群、症状、检查、病程中的多个信息类别
- 选项之间词面较接近，适合逐项分析和排除

低分样本特征：
- `处方药的含义是`
- `GTV的定义为`
- `医疗保险的概念`
- `最常见的糖尿病性神经病变是`

这些样本大多属于：
- 纯定义/概念识别题
- 题干很短、无病例信息
- 主要依赖知识点 recall，不太值得 teacher 生成长 CoT

当前结论：
- 第一阶段规则打分已经能把“病例型、多选型、选项相似型”的题提到前列。
- 同时能把“定义题、纯 recall 短题”压到后面。
- 这一步适合作为 teacher 精筛之前的候选池构建步骤，而不是直接替代第二阶段 teacher 判断。

下一步：
- 从 `cmb_cot_candidates_scored.jsonl` 中按 `cot_candidate_score` 先截取一批高分候选题。
- 再调用 DeepSeek 做第二阶段 `CoT-worthiness` 精筛和 teacher rationale 生成。

#### CoT 候选题第二阶段 DeepSeek 精筛（2026-05-13）

目标：
- 在第一阶段规则筛选后的候选池中，进一步判断哪些题真正“值得写 CoT”。
- 第二阶段不生成题目解析，只判断 `CoT-worthiness`，避免把“是否适合 CoT”与“如何写 CoT”混在一起。

精筛脚本：
- `scripts/filter_cot_candidates_with_deepseek.py`

脚本能力：
1. 读取第一阶段输出的 `cmb_cot_candidates_scored.jsonl`
2. 默认只筛规则前 `top 3000` 候选题
3. 调用 DeepSeek 返回结构化字段：
   - `decision`
   - `cot_worthiness_score`
   - `reasoning_need_score`
   - `case_richness_score`
   - `option_confusion_score`
   - `rote_recall_penalty`
   - `definition_penalty`
   - `reason`
   - `tags`
4. 本地校验 JSON 和字段范围，保留 append-only audit
5. 生成最终精筛候选子集，供后续 CoT rationale 生成和混合 SFT 使用

输出文件：
- audit：`data/teacher/cot_candidate_deepseek_audit.jsonl`
- selected：`data/processed/cmb_clean/cmb_cot_candidates_deepseek_selected.jsonl`
- summary：`data/processed/cmb_clean/cmb_cot_candidates_deepseek_summary.md`

Pilot 验证：
- 已用 3 条真实高分候选题跑通 API 调用
- 3/3 返回合法 JSON
- 3/3 校验通过
- DeepSeek 返回的标签集中在：
  - `case`
  - `multi_choice`
  - `option_confusing`
  - `diagnosis`

Pilot 观察：
- 对创伤病例、多选题、检查解释题，DeepSeek 的理由与第一阶段规则方向一致：
  - 病例信息丰富
  - 需要多步推理
  - 选项存在混淆
  - 适合显式写出分析过程

当前结论：
- 第一阶段规则筛负责“可解释、低成本地收窄候选池”
- 第二阶段 DeepSeek 精筛负责“在候选池内部做更细的 CoT 价值判断”
- 两阶段组合比只靠规则或只靠 teacher 都更稳

下一步：
- 正式对规则前 `top 3000` 跑 DeepSeek 精筛
- 从中保留 `top 1000` 作为 CoT 子集
- 再用同一个 teacher 生成 rationale，构造 `Direct + CoT` 混合 SFT 数据

工程修正（2026-05-14）：
- 用户在正式运行 `filter_cot_candidates_with_deepseek.py` 时，脚本长时间运行后失败。
- 复盘发现：虽然流程中断，但 `audit` 中已经成功写入 801 条 `ok` 样本，说明不是“全部失败”，而是中途单条异常导致进程整体退出，且没有及时产出 `selected` 和 `summary`。

已修复的问题：
1. `sample_id` 稳定性：
   - 旧版使用“当前切片局部序号”，在 `offset/limit` 或局部续跑时会影响 `resume`
   - 新版优先使用全局 `cot_candidate_rank` 推导出的稳定 id
2. 长跑进度可见性：
   - 新增 `--progress-every`
   - 定期输出：已尝试数、成功数、错误数、速率、耗时、ETA
3. 容错能力：
   - 单条请求异常不再直接终止全局
   - 异常样本写入 audit，`final_status=request_error`
   - 新增 `--max-consecutive-errors`，连续错误过多时自动停下并保留当前结果
4. 恢复能力：
   - 即使 API 长跑中断，也能基于已有 audit 重新生成 `selected_output` 和 `summary`
   - `resume` 默认跳过已完成的 `ok/invalid` 样本，保留对请求错误样本的重试可能

恢复验证：
- 基于昨天遗留的 `audit` 文件，成功识别已有 801 条 `ok` 结果
- 只读恢复模式下，已能直接重建：
  - `data/processed/cmb_clean/cmb_cot_candidates_deepseek_selected.jsonl`
  - `data/processed/cmb_clean/cmb_cot_candidates_deepseek_summary.md`
- 修复后对中断位置附近又续跑了 4 条新样本，将 audit 从 801 扩展到 805
- 说明新的 `resume + progress + 容错` 逻辑已经可用于正式跑 `top 3000`

### 3.2 脚本准备

#### `scripts/prepare_cmb_sft.py`
数据处理脚本，支持：
- 质量过滤（`--min-quality`）
- 分布对齐（`--distribution-align`）
- 难度分层采样（`--stratified --target-size`）
- 统计信息（`--stats`）

#### `scripts/build_internal_holdout.py`
Holdout 数据集构建脚本（**本地已完成，服务器上未执行**）

#### `scripts/eval_medreason.py`
评测脚本，支持：
- 7 个评测指标：JSON 合规率、鉴别诊断覆盖、风险提示召回、过度自信率、主诊断准确率、字段完整率、响应长度
- 按科室/难度分组统计
- 基座模型 vs SFT 对比
- **注意**：当前评测脚本针对结构化 JSON 输出场景，GRPO 刷分任务需要新的评测脚本

### 3.3 项目路径结构
```
MedicalGPT/
├── data/
│   ├── sft/
│   │   ├── cmb_baseline/           # 尚未上传服务器
│   │   │   ├── cmb_sft_baseline.jsonl
│   │   │   └── cmb_sft_baseline_val.jsonl
│   │   ├── cmb_quality/           # 尚未上传服务器
│   │   │   ├── cmb_sft_quality_only.jsonl
│   │   │   └── cmb_sft_quality_only_val.jsonl
│   │   └── cmb_optimized/          # 尚未上传服务器
│   │       ├── cmb_sft_optimized.jsonl
│   │       └── cmb_sft_optimized_val.jsonl
│   └── raw/hf/CMB/CMB-Exam/      # 已在服务器
│       ├── CMB-train/CMB-train-merge.json
│       ├── CMB-val/CMB-val-merge.json
│       └── CMB-test/CMB-test-choice-question-merge.json
├── scripts/
│   ├── prepare_cmb_sft.py         # 本地完成
│   ├── build_internal_holdout.py   # 本地完成
│   └── eval_medreason.py           # 本地完成
└── training/
    └── supervised_finetuning.py   # 已在服务器（做过两处小修改）
```

---

## 四、当前进度

### 4.1 SFT 训练（Step 3 - 进行中）

**目标**：先跑通 Baseline SFT，拿到第一个 checkpoint。

**数据**：需将本地 `data/sft/cmb_baseline/` 目录上传到服务器。

**SFT 训练命令**（待执行）：
```bash
cd /root/autodl-tmp/MedicalGPT
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/autodl-tmp/hf-cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python training/supervised_finetuning.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
  --train_file_dir data/sft/cmb_baseline \
  --validation_file_dir data/sft/cmb_baseline \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples 5000 \
  --max_eval_samples 280 \
  --model_max_length 1024 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --eval_steps 200 \
  --eval_strategy steps \
  --save_steps 500 \
  --save_strategy steps \
  --save_total_limit 3 \
  --gradient_accumulation_steps 16 \
  --preprocessing_num_workers 2 \
  --output_dir outputs/sft/cmb_baseline \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --bf16 \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --cache_dir ./cache
```

**已验证可用的启动方式**：用 `python3` 直接启动（不用 torchrun），单卡模式。

### 4.2 初步评测结果（2026-05-06）

已使用 `lm-evaluation-harness` 对以下三种模型进行 CMB-val 子集评测：

- Base：未微调的 `Qwen2.5-7B-Instruct`
- Baseline LoRA：`outputs/sft/cmb_baseline`
- Optimized LoRA：`outputs/sft/cmb_optimized`

#### 实验 1：CMB-val 单选题（multiple_choice）

评测设置：
- 数据：CMB-val 单选题子集
- 指标：accuracy
- 方式：`lm_eval` 的 `multiple_choice` 任务

| 模型 | Accuracy | Stderr | 观察 |
|---|---:|---:|---|
| Base | 0.8042 | 0.0257 | 基座模型单选能力已经较强 |
| Baseline LoRA | 0.9542 | 0.0135 | SFT 后接近饱和，说明 CMB 格式适配明显 |
| Optimized LoRA | 0.8000 | 0.0259 | 与 Base 基本持平，未体现优化收益 |

初步结论：
- CMB-val 单选题区分度偏低，适合作为 sanity check，不适合作为唯一主指标。
- Baseline LoRA 提升显著，但可能主要来自 CMB 答题格式/分布适配。
- Optimized LoRA 在单选上未提升，提示质量过滤 + 难度分层可能改变了训练分布。

#### 实验 2：CMB-val 多选题（generate_until + exact_match）

评测设置：
- 数据：CMB-val 多选题子集
- 指标：exact match
- 方式：生成答案字母，抽取 `[A-E]{1,5}` 后与标准答案完全匹配

| 模型 | Exact Match | Stderr | 观察 |
|---|---:|---:|---|
| Base | 0.4359 | 0.0804 | 多选明显更难，基座模型表现下降 |
| Baseline LoRA | 0.8462 | 0.0585 | SFT 对多选 exact match 提升明显 |
| Optimized LoRA | 0.6154 | 0.0789 | 相比 Base 有提升，但弱于 Baseline |

初步结论：
- 多选 exact match 比单选更有区分度，更适合作为 CMB-val 的主要观察指标。
- Baseline LoRA 在单选和多选上均显著优于 Base。
- Optimized LoRA 虽然在多选上优于 Base，但低于 Baseline，说明当前数据优化策略不一定优于随机采样。
- 下一步需要补充 `cmb_quality` 中间消融，并按题型、难度、学科统计训练集分布和评测准确率，定位 optimized 退化原因。

#### 实验 3：训练集分布诊断（2026-05-07）

比较对象：
- `baseline_train`：`data/sft/cmb_baseline/cmb_sft_baseline.jsonl`
- `quality_train`：`data/sft/cmb_quality/cmb_sft_quality_only.jsonl`
- `optimized_train`：`data/sft/cmb_optimized/cmb_sft_optimized.jsonl`
- `val`：`data/sft/cmb_optimized/cmb_sft_optimized_val.jsonl`

关键分布对比：

| 数据集 | 单选比例 | 多选比例 | easy | medium | hard | 平均质量分 | 平均题长 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_train | 89.6% | 10.4% | 1.7% | 98.0% | 0.3% | 0.542 | 23.0 |
| quality_train | 94.0% | 6.0% | 1.9% | 97.6% | 0.5% | 0.652 | 29.0 |
| optimized_train | 95.2% | 4.8% | 31.3% | 50.0% | 18.7% | 0.673 | 39.7 |
| val | 85.7% | 14.3% | 2.9% | 95.7% | 1.4% | 0.534 | 34.4 |

相对 `val` 的主要偏移：
- `baseline_train`：多选比例低 3.8 个百分点，难度分布基本接近 `val`。
- `quality_train`：多选比例低 8.3 个百分点，单选比例偏高。
- `optimized_train`：多选比例低 9.5 个百分点；medium 少 45.7 个百分点；easy 多 28.4 个百分点；hard 多 17.3 个百分点。

初步失败原因判断：
- `optimized_train` 的平均质量分最高，但题型和难度分布明显偏离验证集。
- 分层采样把 `val` 中占绝对多数的 medium 题压到 50%，同时显著提高 easy/hard 比例，导致训练分布与评测分布不匹配。
- 质量过滤和分层采样同时降低了多选题占比，而多选 exact match 是当前更有区分度的指标。
- 因此，这次 optimized 失败更像是 **分布破坏导致的性能下降**，而不是模型训练本身失败。

后续修正方向：
- 不再使用硬阈值过滤后直接分层采样作为主策略。
- 改为 **distribution-preserving quality sampling**：优先保持题型、难度、学科分布接近 `val/test`，质量分只作为采样权重。
- 补充 `quality_train` 模型评测，用于区分“质量过滤问题”和“难度分层问题”。

#### 实验 4：CMExam-test 单选外部泛化评测（2026-05-07）

评测设置：
- 数据：CMExam-test 单选题子集，共 6,606 条
- 指标：accuracy
- 方式：`lm_eval` 的 `multiple_choice` 任务
- 模型：Base、Baseline LoRA、Optimized LoRA

| 模型 | Accuracy | Stderr | 观察 |
|---|---:|---:|---|
| Base | 0.8295 | 0.0046 | 基座模型在 CMExam 单选上已经较强 |
| Baseline LoRA | 0.8438 | 0.0045 | 相比 Base 小幅提升 |
| Optimized LoRA | 0.8521 | 0.0044 | 三者中最高，外部泛化最好 |

关键结论：
- `optimized` 不能简单判定为失败。它在 CMB-val 上不如 Baseline，但在更大规模的 CMExam-test 单选上最好。
- CMB-val 只有 280 条，且拆分单选/多选后样本更小，更适合作为开发集和 sanity check，不适合作为最终结论。
- 当前结果提示：Baseline LoRA 可能更贴合 CMB-val 分布，而 Optimized LoRA 可能更有助于跨 benchmark 泛化。
- 后续需要继续跑 CMExam-test 多选题，并补充 `quality_train` 消融，判断提升来自质量过滤还是难度分层。

#### 实验 5：CMExam-test 多选外部泛化评测（2026-05-07）

评测设置：
- 数据：CMExam-test 多选题子集，共 204 条
- 指标：exact match
- 方式：生成答案字母，抽取 `[A-E]{1,5}` 后与标准答案完全匹配
- 模型：Base、Baseline LoRA、Optimized LoRA

| 模型 | Exact Match | Stderr | 观察 |
|---|---:|---:|---|
| Base | 0.5245 | 0.0351 | 基座多选能力明显弱于 SFT 模型 |
| Baseline LoRA | 0.6324 | 0.0338 | SFT 后多选 exact match 明显提升 |
| Optimized LoRA | 0.6373 | 0.0337 | 略高于 Baseline，三者中最高 |

关键结论：
- CMExam-test 多选结果与单选一致：Optimized LoRA 在外部 benchmark 上略优于 Baseline LoRA。
- 多选样本只有 204 条，Optimized 与 Baseline 差距较小，不能过度解读；但两者均明显优于 Base。
- 结合单选和多选，当前更合理的结论是：Baseline 更贴合 CMB-val，Optimized 更可能提升跨 benchmark 泛化。
- 后续重点是补充 `quality_train` 消融，判断外部泛化提升来自质量过滤、难度分层，还是两者叠加。

#### 实验 6：Quality-only 消融初步结果（2026-05-07）

评测对象：
- Quality-only LoRA：`outputs/sft/cmb_quality`

已完成评测：

| Benchmark | 子集 | 指标 | Quality-only | Stderr | 观察 |
|---|---|---|---:|---:|---|
| CMB-val | 单选 | accuracy | 0.8042 | 0.0257 | 与 Base 持平，明显低于 Baseline |
| CMExam-test | 单选 | accuracy | 0.8513 | 0.0044 | 接近 Optimized 的 0.8521 |
| CMExam-test | 多选 | exact match | 0.6373 | 0.0337 | 与 Optimized 持平 |

暂未完成：
- CMB-val 多选：当前 lm_eval 指令/任务配置待修正。

关键结论：
- Quality-only 在 CMExam-test 单选和多选上几乎追平 Optimized，说明外部泛化提升很可能主要来自 **质量过滤**。
- 难度分层在当前结果中没有表现出额外的明显收益；它可能主要改变了 CMB-val 分布，使 optimized 在 CMB-val 上不占优。
- CMB-val 单选中 Quality-only 与 Base 持平，说明质量过滤并不提升 CMB-val in-domain 贴合度。
- 当前更准确的项目结论是：**随机采样更贴合 CMB-val，质量过滤更有助于 CMExam 外部泛化，强制难度分层收益不明确。**

#### 实验 7：Quality Direct SFT 对照（2026-05-07）

评测对象：
- Quality Direct LoRA：`outputs/sft/cmb_quality_direct`

评测结果：

| Benchmark | 子集 | 指标 | Quality Direct | Stderr | 对比观察 |
|---|---|---|---:|---:|---|
| CMB-val | 单选 | accuracy | 0.8042 | 0.0257 | 与 Quality-only 相同 |
| CMExam-test | 单选 | accuracy | 0.8513 | 0.0044 | 与 Quality-only 相同 |
| CMExam-test | 多选 | exact match | 0.6373 | 0.0337 | 与 Quality-only 相同 |

关键结论：
- Quality Direct 与 Quality-only 结果完全一致，符合预期。
- 原因是三套 CMB train 文件本身已经是 `答案：X` 的 Direct 格式；Direct 数据处理主要统一了 val 标签，把 `分析 + 答案` 改成 `答案：X`。
- 因此，当前已有 SFT checkpoint 不能被严格称为 CoT SFT；更准确地说，它是 **Direct-answer SFT**。
- 若要做 Direct vs CoT 对比，需要重新构造真正的 CoT/explanation-style 训练数据，而不是只改 val。
- 当前项目后续 CoT 工作应调整为：构造 `analysis + answer` 训练集，再与已训练的 Direct SFT 做对照。

### 4.3 后续训练计划

| Step | 任务 | 状态 |
|---|---|---|
| Step 3 | Baseline SFT（随机5k数据） | 已完成初步训练与 CMB-val 评测 |
| Step 4 | Quality-filtered SFT（质量过滤5k数据） | 已完成训练，CMExam 初评完成 |
| Step 5 | Quality+Stratified SFT（分层采样5k数据） | 已完成初步训练与 CMB-val 评测 |
| Step 6 | CMExam-test 外部评测 | 已完成单选和多选初评 |
| Step 7 | Direct vs CoT SFT 对比 | Direct 已确认，CoT train 数据待构造 |
| Step 8 | GRPO 训练（多维度 reward shaping） | 待设计 |
| Step 9 | 对比分析 + 实验报告 | 待执行 |

---

## 五、GRPO 设计方案（待实现）

### 5.1 Reward Function 设计

```
total_reward =
    answer_correct_reward
  + format_reward
  + cot_structure_reward
  + safety_reward
  + uncertainty_reward
```

| 奖励项 | 权重 | 设计意图 |
|---|---|---|
| 答案正确 | +1 / -0.5 | 基础目标 |
| 答案格式合规 | +0.3 / -0.5 | 能稳定抽取 `答案：A/BCDE` |
| CoT 结构完整 | +0.2 | 鼓励 `分析：...\n\n答案：...` |
| 包含"建议"、"可能"等 | +0.1 ~ +0.2 | 鼓励保守表达 |
| 包含"一定"、"肯定"等 | -0.3 ~ -0.5 | 惩罚过度确定 |
| 过长/重复推理 | -0.1 ~ -0.3 | 防止 CoT 冗长和 reward hacking |

注意：医学考试题的 GRPO reward 以可验证的 answer correctness 为主，不声称直接验证真实临床安全性；过度自信率和保守表达只作为医疗安全表达的 proxy 指标。

### 5.2 GRPO vs DPO 选择理由

- GRPO 不需要单独训练 Reward Model，直接用 scalar reward
- 显存友好，适合双卡 48GB 配置
- 多维度 reward shaping 比 DPO 的偏好对更容易精细控制

### 5.3 待解决问题
- reward hacking 的诊断和修复
- 各项 reward 权重的调优
- CoT 的加入：比较 direct answer、CoT SFT、CoT + GRPO 三种设置
- 评测区分：准确率、格式合规率、CoT 结构合规率、过度自信率分开报告，避免只看单一 accuracy

### 5.4 计划中的实验矩阵

| 实验 | 训练方式 | 输出格式 | 主要问题 |
|---|---|---|---|
| Base | 无微调 | 模型原生输出 | 基座模型医学考试能力 |
| SFT-Baseline | CMB 随机 5k | `答案：A` | 随机采样 Direct SFT 是否提升答题能力 |
| SFT-Quality | CMB 质量过滤 5k | `答案：A` | 质量过滤是否提升外部泛化 |
| SFT-Optimized | 质量过滤 + 难度分层 5k | `答案：A` | 数据优化是否破坏分布 |
| Direct SFT | CMB v2 clean pool 采样 | `答案：A` | 去重清洗后的直接答题能力 |
| CoT SFT | Direct + teacher rationale 子集混合 | `分析：...\n\n答案：A` | CoT 是否提升多选和泛化 |
| CoT + GRPO | SFT checkpoint 继续训练 | `分析 + 答案` | 多目标 reward 是否进一步改善格式、准确率和安全表达 |

---

## 六、面试要点总结

### 6.1 项目能闭环的五个地方

1. **数据处理**：随机采样、质量过滤、难度分层都有消融，不只讲"清洗后更好"。
2. **失败分析**：optimized 平均质量分更高但效果更差，定位到题型和难度分布被破坏。
3. **训练选择**：SFT 教任务格式，CoT 显式化推理，GRPO 做多目标偏好优化。
4. **评测设计**：CMB-val 只做开发集，CMExam-test 做外部泛化，C-Eval 医学子集做补充。
5. **安全表达**：不声称临床可用，只用过度自信率、不确定性表达、格式合规率作为 proxy 指标。

### 6.2 推荐面试叙事

> "我选择中文医学考试作为可控实验场景，因为它有明确答案，适合量化评测。项目先用 CMB 做 SFT，比较不同数据构造策略；然后引入 CoT 格式，让模型输出分析过程和答案；最后用 GRPO 做多维 reward shaping，不只优化答案正确率，也约束输出格式、推理结构和医疗安全表达。评测上，我不用单一 CMB-val，而是扩展到 CMExam-test 和 C-Eval 医学子集，验证跨 benchmark 泛化。"

### 6.3 最容易被问的问题及答案

**Q：GRPO 和 DPO 的核心区别是什么？**
A：DPO 用偏好对教模型，需要先构造 chosen/rejected；GRPO 直接用 scalar reward，不需要偏好数据，更适合多维度优化。

**Q：你的 reward function 为什么这样设计？**
A：每一项 reward 对应一个观察到的具体问题。答案正确是基础，格式合规保证可评测和可控，CoT 结构奖励约束先分析再答题，过度自信惩罚和保守表达奖励对应医疗场景中的安全表达需求。

**Q：遇到过什么失败？**
A：第一版 optimized 数据平均质量分最高，但在 CMB-val 上弱于随机 baseline。分布诊断发现它把 medium 题从 val 的 95.7% 压到 50%，同时多选比例也下降，说明单样本质量提高不一定带来整体效果提升，分布保持比硬过滤更重要。

**Q：数据怎么处理，为什么？**
A：质量过滤去掉低质量题目，难度分层保证分布合理，分布对齐让训练集和测试集更接近。每个决策都用题目文本特征，不依赖答案，避免数据泄露。

**Q：为什么不用 CMB-val 当最终结论？**
A：CMB-val 只有 280 条，拆成单选/多选后样本更少，stderr 较高，因此只作为开发集。正式泛化评测扩展到 CMExam-test，它有 6,811 条且包含答案。

**Q：为什么这个项目不是简单刷题？**
A：刷题只看 accuracy；这个项目还比较数据策略、CoT 输出结构、格式合规、过度自信表达和跨 benchmark 泛化，并且有失败诊断和后续 GRPO 设计。

---

## 七、注意事项

1. **脚本修改**：`training/supervised_finetuning.py` 在第 369 行附近加了一行 `"use_auth_token": None`，在第 773 行附近加了 `weights_only=False`，解决本地路径加载问题
2. **数据上传**：本地 `data/sft/cmb_baseline/`、`data/sft/cmb_quality/`、`data/sft/cmb_optimized/` 需要上传到服务器对应目录
3. **不要用 torchrun 多卡启动**：单卡 `python3` 更稳，双卡 NCCL 通信容易出错
4. **不要用 `--warmup_ratio`**：脚本已废弃，用 `--warmup_steps`
5. **不要用 `--flash_attn`**：服务器环境可能不支持

---

## 八、下一步行动

1. **补充 quality-only 消融**：训练并评测 `outputs/sft/cmb_quality`，区分质量过滤问题和难度分层问题。
2. **扩展 CMExam-test 评测**：将 `data/eval/lm_eval/cmexam_*` 和 `tasks/cmexam_choice/` 同步到服务器，评测 Base、Baseline LoRA、Optimized LoRA。
3. **做 Direct vs CoT 对比**：基于 v2 clean pool 构造 Direct SFT，并混合 10%-30% teacher-generated CoT 子集做对照。
4. **重做数据优化策略**：把硬阈值过滤 + 强制难度分层改为 distribution-preserving quality sampling。
5. **实现 GRPO reward**：先做答案正确 + 格式合规 + CoT 结构三个稳定 reward，再加入过度自信惩罚和保守表达奖励。
6. **整理实验报告**：明确区分 CMB-val 开发集结果、CMExam-test 外部泛化结果和 C-Eval 补充结果。

---

## 九、最新进展：CoT SFT 数据构造（2026-05-15）

### 9.1 Teacher CoT 子集生成

- 输入：`data/processed/cmb_clean/cmb_cot_candidates_deepseek_selected.jsonl`
- Teacher：DeepSeek API
- 目标：只让 teacher 生成 `analysis`，不允许修改标准答案。
- 校验策略：
  - teacher 输出必须是合法 JSON
  - teacher `answer` 必须等于 gold answer
  - `analysis` 不能为空，且不能包含 `答案：` 或 `标准答案` 标签
  - 只保留校验通过样本进入 SFT

产物：

| 文件 | 数量 | 说明 |
|---|---:|---|
| `data/teacher/cmb_cot_deepseek_selected_teacher.jsonl` | 1000 | teacher 原始审计输出 |
| `data/sft/cmb_cot_deepseek_selected/cmb_sft_cot_deepseek_selected.jsonl` | 986 | 通过校验的 CoT SFT 子集 |

输出格式：

```text
分析：{teacher_analysis}

答案：{gold_answer}
```

### 9.2 Direct + CoT 混合训练集

为了和此前 `cmb_quality_direct` 做更公平对比，混合集保持训练总量为 5000 条：

- Direct 样本：4014 条
- CoT 样本：986 条
- CoT 占比：19.72%
- 验证集：沿用 `cmb_quality_direct` 的 280 条 Direct val

产物：

| 文件 | 数量 | 说明 |
|---|---:|---|
| `data/sft/cmb_cot_mixed/cmb_sft_cot_mixed.jsonl` | 5000 | Direct + CoT 混合训练集 |
| `data/sft/cmb_cot_mixed/cmb_sft_cot_mixed_val.jsonl` | 280 | 验证集 |
| `data/sft/cmb_cot_mixed/cmb_sft_cot_mixed_summary.md` | - | 混合集分布摘要 |

当前混合比例：

| format | count | ratio |
|---|---:|---:|
| `direct_answer` | 4014 | 80.28% |
| `cot_teacher_deepseek` | 986 | 19.72% |

面试表述：

> "我没有把所有样本都强行改成 CoT，而是先用规则和 DeepSeek 筛出最适合显式推理的题，再让 teacher 生成解析。最终训练集保持 5000 条总量不变，其中约 20% 是 CoT 样本，其余仍是 Direct-answer 样本。这样可以验证 CoT 监督本身是否带来收益，而不是因为训练样本总量变多导致结果变化。"
