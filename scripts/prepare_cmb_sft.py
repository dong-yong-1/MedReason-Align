#!/usr/bin/env python3
"""
prepare_cmb_sft.py
==================
将 CMB-Exam 数据转换为 MedicalGPT SFT 格式，并支持：
  1. 分布对齐（embedding 相似度筛选）
  2. 质量过滤（基于 explanation 字段和题目特征）
  3. 难度分层采样

Usage:
  # Baseline SFT（原始数据，不处理）
  python prepare_cmb_sft.py --output data/sft/cmb_sft_baseline.jsonl

  # 优化版 SFT（分布对齐 + 质量过滤）
  python prepare_cmb_sft.py --output data/sft/cmb_sft_optimized.jsonl \\
       --distribution-align --topk 5 --min-quality 0.5

  # 查看数据概况
  python prepare_cmb_sft.py --stats
"""

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────────────────────
TRAIN_PATH = PROJECT_ROOT / "data/raw/hf/CMB/CMB-Exam/CMB-train/CMB-train-merge.json"
VAL_PATH = PROJECT_ROOT / "data/raw/hf/CMB/CMB-Exam/CMB-val/CMB-val-merge.json"
TEST_PATH = PROJECT_ROOT / "data/raw/hf/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json"

# ── 难度关键词 ───────────────────────────────────────────────────────────
DIFFICULT_KEYWORDS = [
    "鉴别诊断", "鉴别", "不属于", "错误的是", "不包括",
    "急性", "慢性", "并发症", "手术指征", "首选",
    "禁忌", "最常见", "最少见", "首先", "原则",
]
EASY_KEYWORDS = ["体检", "正常", "标准", "定义", "概念"]
MEDICAL_TERMS_DENSITY_PATTERNS = [
    r"[A-Z][αβγδεμμ]+",  # Greek letters
    r"\d+[.]\d+",  # decimals
    r"[+-]?\d+/\d+",  # fractions
    r"\d+mg", r"\d+ml",
]


# ── 辅助函数 ─────────────────────────────────────────────────────────────

def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_difficulty(item: dict) -> str:
    """
    基于题目文本特征估算难度。
    不使用答案，只用题目内容。
    """
    question = item.get("question", "")

    hard_score = sum(1 for kw in DIFFICULT_KEYWORDS if kw in question)
    easy_score = sum(1 for kw in EASY_KEYWORDS if kw in question)
    term_density = sum(
        len(re.findall(p, question)) for p in MEDICAL_TERMS_DENSITY_PATTERNS
    )
    option_count = len(item.get("option", {}))

    # 多选题通常比单选题难
    is_multi = "多选" in item.get("question_type", "")
    if is_multi:
        hard_score += 1

    # 题目长度越长通常越难
    if len(question) > 80:
        hard_score += 1
    elif len(question) < 30:
        easy_score += 1

    if hard_score >= 3 or term_density >= 3:
        return "hard"
    elif easy_score >= 2 and hard_score == 0:
        return "easy"
    return "medium"


def compute_quality_score(item: dict) -> float:
    """
    基于题目文本特征计算质量分数（0.0 - 1.0）。
    训练集没有 explanation 字段，完全基于题目自身特征。
    不使用 answer 字段。
    """
    score = 0.0

    # 题目长度：太短说不清问题，太长可能是复杂病例
    q_len = len(item.get("question", ""))
    if 30 <= q_len <= 200:
        score += 0.20   # 长度适中
    elif 15 <= q_len < 30:
        score += 0.10   # 偏短但可用
    elif q_len > 10:
        score += 0.05   # 勉强可用

    # 选项数量：5个选项是标准考试格式
    opt_count = len(item.get("option", {}))
    if opt_count == 5:
        score += 0.20   # 标准5选
    elif opt_count == 4:
        score += 0.15   # 4选也可以
    elif opt_count >= 3:
        score += 0.05

    # 选项内容质量：每个选项长度适中（非空且有一定内容）
    options = item.get("option", {})
    opt_lens = [len(str(v).strip()) for v in options.values()]
    if opt_lens and all(l > 2 for l in opt_lens):
        avg_len = sum(opt_lens) / len(opt_lens)
        if 5 <= avg_len <= 30:
            score += 0.20   # 选项长度合理
        elif avg_len > 2:
            score += 0.10

    # 包含医学术语密度（说明是专业题目）
    med_terms = re.findall(r"[一-鿿]{2,}", item.get("question", ""))
    if len(med_terms) >= 4:
        score += 0.20   # 术语丰富
    elif len(med_terms) >= 2:
        score += 0.10   # 有一定术语

    # 多选题通常比单选题信息量更大
    if "多选" in item.get("question_type", ""):
        score += 0.05

    # 考试类型为"医师考试"的题目通常质量更高
    if "医师考试" in item.get("exam_type", ""):
        score += 0.15

    return min(score, 1.0)


def build_sft_row(item: dict, sample_id: str) -> dict:
    """
    将一条 CMB-Exam 题目转换为 MedicalGPT ShareGPT 格式。
    使用 CoT 风格：先推理再给出答案。
    """
    question = item["question"]
    options = item["option"]

    # 构建选项字符串
    opt_lines = []
    for key in sorted(options.keys()):
        opt_lines.append(f"{key}. {options[key]}")
    options_text = "\n".join(opt_lines)

    # 构建 user message（不带答案）
    user_message = f"请回答以下医学考试题，直接给出答案选项，不需要解释。\n\n题目：{question}\n\n选项：\n{options_text}"

    # 构建 assistant message
    answer = item.get("answer", "")

    # 构建推理过程（从 explanation 中提取，或通用模板）
    explanation = item.get("explanation", "")
    if explanation and len(explanation) > 20:
        # 取 explanation 前200字作为参考
        reasoning = explanation[:200].strip()
        assistant_message = f"分析：{reasoning}...\n\n答案：{answer}"
    else:
        assistant_message = f"答案：{answer}"

    return {
        "sample_id": sample_id,
        "conversations": [
            {"from": "human", "value": user_message},
            {"from": "gpt", "value": assistant_message},
        ],
        "question": question,
        "options": options,
        "answer": answer,
        "question_type": item.get("question_type", ""),
        "exam_type": item.get("exam_type", ""),
        "exam_subject": item.get("exam_subject", ""),
        "difficulty": estimate_difficulty(item),
        "quality_score": compute_quality_score(item),
    }


# ── 分布对齐 ─────────────────────────────────────────────────────────────

def load_embedding_model():
    """
    加载 sentence embedding 模型。
    优先用本地模型，如果不存在则从 HuggingFace 下载。
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: paraphrase-multilingual-MiniLM-L12-v2")
        # 使用多语言模型，支持中文
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )
        return None


def compute_embeddings(texts: list[str], model, batch_size: int = 64) -> list:
    """批量计算文本 embedding。"""
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def distribution_align(
    train_items: list[dict],
    test_items: list[dict],
    embed_model,
    topk: int = 5,
    min_similarity: float = 0.3,
) -> list[dict]:
    """
    分布对齐：对于每个测试样本，找到训练集中最相似的 topk 个样本。
    返回：去重后的训练集子集。
    """
    logger.info(f"Computing embeddings for {len(train_items)} train + {len(test_items)} test items...")

    train_texts = [f"{i['question']} {' '.join(i['option'].values())}" for i in train_items]
    test_texts = [f"{i['question']} {' '.join(i['option'].values())}" for i in test_items]

    train_embs = compute_embeddings(train_texts, embed_model)
    test_embs = compute_embeddings(test_texts, embed_model)

    logger.info("Computing similarity matrix and selecting aligned samples...")
    selected_indices: set[int] = set()
    sim_count = 0

    for ti, test_emb in enumerate(test_embs):
        similarities = [
            (i, cosine_similarity(test_emb, te)) for i, te in enumerate(train_embs)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        for idx, sim in similarities[:topk]:
            if sim >= min_similarity:
                selected_indices.add(idx)
                sim_count += 1

    logger.info(
        f"Distribution alignment: selected {len(selected_indices)} unique train samples "
        f"({len(selected_indices)/len(train_items)*100:.1f}%) "
        f"from {len(train_items)} candidates, "
        f"avg_sim_calls={sim_count/len(test_items):.1f} per test sample"
    )

    return [train_items[i] for i in sorted(selected_indices)]


# ── 主流程 ────────────────────────────────────────────────────────────────

def load_raw_data() -> tuple[list[dict], list[dict], list[dict]]:
    """加载原始 CMB-Exam 数据。"""
    with open(TRAIN_PATH) as f:
        train_data = json.load(f)
    with open(VAL_PATH) as f:
        val_data = json.load(f)
    with open(TEST_PATH) as f:
        test_data = json.load(f)
    return train_data, val_data, test_data


def filter_and_convert(
    items: list[dict],
    min_quality: float = 0.0,
    difficulty_filter: Optional[str] = None,
    max_samples: Optional[int] = None,
    sample_id_prefix: str = "cmb",
) -> list[dict]:
    """
    过滤并转换数据为 SFT 格式。
    """
    filtered = []
    for i, item in enumerate(items):
        q = item.get("question", "")
        if not q or len(q) < 10:
            continue
        opts = item.get("option", {})
        if len(opts) < 3:
            continue

        quality = compute_quality_score(item)
        if quality < min_quality:
            continue

        if difficulty_filter and estimate_difficulty(item) != difficulty_filter:
            continue

        sid = f"{sample_id_prefix}_{i:06d}"
        sft_row = build_sft_row(item, sid)
        filtered.append(sft_row)

    if max_samples and len(filtered) > max_samples:
        import random
        random.seed(42)
        filtered = random.sample(filtered, max_samples)

    return filtered


def stratified_sample(
    items: list[dict],
    target_size: int,
    seed: int = 42,
) -> list[dict]:
    """
    按难度分层采样，保证训练集难度分布合理。
    """
    import random
    random.seed(seed)

    hard = [i for i in items if i.get("difficulty") == "hard"]
    medium = [i for i in items if i.get("difficulty") == "medium"]
    easy = [i for i in items if i.get("difficulty") == "easy"]

    # 目标分布：hard 30%, medium 50%, easy 20%
    n_hard = min(int(target_size * 0.30), len(hard))
    n_medium = min(int(target_size * 0.50), len(medium))
    n_easy = target_size - n_hard - n_medium
    n_easy = min(n_easy, len(easy))

    sampled = (
        random.sample(hard, n_hard)
        + random.sample(medium, n_medium)
        + random.sample(easy, n_easy)
    )
    random.shuffle(sampled)
    return sampled


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(items)} items → {path}")


def print_stats(items: list[dict], title: str = "") -> None:
    """打印数据统计。"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    diff_counts = defaultdict(int)
    quality_bins = defaultdict(int)
    type_counts = defaultdict(int)
    total_quality = 0.0

    for item in items:
        # 如果没有预先计算，就现算
        diff = item.get("difficulty")
        if diff is None:
            diff = estimate_difficulty(item)
        diff_counts[diff] += 1

        q = item.get("quality_score")
        if q is None:
            q = compute_quality_score(item)
        total_quality += q
        if q >= 0.7:
            quality_bins["high(≥0.7)"] += 1
        elif q >= 0.4:
            quality_bins["medium(0.4-0.7)"] += 1
        else:
            quality_bins["low(<0.4)"] += 1

        qt = item.get("question_type", "unknown")
        if "多选" in qt:
            type_counts["multi-choice"] += 1
        elif "C型" in qt:
            type_counts["c-type"] += 1
        else:
            type_counts["single-choice"] += 1

    print(f"  Total: {len(items)} samples")
    print(f"  Difficulty: {dict(diff_counts)}")
    print(f"  Quality: {dict(quality_bins)}")
    print(f"  Question type: {dict(type_counts)}")
    print(f"  Avg quality score: {total_quality/max(len(items),1):.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare CMB-Exam data for SFT training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stats", action="store_true", help="Show data statistics only")
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "data/sft/cmb_sft_baseline.jsonl"),
    )
    parser.add_argument(
        "--distribution-align", action="store_true",
        help="Enable distribution alignment via embedding similarity"
    )
    parser.add_argument(
        "--topk", type=int, default=5,
        help="For each test sample, select top-k similar train samples (default: 5)"
    )
    parser.add_argument(
        "--min-similarity", type=float, default=0.3,
        help="Minimum cosine similarity threshold (default: 0.3)"
    )
    parser.add_argument(
        "--min-quality", type=float, default=0.0,
        help="Minimum quality score threshold (default: 0.0)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to keep (for quick experiments)"
    )
    parser.add_argument(
        "--stratified", action="store_true",
        help="Enable stratified sampling by difficulty"
    )
    parser.add_argument(
        "--target-size", type=int, default=5000,
        help="Target size after filtering (default: 5000)"
    )
    args = parser.parse_args()

    # 加载数据
    logger.info("Loading CMB-Exam data...")
    train_data, val_data, test_data = load_raw_data()
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    if args.stats:
        print_stats(train_data, "Raw Train Data")
        print_stats(val_data, "Raw Val Data")
        return

    # Step 1: 基础过滤（只保留有答案和选项的题目）
    logger.info("[1/4] Basic filtering...")
    train_filtered = [
        i for i in train_data
        if i.get("question") and i.get("answer") and len(i.get("option", {})) >= 3
    ]
    logger.info(f"After basic filter: {len(train_filtered)} / {len(train_data)}")

    # Step 2: 质量过滤
    if args.min_quality > 0:
        logger.info(f"[2/4] Quality filtering (min={args.min_quality})...")
        before = len(train_filtered)
        train_filtered = [
            i for i in train_filtered
            if compute_quality_score(i) >= args.min_quality
        ]
        logger.info(f"After quality filter: {len(train_filtered)} / {before}")

    # Step 3: 分布对齐
    if args.distribution_align:
        logger.info("[3/4] Distribution alignment...")
        embed_model = load_embedding_model()
        if embed_model is None:
            logger.error("Embedding model required for distribution alignment. Exiting.")
            sys.exit(1)
        # 用 val 数据作为参考分布（val 有 explanation，质量最高）
        # 实际也可以用 test 题目文本做参考（不用答案）
        train_filtered = distribution_align(
            train_filtered, val_data, embed_model,
            topk=args.topk, min_similarity=args.min_similarity
        )
        del embed_model
    else:
        logger.info("[3/4] Skipped distribution alignment (--distribution-align not set)")

    # Step 4: 转换为 SFT 格式
    logger.info("[4/4] Converting to SFT format...")
    sft_items = []
    for i, item in enumerate(train_filtered):
        sid = f"cmb_{i:06d}"
        sft_items.append(build_sft_row(item, sid))

    # 分层采样（如果启用）
    if args.stratified and len(sft_items) > args.target_size:
        logger.info(f"Stratified sampling: {len(sft_items)} → {args.target_size}")
        # 先标记难度
        for item in sft_items:
            item["difficulty"] = estimate_difficulty(item)
        sft_items = stratified_sample(sft_items, args.target_size)

    # 限制数量
    if args.max_samples and len(sft_items) > args.max_samples:
        import random
        random.seed(42)
        sft_items = random.sample(sft_items, args.max_samples)

    # 打印统计
    print_stats(sft_items, "Output SFT Data")

    # 保存
    output_path = Path(args.output)
    save_jsonl(sft_items, output_path)

    # 同时保存一份 val 数据（SFT 验证用）
    val_sft = [
        build_sft_row(item, f"val_{i:06d}")
        for i, item in enumerate(val_data)
        if item.get("question") and item.get("answer")
    ]
    val_path = output_path.parent / f"{output_path.stem}_val.jsonl"
    save_jsonl(val_sft, val_path)

    # 保存配置说明
    config_path = output_path.with_suffix(".config.json")
    config = {
        "source": "CMB-Exam train",
        "total_raw": len(train_data),
        "after_basic_filter": len([i for i in train_data if i.get("question") and i.get("answer")]),
        "distribution_align": args.distribution_align,
        "min_quality": args.min_quality,
        "min_similarity": args.min_similarity,
        "topk": args.topk if args.distribution_align else None,
        "stratified": args.stratified,
        "final_count": len(sft_items),
        "output": str(output_path),
    }
    with config_path.open("w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"Config saved → {config_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
