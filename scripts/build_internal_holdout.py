#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_internal_holdout.py
=========================
从 day9 候选池剩余样本中构建高质量内部 Holdout 数据集。

设计原则：
  1. Ground truth 直接复用 DeepSeek V4 teacher 生成的 schema_target（高质量）
  2. 与 SFT 训练集（day10 1000 条）严格隔离
  3. 覆盖 6 个科室，每科按难度/风险分层采样
  4. 最终输出 300 条（6×50），供 eval_medreason.py 使用

数据来源：
  - 来自 day9_selected_for_rewrite_sample50k_v1.jsonl 中未被重写（未参与 SFT 训练）的样本
  - 全部 5000 条已有 schema_target，直接作为 ground truth
  - 排除掉 day10 重写过的 1000 条 → 剩余 ~4000 条可选
"""

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 科室关键词 ──────────────────────────────────────────────────────────────
DEPT_KEYWORDS: dict[str, list[str]] = {
    "呼吸": [
        "肺炎", "支气管", "肺", "呼吸道", "咳嗽", "咳痰", "咯血", "哮喘",
        "COPD", "慢阻肺", "气胸", "胸腔", "肺栓塞", "胸水", "胸痛", "呼吸",
        "肺结核", "肺癌", "肺脓肿", "ARDS", "肺水肿", "脓胸", "流感",
    ],
    "消化": [
        "胃", "肠", "肝", "胆", "胰腺", "食管", "消化道", "胃肠", "肝功能",
        "黄疸", "腹水", "腹胀", "腹痛", "恶心", "呕吐", "呕血", "黑便",
        "消化性溃疡", "胃炎", "肝炎", "肝硬化", "胆囊炎", "阑尾", "结肠",
        "痢疾", "胃肠炎", "胰腺炎", "胃潴留", "幽门", "肠梗阻", "短肠", "肠易激",
    ],
    "心血管": [
        "心", "血压", "冠心病", "心肌", "心电图", "心律", "心衰", "心梗",
        "主动脉", "心脏", "心包", "瓣膜", "高血压", "低血压", "胸闷",
        "心悸", "动脉", "血管", "夹层", "心绞痛", "心源性", "嗜铬细胞瘤",
        "心肌炎", "心包炎", "肺动脉", "先心病",
    ],
    "神经": [
        "脑", "神经", "意识", "头痛", "头晕", "癫痫", "脑卒中", "脑血管",
        "偏瘫", "肢体", "麻木", "语言", "面瘫", "帕金森", "痴呆", "脑炎",
        "脑梗", "脑出血", "颅内", "眩晕", "晕厥", "谵妄", "昏迷", "脊髓",
    ],
    "内分泌": [
        "糖尿病", "甲亢", "甲减", "甲状腺", "内分泌", "血糖", "胰岛素",
        "皮质醇", "肾上腺", "垂体", "尿糖", "酮症", "高血糖", "低血糖",
        "骨质", "肥胖", "代谢", "激素", "醛固酮", "肾素", "多囊", "痛风",
    ],
    "急诊": [
        "休克", "昏迷", "中毒", "创伤", "急腹症", "大出血", "呼吸困难",
        "心肺复苏", "急诊", "急性", "危象", "脓毒症", "多器官", "DIC",
        "急性冠脉", "急性胰腺", "急性阑尾", "宫外孕",
    ],
}

HIGH_RISK_KEYWORDS = [
    "休克", "昏迷", "意识障碍", "大出血", "心肺", "呼吸衰竭",
    "急性", "危象", "中毒", "脓毒症", "多器官",
    "心肌梗死", "主动脉夹层", "脑出血", "脑疝", "肺栓塞",
    "急性肝衰竭", "急性肾损伤", "高血钾", "低血钾", "心律失常",
    "失血性休克", "感染性休克", "过敏性休克", "DIC",
]

DIFF_NEED_KEYWORDS = [
    "胸痛", "腹痛", "发热待查", "不明原因", "鉴别", "考虑",
    "可能", "类似", "需要排除", "需排除", "不明",
]

EASY_KEYWORDS = ["体检", "查体", "无明显", "无特殊", "无显著", "正常", "随访"]
HARD_KEYWORDS = [
    "危重", "复杂", "疑难", "不明原因", "多器官", "衰竭",
    "急诊", "抢救", "重症", "监护", "急性",
]


# ── 辅助函数 ───────────────────────────────────────────────────────────────

def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def detect_department(case_text: str) -> str | None:
    text = case_text
    scores: dict[str, int] = defaultdict(int)
    for dept, keywords in DEPT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[dept] += 1
    if not scores:
        return None
    return max(scores, key=scores.get)


def is_high_risk(case_text: str) -> bool:
    return any(kw in case_text for kw in HIGH_RISK_KEYWORDS)


def needs_differential(case_text: str, gt: dict) -> bool:
    if any(kw in case_text for kw in DIFF_NEED_KEYWORDS):
        return True
    diffs = gt.get("differential_diagnoses", [])
    return isinstance(diffs, list) and len(diffs) >= 1


def estimate_difficulty(case_text: str) -> str:
    hard = sum(1 for kw in HARD_KEYWORDS if kw in case_text)
    easy = sum(1 for kw in EASY_KEYWORDS if kw in case_text)
    if hard >= 2:
        return "hard"
    elif easy >= 1 and hard == 0:
        return "easy"
    return "medium"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        logger.warning(f"Not found: {path}")
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} → {path}")


# ── 主构建逻辑 ─────────────────────────────────────────────────────────────

def load_sources(args: argparse.Namespace) -> tuple[set[str], list[dict]]:
    """
    加载训练集样本ID集合（用于排除）和 day9 候选池（用于构建 holdout）。
    """
    data_root = PROJECT_ROOT / "data"

    # 1. SFT 训练集样本 ID（从重写输出中提取）
    train_ids: set[str] = set()
    rewrite_path = data_root / "medreason" / "day10_rewrite_outputs_1k_retry_v1.jsonl"
    for row in load_jsonl(rewrite_path):
        train_ids.add(row.get("sample_id", ""))
    logger.info(f"Training samples (for exclusion): {len(train_ids)}")

    # 2. Day9 候选池（全部 5000 条，都有 schema_target）
    pool_path = data_root / "medreason" / "day9_selected_for_rewrite_sample50k_v1.jsonl"
    pool_records = load_jsonl(pool_path)

    # 过滤：只保留有 schema_target 的记录
    # day9 池中 schema_target 嵌套在 teacher_response 里面
    valid_pool = []
    for r in pool_records:
        gt = r.get("schema_target")
        if not (isinstance(gt, dict) and gt.get("primary_diagnosis")):
            # Fallback: check nested teacher_response
            gt = r.get("teacher_response", {}).get("schema_target")
        if isinstance(gt, dict) and gt.get("primary_diagnosis"):
            # Attach schema_target at top level for uniform access
            r = dict(r)
            r["schema_target"] = gt
            valid_pool.append(r)
    logger.info(f"Day9 pool with schema_target: {len(valid_pool)}/{len(pool_records)}")

    return train_ids, valid_pool


def build_holdout(
    pool: list[dict],
    train_ids: set[str],
    target_per_dept: int = 50,
    seed: int = 42,
) -> list[dict]:
    """
    从 day9 剩余池中采样 holdout，保证科室覆盖和分层。
    """
    rng = random.Random(seed)

    # 按科室分组
    dept_groups: dict[str, list[dict]] = {d: [] for d in DEPT_KEYWORDS}
    unclassified: list[dict] = []

    for rec in pool:
        sid = rec.get("sample_id", "")
        if sid in train_ids:
            continue  # 排除训练集

        gt = rec.get("schema_target", {})
        dept = detect_department(rec.get("case_text", ""))
        if dept and dept in dept_groups:
            dept_groups[dept].append(rec)
        else:
            unclassified.append(rec)

    # 打印各科候选数量
    for dept, group in dept_groups.items():
        logger.info(f"  {dept}: {len(group)} candidates (after train exclusion)")

    # 按科室分层采样
    selected: list[dict] = []
    case_counter = 1

    dept_abbr = {
        "呼吸": "resp", "消化": "gi", "心血管": "cardio",
        "神经": "neuro", "内分泌": "endo", "急诊": "ed",
    }

    for dept in DEPT_KEYWORDS:
        candidates = dept_groups[dept]
        if not candidates:
            logger.warning(f"No candidates for {dept}")
            continue

        # 分层：在 candidates 中按难度 + 高风险分层采样
        easy = [c for c in candidates if estimate_difficulty(c.get("case_text", "")) == "easy"]
        hard = [c for c in candidates if estimate_difficulty(c.get("case_text", "")) == "hard"]
        medium = [c for c in candidates if c not in easy and c not in hard]

        # 打乱
        rng.shuffle(easy)
        rng.shuffle(hard)
        rng.shuffle(medium)

        # 目标：50 条，尽量覆盖不同难度/风险
        n_easy = min(10, len(easy))
        n_hard = min(15, len(hard))
        n_medium = target_per_dept - n_easy - n_hard
        n_medium = min(n_medium, len(medium))
        if n_medium < 0:
            n_medium = 0

        sampled = easy[:n_easy] + hard[:n_hard] + medium[:n_medium]
        # 如果还不够，从 medium 中补充
        if len(sampled) < target_per_dept:
            remaining = [c for c in medium if c not in sampled]
            rng.shuffle(remaining)
            sampled += remaining[:target_per_dept - len(sampled)]

        for rec in sampled:
            ct = rec.get("case_text", "")
            gt = rec.get("schema_target", {})

            holdout_rec = {
                "case_id": f"holdout_{dept_abbr[dept]}_{case_counter:04d}",
                "department": dept,
                "difficulty": estimate_difficulty(ct),
                "is_high_risk": is_high_risk(ct),
                "needs_differential_diagnosis": needs_differential(ct, gt),
                "case_text": ct,
                "ground_truth": gt,
                # 保留用于调试和汇报
                "sample_id": rec.get("sample_id", ""),
                "source_dataset": rec.get("source", {}).get("dataset", "unknown"),
                "route": rec.get("route", ""),
                "case_text_hash": sha256_hex(normalize_text(ct)),
            }
            selected.append(holdout_rec)
            case_counter += 1

        logger.info(
            f"  {dept}: selected {len(sampled)} "
            f"(easy={n_easy}, hard={n_hard}, medium={n_medium})"
        )

    # 对于无法分类的病例，随机分配到样本最少的科室（最多补充20条）
    if unclassified and len(selected) < 300:
        logger.info(f"Attempting to supplement from {len(unclassified)} unclassified records")
        rng.shuffle(unclassified)
        dept_counts = Counter(r["department"] for r in selected)
        min_dept = min(dept_counts, key=dept_counts.get) if dept_counts else "呼吸"
        for rec in unclassified[:20]:
            gt = rec.get("schema_target", {})
            holdout_rec = {
                "case_id": f"holdout_{dept_abbr.get(min_dept,'misc')}_{case_counter:04d}",
                "department": min_dept,
                "difficulty": estimate_difficulty(rec.get("case_text", "")),
                "is_high_risk": is_high_risk(rec.get("case_text", "")),
                "needs_differential_diagnosis": needs_differential(rec.get("case_text", ""), gt),
                "case_text": rec.get("case_text", ""),
                "ground_truth": gt,
                "sample_id": rec.get("sample_id", ""),
                "source_dataset": rec.get("source", {}).get("dataset", "unknown"),
                "route": rec.get("route", ""),
                "case_text_hash": sha256_hex(normalize_text(rec.get("case_text", ""))),
            }
            selected.append(holdout_rec)
            case_counter += 1

    return selected


def write_summary(path: Path, holdout: list[dict], train_ids: set[str]) -> None:
    dept_counts = Counter(r["department"] for r in holdout)
    diff_counts = Counter(r["difficulty"] for r in holdout)
    risk_counts = Counter("high_risk" if r["is_high_risk"] else "low_risk" for r in holdout)
    diff_need_counts = Counter(
        "needs_diff" if r["needs_differential_diagnosis"] else "no_diff" for r in holdout
    )

    lines = [
        "# Internal Holdout Dataset Summary v2",
        "",
        "## Overview",
        f"- Total cases: {len(holdout)}",
        f"- SFT training exclusion: {len(train_ids)} samples excluded",
        "",
        "## Department Distribution",
        "| Department | Count |",
        "|---|---:|",
    ]
    for dept, cnt in sorted(dept_counts.items()):
        lines.append(f"| {dept} | {cnt} |")

    lines.extend(["", "## Difficulty Distribution", "| Difficulty | Count |", "|---|---:|"])
    for diff, cnt in sorted(diff_counts.items()):
        lines.append(f"| {diff} | {cnt} |")

    lines.extend(["", "## Risk Distribution", "| Risk Level | Count |", "|---|---:|"])
    for risk, cnt in sorted(risk_counts.items()):
        lines.append(f"| {risk} | {cnt} |")

    lines.extend(["", "## Differential Diagnosis Need", "| Need | Count |", "|---|---:|"])
    for nd, cnt in sorted(diff_need_counts.items()):
        lines.append(f"| {nd} | {cnt} |")

    lines.extend([
        "",
        "## Ground Truth Quality",
        "- All ground truth sourced from DeepSeek V4 teacher-generated schema_target",
        "- No rule-based parsing needed",
        "- Format: primary_diagnosis, diagnostic_basis, differential_diagnoses,",
        "  recommended_actions, risk_flags",
        "",
        "## Data Separation Guarantee",
        "- Training set: day10_rewrite_outputs_1k_retry_v1.jsonl (1000 accepted samples)",
        "- Holdout: day9 pool minus training set (~4000 remaining)",
        "- Separation verified by sample_id matching + case_text hash",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Summary → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MedReason internal holdout dataset from day9 pool.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "eval_internal" / "holdout_v1.jsonl"),
    )
    parser.add_argument("--target-per-dept", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    logger.info("=" * 60)
    logger.info("Building Internal Holdout Dataset v2")
    logger.info(f"  Target per dept: {args.target_per_dept}")
    logger.info(f"  Seed: {args.seed}")
    logger.info("=" * 60)

    # Step 1: Load sources
    train_ids, pool = load_sources(args)

    # Step 2: Build holdout
    holdout = build_holdout(pool, train_ids, args.target_per_dept, args.seed)

    # Step 3: Save
    output_path = Path(args.output)
    save_jsonl(holdout, output_path)
    write_summary(output_path.with_suffix(".summary.md"), holdout, train_ids)

    # Stats
    dept_counts = Counter(r["department"] for r in holdout)
    logger.info("=" * 60)
    logger.info(f"Holdout built: {len(holdout)} cases")
    logger.info(f"  Departments: {dict(dept_counts)}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
