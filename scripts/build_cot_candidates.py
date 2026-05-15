#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score CMB-style questions for CoT candidate selection.

This script implements the first-stage, rule-based CoT candidate filter:

    cot_candidate_score
    = 2.0 * is_multi_choice
    + case_info_score
    + option_confusion_score
    - definition_penalty
    - low_reasoning_penalty

The goal is not to estimate absolute difficulty, but to rank questions by how
useful they are for explicit reasoning supervision.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata
from typing import Any


DEFAULT_INPUT = "data/processed/cmb_clean/cmb_train_vector_dedup_bucket.jsonl"
DEFAULT_OUTPUT = "data/processed/cmb_clean/cmb_cot_candidates_scored.jsonl"
DEFAULT_SUMMARY = "data/processed/cmb_clean/cmb_cot_candidates_summary.md"


CASE_CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "person": re.compile(
        r"患者|病人|患儿|男性|女性|男[，,。]?\d+岁|女[，,。]?\d+岁|孕妇|孕\d+月|老年|儿童|婴儿|婴幼儿|新生儿|产妇"
    ),
    "symptom": re.compile(
        r"发热|疼痛|腹痛|胸痛|头痛|咳嗽|咳痰|气促|呼吸困难|呕吐|恶心|腹泻|便血|咯血|浮肿|乏力|眩晕|抽搐|心悸|黄疸|瘙痒|排尿困难|吞咽困难"
    ),
    "exam": re.compile(
        r"查体|体检|血压|脉搏|体温|呼吸|X线|CT|MRI|B超|超声|彩超|心电图|ECG|实验室|血常规|尿常规|病理|造影|镜检|活检|阳性|阴性|抗体|血清|片示"
    ),
    "timeline": re.compile(
        r"\d+(?:小时|天|周|月|年)|反复|突发|进行性|加重|缓解|既往|病史|起病|来诊|入院|余"
    ),
}

STRONG_DEFINITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"什么是"),
    re.compile(r"什么指"),
    re.compile(r"概念"),
    re.compile(r"定义"),
    re.compile(r"含义"),
)
WEAK_DEFINITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"特点"),
    re.compile(r"特征"),
)
LOW_REASONING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"最常见"),
    re.compile(r"最常用"),
    re.compile(r"首选"),
    re.compile(r"适应证"),
    re.compile(r"禁忌证"),
    re.compile(r"作用"),
    re.compile(r"机制"),
    re.compile(r"第一部"),
    re.compile(r"主要用于"),
)

LENGTH_BALANCE_HIGH_CV = 0.15
LENGTH_BALANCE_MEDIUM_CV = 0.30
BIGRAM_OVERLAP_HIGH = 0.30
BIGRAM_OVERLAP_MEDIUM = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rule-based CoT candidate scores for CMB-style questions.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="All rows with score fields, sorted descending.")
    parser.add_argument("--selected-output", default="", help="Optional filtered subset output.")
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--top-k", type=int, default=0, help="If > 0, keep only the top-k rows in selected output.")
    parser.add_argument("--min-score", type=float, default=None, help="If set, keep rows with score >= min-score in selected output.")
    parser.add_argument("--short-question-threshold", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for debugging or pilot runs.")
    parser.add_argument("--max-examples", type=int, default=10)
    return parser.parse_args()


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} JSON root is not a list")
        return [row for row in data if isinstance(row, dict)]

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def normalize_text(text: Any) -> str:
    value = clean_text(text).lower()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[，。、“”‘’：:；;,.!?？！（）()\[\]【】{}<>《》\"'`~\-—_·|/\\]", "", value)
    return value


def get_options(row: dict[str, Any]) -> dict[str, str]:
    options = row.get("options") if "options" in row else row.get("option")
    if not isinstance(options, dict):
        return {}
    return {str(key).strip().upper(): clean_text(value) for key, value in sorted(options.items())}


def is_multi_choice(row: dict[str, Any]) -> bool:
    question_type = clean_text(row.get("question_type"))
    answer = clean_text(row.get("answer")).upper()
    return "多项选择题" in question_type or len(answer) > 1


def find_pattern_matches(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        found = pattern.findall(text)
        if not found:
            continue
        for match in found:
            if isinstance(match, tuple):
                value = "".join(part for part in match if part)
            else:
                value = str(match)
            if value and value not in matches:
                matches.append(value)
    return matches


def case_info_features(question: str) -> tuple[float, list[str], dict[str, list[str]]]:
    matches_by_category: dict[str, list[str]] = {}
    for category, pattern in CASE_CATEGORY_PATTERNS.items():
        matches = []
        for match in pattern.findall(question):
            value = "".join(match) if isinstance(match, tuple) else str(match)
            if value and value not in matches:
                matches.append(value)
        if matches:
            matches_by_category[category] = matches

    coverage = len(matches_by_category)
    if coverage == 0:
        score = 0.0
    elif coverage == 1:
        score = 0.5
    elif coverage == 2:
        score = 1.0
    else:
        score = 1.5
    return score, sorted(matches_by_category), matches_by_category


def char_bigrams(text: str) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) < 2:
        return {normalized}
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def option_confusion_features(options: dict[str, str]) -> dict[str, float]:
    texts = [text for text in options.values() if normalize_text(text)]
    lengths = [len(normalize_text(text)) for text in texts]

    if not lengths:
        return {
            "option_length_cv": 0.0,
            "option_length_balance_score": 0.0,
            "option_avg_bigram_jaccard": 0.0,
            "option_bigram_overlap_score": 0.0,
            "option_confusion_score": 0.0,
        }

    mean_length = statistics.mean(lengths)
    if len(lengths) > 1 and mean_length > 0:
        std_length = statistics.pstdev(lengths)
        length_cv = std_length / mean_length
    else:
        length_cv = 0.0

    if length_cv <= LENGTH_BALANCE_HIGH_CV:
        length_balance_score = 0.8
    elif length_cv <= LENGTH_BALANCE_MEDIUM_CV:
        length_balance_score = 0.4
    else:
        length_balance_score = 0.0

    pairwise_scores: list[float] = []
    bigram_sets = [char_bigrams(text) for text in texts]
    for i in range(len(bigram_sets)):
        for j in range(i + 1, len(bigram_sets)):
            pairwise_scores.append(jaccard(bigram_sets[i], bigram_sets[j]))
    avg_bigram_jaccard = statistics.mean(pairwise_scores) if pairwise_scores else 0.0

    if avg_bigram_jaccard >= BIGRAM_OVERLAP_HIGH:
        bigram_overlap_score = 0.8
    elif avg_bigram_jaccard >= BIGRAM_OVERLAP_MEDIUM:
        bigram_overlap_score = 0.4
    else:
        bigram_overlap_score = 0.0

    return {
        "option_length_cv": round(length_cv, 6),
        "option_length_balance_score": length_balance_score,
        "option_avg_bigram_jaccard": round(avg_bigram_jaccard, 6),
        "option_bigram_overlap_score": bigram_overlap_score,
        "option_confusion_score": round(length_balance_score + bigram_overlap_score, 6),
    }


def definition_features(question: str) -> tuple[float, str, list[str]]:
    strong_matches = find_pattern_matches(question, STRONG_DEFINITION_PATTERNS)
    if strong_matches:
        return 1.0, "strong", strong_matches

    weak_matches = find_pattern_matches(question, WEAK_DEFINITION_PATTERNS)
    if weak_matches:
        return 0.3, "weak", weak_matches

    return 0.0, "none", []


def low_reasoning_features(question: str, case_info_score: float, short_threshold: int) -> tuple[float, list[str], list[str]]:
    flags: list[str] = []
    recall_matches = find_pattern_matches(question, LOW_REASONING_PATTERNS)
    question_length = len(normalize_text(question))

    if question_length < short_threshold:
        flags.append("short_question")
    if case_info_score == 0:
        flags.append("no_case_info")
    if recall_matches:
        flags.append("recall_template")

    if len(flags) <= 1:
        penalty = 0.0
    elif len(flags) == 2:
        penalty = 0.5
    else:
        penalty = 1.0
    return penalty, flags, recall_matches


def cot_reasons(
    multi_choice: bool,
    case_categories: list[str],
    option_features: dict[str, float],
    definition_level: str,
    definition_matches: list[str],
    low_reasoning_penalty: float,
    low_reasoning_flags: list[str],
    recall_matches: list[str],
) -> list[str]:
    reasons: list[str] = []
    if multi_choice:
        reasons.append("multi_choice")
    if case_categories:
        reasons.append(f"case_info:{','.join(case_categories)}")
    if option_features["option_length_balance_score"] > 0:
        reasons.append("option_length_balanced")
    if option_features["option_bigram_overlap_score"] > 0:
        reasons.append("option_bigram_overlap")
    if definition_level != "none":
        reasons.append(f"{definition_level}_definition:{'|'.join(definition_matches)}")
    if low_reasoning_penalty > 0:
        suffix = ",".join(low_reasoning_flags)
        if recall_matches:
            suffix += f"; recall={'|'.join(recall_matches)}"
        reasons.append(f"low_reasoning:{suffix}")
    return reasons


def score_row(row: dict[str, Any], short_threshold: int) -> dict[str, Any]:
    question = clean_text(row.get("question"))
    options = get_options(row)
    multi_choice = is_multi_choice(row)

    case_score, case_categories, case_matches = case_info_features(question)
    option_features = option_confusion_features(options)
    definition_penalty, definition_level, definition_matches = definition_features(question)
    low_penalty, low_flags, recall_matches = low_reasoning_features(question, case_score, short_threshold)

    total_score = (
        2.0 * float(multi_choice)
        + case_score
        + option_features["option_confusion_score"]
        - definition_penalty
        - low_penalty
    )

    scored = dict(row)
    scored["cot_candidate_score"] = round(total_score, 6)
    scored["is_multi_choice"] = multi_choice
    scored["case_info_score"] = case_score
    scored["case_info_categories"] = case_categories
    scored["case_info_matches"] = case_matches
    scored["option_confusion_score"] = option_features["option_confusion_score"]
    scored["option_length_balance_score"] = option_features["option_length_balance_score"]
    scored["option_length_cv"] = option_features["option_length_cv"]
    scored["option_bigram_overlap_score"] = option_features["option_bigram_overlap_score"]
    scored["option_avg_bigram_jaccard"] = option_features["option_avg_bigram_jaccard"]
    scored["definition_penalty"] = definition_penalty
    scored["definition_level"] = definition_level
    scored["definition_matches"] = definition_matches
    scored["low_reasoning_penalty"] = low_penalty
    scored["low_reasoning_flags"] = low_flags
    scored["low_reasoning_matches"] = recall_matches
    scored["cot_candidate_reason"] = cot_reasons(
        multi_choice=multi_choice,
        case_categories=case_categories,
        option_features=option_features,
        definition_level=definition_level,
        definition_matches=definition_matches,
        low_reasoning_penalty=low_penalty,
        low_reasoning_flags=low_flags,
        recall_matches=recall_matches,
    )
    return scored


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = position - lower
    return lower_value * (1 - weight) + upper_value * weight


def select_rows(rows: list[dict[str, Any]], top_k: int, min_score: float | None) -> list[dict[str, Any]]:
    selected = rows
    if min_score is not None:
        selected = [row for row in selected if float(row["cot_candidate_score"]) >= min_score]
    if top_k > 0:
        selected = selected[:top_k]
    return selected


def default_selected_output(path: Path) -> Path:
    return path.with_name(f"{path.stem}_selected{path.suffix}")


def write_summary(
    path: Path,
    args: argparse.Namespace,
    scored_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = [float(row["cot_candidate_score"]) for row in scored_rows]
    multi_count = sum(1 for row in scored_rows if row["is_multi_choice"])
    definition_counts = Counter(str(row["definition_level"]) for row in scored_rows)
    low_reasoning_counts = Counter(float(row["low_reasoning_penalty"]) for row in scored_rows)
    case_coverage_counts = Counter(len(row["case_info_categories"]) for row in scored_rows)
    reason_counts = Counter()
    for row in scored_rows:
        reason_counts.update(row.get("cot_candidate_reason", []))

    lines = [
        "# CoT Candidate Scoring Summary",
        "",
        f"- Input: `{args.input}`",
        f"- Output: `{args.output}`",
        f"- Rows scored: {len(scored_rows)}",
        f"- Selected rows: {len(selected_rows)}",
        f"- top_k: {args.top_k}",
        f"- min_score: {args.min_score}",
        f"- short_question_threshold: {args.short_question_threshold}",
        "",
        "## Score Distribution",
        "",
        f"- min: {min(scores):.3f}" if scores else "- min: 0.000",
        f"- p50: {percentile(scores, 0.50):.3f}" if scores else "- p50: 0.000",
        f"- p75: {percentile(scores, 0.75):.3f}" if scores else "- p75: 0.000",
        f"- p90: {percentile(scores, 0.90):.3f}" if scores else "- p90: 0.000",
        f"- max: {max(scores):.3f}" if scores else "- max: 0.000",
        f"- mean: {statistics.mean(scores):.3f}" if scores else "- mean: 0.000",
        "",
        "## Component Counts",
        "",
        f"- multi_choice: {multi_count}",
        f"- definition strong: {definition_counts.get('strong', 0)}",
        f"- definition weak: {definition_counts.get('weak', 0)}",
        f"- low_reasoning penalty=0.5: {low_reasoning_counts.get(0.5, 0)}",
        f"- low_reasoning penalty=1.0: {low_reasoning_counts.get(1.0, 0)}",
        "",
        "## Case Coverage",
        "",
        "| Covered categories | Count |",
        "|---|---:|",
    ]

    for covered, count in sorted(case_coverage_counts.items()):
        lines.append(f"| {covered} | {count} |")

    lines.extend(["", "## Top Reasons", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in reason_counts.most_common(12):
        lines.append(f"| {reason} | {count} |")

    lines.extend(["", "## Top Candidate Examples"])
    for row in scored_rows[: args.max_examples]:
        question = clean_text(row.get("question"))[:180]
        reasons = "; ".join(row.get("cot_candidate_reason", [])[:4])
        lines.append(f"- score={row['cot_candidate_score']:.3f}; question={question}; reasons={reasons}")

    lines.extend(["", "## Low Candidate Examples"])
    for row in scored_rows[-args.max_examples :]:
        question = clean_text(row.get("question"))[:180]
        reasons = "; ".join(row.get("cot_candidate_reason", [])[:4])
        lines.append(f"- score={row['cot_candidate_score']:.3f}; question={question}; reasons={reasons}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    rows = load_json_or_jsonl(input_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    scored_rows = [score_row(row, args.short_question_threshold) for row in rows]
    scored_rows.sort(key=lambda row: (float(row["cot_candidate_score"]), len(clean_text(row.get("question")))), reverse=True)
    for rank, row in enumerate(scored_rows, start=1):
        row["cot_candidate_rank"] = rank

    selected_rows = select_rows(scored_rows, args.top_k, args.min_score)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, scored_rows)

    selected_path: Path | None = None
    if args.selected_output or args.top_k > 0 or args.min_score is not None:
        selected_path = Path(args.selected_output) if args.selected_output else default_selected_output(output_path)
        write_jsonl(selected_path, selected_rows)

    write_summary(summary_path, args, scored_rows, selected_rows)

    message = f"rows={len(scored_rows)} selected={len(selected_rows)} wrote={output_path}"
    if selected_path is not None:
        message += f" selected_output={selected_path}"
    print(message)
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
