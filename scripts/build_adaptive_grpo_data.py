#!/usr/bin/env python3
"""Build adaptive GRPO data from scored CMB clean-pool rows."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/processed/cmb_clean/cmb_cot_candidates_scored.jsonl"
DEFAULT_OUTPUT_DIR = "data/grpo"
TRAIN_FILENAME = "cmb_adaptive_grpo_train.jsonl"
SUMMARY_FILENAME = "cmb_adaptive_grpo_summary.md"
FULL_DISTRIBUTION_FILENAME = "cmb_adaptive_grpo_full_distribution.json"
DIRECT_REVIEW_FILENAME = "cmb_adaptive_grpo_direct_review_50.jsonl"
BRIEF_REVIEW_FILENAME = "cmb_adaptive_grpo_brief_review_50.jsonl"
COT_REVIEW_FILENAME = "cmb_adaptive_grpo_cot_review_50.jsonl"
DIFFICULTIES = ("direct", "brief", "cot")

PERSON_PATTERN = re.compile(r"患者|病人|患儿|男性|女性|男孩|女孩|孕妇|产妇")
SPECIFIC_PATIENT_PATTERN = re.compile(
    r"\d+\s*岁|老年男性|老年女性|中年男性|中年女性|青年男性|青年女性|"
    r"男[，,。]?\d+岁|女[，,。]?\d+岁|初孕妇|初产妇"
)
CLINICAL_CONTEXT_PATTERN = re.compile(
    r"主诉|因|就诊|来诊|入院|急诊|查体|体检|病史|既往|诊断|治疗|术后|检查|"
    r"血压|体温|脉搏|CT|MRI|B超|X线|心电图|实验室|阳性|阴性|"
    r"发热|疼痛|腹痛|胸痛|咳嗽|气促|呼吸困难|呕吐|腹泻|便血|尿频|尿急|"
    r"乏力|水肿|肿块|出血|偏瘫|昏迷|抽搐"
)

PROMPT_TEMPLATE = """你是一名医学考试助手。请根据题目合理作答，最后严格按照"答案：X"格式输出。

作答要求：
1. 简单概念题和定义题直接作答，不需要分析。
2. 复杂的病例分析题先给出推理过程再作答。
3. 单选题只能输出一个选项。
4. 多选题输出多个选项，只选确定正确的，不要多选。
5. 最终答案只允许包含选项字母，例如：答案：A 或 答案：ABCD。

题目：{question}

选项：
{options_text}

请作答："""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-output", default="")
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--full-distribution-output", default="")
    parser.add_argument("--direct-review-output", default="")
    parser.add_argument("--brief-review-output", default="")
    parser.add_argument("--cot-review-output", default="")
    parser.add_argument("--samples-per-difficulty", type=int, default=1500)
    parser.add_argument("--review-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def output_path(args: argparse.Namespace, override: str, filename: str) -> Path:
    if override:
        return Path(override)
    return Path(args.output_dir) / filename


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def as_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def normalize_options(row: dict[str, Any]) -> dict[str, str] | None:
    raw_options = row.get("options") if isinstance(row.get("options"), dict) else row.get("option")
    if not isinstance(raw_options, dict):
        return None

    options: dict[str, str] = {}
    for key, value in raw_options.items():
        letter = str(key).strip().upper()
        text = str(value or "").strip()
        if len(letter) == 1 and "A" <= letter <= "Z" and text:
            options[letter] = text
    return dict(sorted(options.items())) if options else None


def raw_option_values(row: dict[str, Any]) -> list[str]:
    raw_options = row.get("options") if isinstance(row.get("options"), dict) else row.get("option")
    if not isinstance(raw_options, dict):
        return []
    return [str(value or "").strip() for value in raw_options.values() if str(value or "").strip()]


def normalize_answer(value: Any, valid_options: set[str]) -> str:
    raw_answer = str(value or "").strip().upper()
    if not raw_answer:
        return ""

    letters: list[str] = []
    for letter in sorted(valid_options):
        if letter in raw_answer:
            letters.append(letter)
    return "".join(letters)


def options_to_text(options: dict[str, str]) -> str:
    return "\n".join(f"{key}. {value}" for key, value in sorted(options.items()) if value)


def infer_is_multi(row: dict[str, Any], answer: str) -> bool:
    return as_bool(row.get("is_multi_choice")) or "多" in str(row.get("question_type") or "") or len(answer) > 1


def infer_question_type(row: dict[str, Any], is_multi: bool) -> str:
    source_type = str(row.get("question_type") or "").strip()
    if "多" in source_type or is_multi:
        return "multi"
    return "single"


def reasoning_score(row: dict[str, Any]) -> float:
    score = (
        as_float(row, "case_info_score")
        + as_float(row, "option_confusion_score")
        - as_float(row, "definition_penalty")
        - as_float(row, "low_reasoning_penalty")
    )
    return round(score, 6)


def classify_difficulty(row: dict[str, Any], score: float, is_multi: bool) -> str:
    definition_level = str(row.get("definition_level") or "").strip().lower()
    case_info_score = as_float(row, "case_info_score")
    definition_penalty = as_float(row, "definition_penalty")
    low_reasoning_penalty = as_float(row, "low_reasoning_penalty")

    if definition_level == "strong":
        return "direct"
    if score <= -0.3:
        return "direct"
    if score > 0.5:
        if is_multi and is_multi_concept_or_recall(row, case_info_score, definition_penalty, low_reasoning_penalty):
            return "brief"
        return "cot"
    return "brief"


def has_real_case_context(row: dict[str, Any]) -> bool:
    texts = [str(row.get("question") or ""), *raw_option_values(row)]
    for text in texts:
        if SPECIFIC_PATIENT_PATTERN.search(text) and CLINICAL_CONTEXT_PATTERN.search(text):
            return True
        if PERSON_PATTERN.search(text) and re.search(r"主诉|就诊|来诊|入院|急诊|查体|体检|病史|既往", text):
            return True
    return False


def is_multi_concept_or_recall(
    row: dict[str, Any],
    case_info_score: float,
    definition_penalty: float,
    low_reasoning_penalty: float,
) -> bool:
    return not has_real_case_context(row)


def convert_row(row: dict[str, Any], index: int, input_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    question = str(row.get("question") or "").strip()
    if not question:
        return None, "empty_question"

    options = normalize_options(row)
    if options is None:
        return None, "invalid_options"

    answer = normalize_answer(row.get("answer"), set(options))
    if not answer:
        return None, "invalid_answer"

    is_multi = infer_is_multi(row, answer)
    question_type = infer_question_type(row, is_multi)
    score = reasoning_score(row)
    difficulty = classify_difficulty(row, score, is_multi)
    sample_id = str(row.get("sample_id") or row.get("id") or f"{input_path.stem}_{index}")
    prompt = PROMPT_TEMPLATE.format(question=question, options_text=options_to_text(options))

    out: dict[str, Any] = {
        "sample_id": sample_id,
        "prompt": prompt,
        "answer": answer,
        "difficulty": difficulty,
        "reasoning_score": score,
        "question": question,
        "options": options,
        "question_type": question_type,
        "source_question_type": row.get("question_type"),
        "is_multi_choice": is_multi,
        "case_info_score": as_float(row, "case_info_score"),
        "option_confusion_score": as_float(row, "option_confusion_score"),
        "definition_penalty": as_float(row, "definition_penalty"),
        "definition_level": row.get("definition_level"),
        "low_reasoning_penalty": as_float(row, "low_reasoning_penalty"),
        "has_real_case_context": has_real_case_context(row),
        "exam_type": row.get("exam_type"),
        "exam_class": row.get("exam_class"),
        "exam_subject": row.get("exam_subject"),
        "source": str(input_path),
    }

    for key in ("clean_flags", "case_info_categories", "cot_candidate_score", "cot_candidate_rank", "cot_candidate_reason"):
        if key in row:
            out[key] = row[key]
    return out, None


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def score_quantiles(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = sorted(float(row["reasoning_score"]) for row in rows)
    if not scores:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "min": scores[0],
        "p25": quantile(scores, 0.25),
        "p50": quantile(scores, 0.50),
        "p75": quantile(scores, 0.75),
        "p90": quantile(scores, 0.90),
        "max": scores[-1],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "difficulty": dict(Counter(row["difficulty"] for row in rows)),
        "question_type": dict(Counter(row["question_type"] for row in rows)),
        "reasoning_score_quantiles": score_quantiles(rows),
    }


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def review_projection(row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "sample_id",
        "difficulty",
        "reasoning_score",
        "question",
        "answer",
        "options",
        "case_info_score",
        "option_confusion_score",
        "definition_penalty",
        "definition_level",
        "low_reasoning_penalty",
        "exam_type",
        "exam_class",
    ]
    return {key: row.get(key) for key in keep}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_counter_table(counter: dict[str, int], keys: tuple[str, ...] | None = None) -> list[str]:
    rows = ["| key | count |", "|---|---:|"]
    ordered_keys = keys if keys is not None else tuple(sorted(counter))
    for key in ordered_keys:
        rows.append(f"| {key} | {counter.get(key, 0)} |")
    return rows


def write_summary(
    path: Path,
    input_path: Path,
    train_output: Path,
    full_distribution_output: Path,
    review_outputs: dict[str, Path],
    full_summary: dict[str, Any],
    train_summary: dict[str, Any],
    skip_counts: Counter[str],
    samples_per_difficulty: int,
    review_count: int,
) -> None:
    lines = [
        "# Adaptive GRPO Data Summary",
        "",
        "## Inputs And Outputs",
        "",
        f"- input: `{input_path}`",
        f"- train_output: `{train_output}`",
        f"- full_distribution_json: `{full_distribution_output}`",
        f"- direct_review_jsonl: `{review_outputs['direct']}`",
        f"- brief_review_jsonl: `{review_outputs['brief']}`",
        f"- cot_review_jsonl: `{review_outputs['cot']}`",
        "",
        "## Classification Rules",
        "",
        "- `reasoning_score = case_info_score + option_confusion_score - definition_penalty - low_reasoning_penalty`",
        "- `definition_level == strong` -> direct",
        "- `reasoning_score <= -0.3` -> direct",
        "- `reasoning_score > 0.5` -> cot, except multi-choice concept/recall rows without real case context are brief",
        "- remaining rows -> brief",
        "",
        "## Full Distribution",
        "",
        f"- total: {full_summary['total']}",
        "",
        "### By Difficulty",
        "",
        *markdown_counter_table(full_summary["difficulty"], DIFFICULTIES),
        "",
        "### By Question Type",
        "",
        *markdown_counter_table(full_summary["question_type"]),
        "",
        "### Full Reasoning Score Quantiles",
        "",
        "```json",
        json.dumps(full_summary["reasoning_score_quantiles"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Train Sampling",
        "",
        f"- target_per_difficulty: {samples_per_difficulty}",
        f"- total: {train_summary['total']}",
        "",
        "### Train By Difficulty",
        "",
        *markdown_counter_table(train_summary["difficulty"], DIFFICULTIES),
        "",
        "### Train By Question Type",
        "",
        *markdown_counter_table(train_summary["question_type"]),
        "",
        "### Train Reasoning Score Quantiles",
        "",
        "```json",
        json.dumps(train_summary["reasoning_score_quantiles"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Review Sampling",
        "",
        f"- review_count_per_difficulty: {review_count}",
        "- direct review: random single-choice rows from direct; falls back to all direct rows if needed.",
        "- brief review: random rows from brief.",
        "- cot review: random multi-choice rows from cot; falls back to all cot rows if needed.",
        "",
        "## Skip Counts",
        "",
        "```json",
        json.dumps(dict(skip_counts), ensure_ascii=False, indent=2),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    rng = random.Random(args.seed)

    train_output = output_path(args, args.train_output, TRAIN_FILENAME)
    summary_output = output_path(args, args.summary_output, SUMMARY_FILENAME)
    full_distribution_output = output_path(args, args.full_distribution_output, FULL_DISTRIBUTION_FILENAME)
    review_outputs = {
        "direct": output_path(args, args.direct_review_output, DIRECT_REVIEW_FILENAME),
        "brief": output_path(args, args.brief_review_output, BRIEF_REVIEW_FILENAME),
        "cot": output_path(args, args.cot_review_output, COT_REVIEW_FILENAME),
    }

    converted_rows: list[dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, row in enumerate(load_jsonl(input_path)):
        converted, skip_reason = convert_row(row, idx, input_path)
        if converted is None:
            skip_counts[skip_reason or "unknown"] += 1
            continue
        converted_rows.append(converted)
        buckets[converted["difficulty"]].append(converted)

    train_rows: list[dict[str, Any]] = []
    sampled_counts: dict[str, int] = {}
    for difficulty in DIFFICULTIES:
        sampled = sample_rows(buckets[difficulty], args.samples_per_difficulty, rng)
        sampled_counts[difficulty] = len(sampled)
        train_rows.extend(sampled)
    rng.shuffle(train_rows)

    direct_review_pool = [row for row in buckets["direct"] if row["question_type"] == "single"] or buckets["direct"]
    cot_review_pool = [row for row in buckets["cot"] if row["question_type"] == "multi"] or buckets["cot"]
    reviews = {
        "direct": [review_projection(row) for row in sample_rows(direct_review_pool, args.review_count, rng)],
        "brief": [review_projection(row) for row in sample_rows(buckets["brief"], args.review_count, rng)],
        "cot": [review_projection(row) for row in sample_rows(cot_review_pool, args.review_count, rng)],
    }

    full_summary = summarize_rows(converted_rows)
    train_summary = summarize_rows(train_rows)
    distribution = {
        "input": str(input_path),
        "outputs": {
            "train": str(train_output),
            "summary": str(summary_output),
            "full_distribution": str(full_distribution_output),
            "direct_review": str(review_outputs["direct"]),
            "brief_review": str(review_outputs["brief"]),
            "cot_review": str(review_outputs["cot"]),
        },
        "samples_per_difficulty": args.samples_per_difficulty,
        "sampled_counts": sampled_counts,
        "review_count": args.review_count,
        "full": full_summary,
        "train": train_summary,
        "skip_counts": dict(skip_counts),
    }

    write_jsonl(train_output, train_rows)
    for difficulty, rows in reviews.items():
        write_jsonl(review_outputs[difficulty], rows)
    write_json(full_distribution_output, distribution)
    write_summary(
        summary_output,
        input_path,
        train_output,
        full_distribution_output,
        review_outputs,
        full_summary,
        train_summary,
        skip_counts,
        args.samples_per_difficulty,
        args.review_count,
    )

    print(f"converted={len(converted_rows)} skipped={sum(skip_counts.values())}")
    print(f"train={len(train_rows)} sampled_counts={sampled_counts}")
    print(f"wrote {train_output}")
    print(f"wrote {summary_output}")
    print(f"wrote {full_distribution_output}")
    for difficulty in DIFFICULTIES:
        print(f"wrote {review_outputs[difficulty]}")


if __name__ == "__main__":
    main()
