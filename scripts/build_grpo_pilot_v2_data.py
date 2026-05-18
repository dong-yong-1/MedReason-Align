#!/usr/bin/env python3
"""Build the answer-only GRPO v2 prompt pool for medical choice questions."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_INPUTS = [
    "data/processed/cmb_clean/cmb_train_vector_dedup_bucket.jsonl",
    "data/sft/cmb_cot_mixed/cmb_sft_cot_mixed.jsonl",
    "data/sft/cmb_optimized_direct/cmb_sft_optimized_direct.jsonl",
]

TARGET_PLAN = {
    "single": 1000,
    "multi_2": 500,
    "multi_3": 500,
    "multi_4": 375,
    "multi_5": 100,
    "multi_6": 25,
}

PROMPT_TEMPLATE_V2 = """你是一名医学考试助手。请根据题目选择正确选项。

作答要求：
1. 最终只输出一行，格式必须为：答案：X
2. X 只能由选项字母组成，例如：答案：A 或 答案：ABCD
3. 单选题只能输出一个选项。
4. 多选题只能选择题干明确支持、符合标准定义或诊疗原则的选项。
5. 仅“可能相关”“不能完全排除”“也可考虑”的选项不要选择。
6. 不要输出分析过程，不要输出多余文字。

题目：{question}

选项：
{options_text}

请作答："""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="", help="Optional explicit input JSONL path.")
    parser.add_argument("--output-jsonl", default="data/grpo/cmb_grpo_pilot_v2_train.jsonl")
    parser.add_argument("--summary-md", default="data/grpo/cmb_grpo_pilot_v2_summary.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-total", type=int, default=3000)
    return parser.parse_args()


def choose_input(explicit: str) -> Path:
    candidates = [explicit] if explicit else DEFAULT_INPUTS
    for item in candidates:
        if item and Path(item).exists():
            return Path(item)
    raise FileNotFoundError("No usable input file found: " + ", ".join(candidates))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def normalize_options(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    options: dict[str, str] = {}
    for key, text in value.items():
        letter = str(key).strip().upper()
        if len(letter) == 1 and "A" <= letter <= "Z":
            options[letter] = str(text).strip()
    return dict(sorted(options.items())) if options else None


def normalize_answer(value: Any, valid_options: set[str]) -> str:
    letters: list[str] = []
    for ch in str(value or "").upper():
        if "A" <= ch <= "Z" and ch in valid_options and ch not in letters:
            letters.append(ch)
    return "".join(sorted(letters))


def options_to_text(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))


def convert_row(row: dict[str, Any], index: int, source_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    question = str(row.get("question") or "").strip()
    if not question:
        return None, "empty_question"

    options = normalize_options(row.get("options") or row.get("option"))
    if options is None:
        return None, "invalid_options"

    valid_options = set(options)
    answer = normalize_answer(row.get("answer"), valid_options)
    if not answer:
        return None, "invalid_answer"
    if any(ch not in valid_options for ch in answer):
        return None, "answer_not_in_options"

    sample_id = str(row.get("sample_id") or row.get("id") or f"{source_path.stem}_{index}")
    question_type = "single" if len(answer) == 1 else "multi"
    return (
        {
            "sample_id": sample_id,
            "prompt": PROMPT_TEMPLATE_V2.format(question=question, options_text=options_to_text(options)),
            "answer": answer,
            "question": question,
            "options": options,
            "valid_options": sorted(options),
            "question_type": question_type,
            "answer_len": len(answer),
            "source": str(source_path),
        },
        None,
    )


def bucket_for(row: dict[str, Any]) -> str:
    return "single" if row["answer_len"] == 1 else f"multi_{row['answer_len']}"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    input_path: Path,
    output_path: Path,
    selected: list[dict[str, Any]],
    skip_counts: Counter[str],
    available_counts: Counter[str],
    selected_counts: Counter[str],
) -> None:
    total = len(selected)
    answer_len_counts = Counter(str(row["answer_len"]) for row in selected)
    single_count = sum(row["question_type"] == "single" for row in selected)
    multi_count = total - single_count
    lines = [
        "# GRPO Pilot v2 数据构造摘要",
        "",
        f"- input_path: `{input_path}`",
        f"- output_jsonl: `{output_path}`",
        "- prompt_version: `answer_only_v2`",
        f"- total: {total}",
        f"- single_count: {single_count}",
        f"- multi_count: {multi_count}",
        "",
        "## Target Plan",
        "",
        "| bucket | target | available | selected |",
        "|---|---:|---:|---:|",
    ]
    for bucket, target in TARGET_PLAN.items():
        lines.append(f"| {bucket} | {target} | {available_counts.get(bucket, 0)} | {selected_counts.get(bucket, 0)} |")
    lines.extend(
        [
            "",
            "## Answer Length Distribution",
            "",
            "```json",
            json.dumps(dict(sorted(answer_len_counts.items(), key=lambda kv: int(kv[0]))), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Skip Counts",
            "",
            "```json",
            json.dumps(dict(skip_counts), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Notes",
            "",
            "- 这是 GRPO prompt pool / reward-labeled prompt set，不是离线生成的模型 completion。",
            "- prompt 中不包含 gold answer，不包含 SFT assistant 输出，也不包含 teacher analysis。",
            "- v2 使用 answer-only prompt，只允许输出一行 `答案：X`，用于先做答案边界对齐。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_path = choose_input(args.input)
    raw_rows = load_jsonl(input_path)

    skip_counts: Counter[str] = Counter()
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(raw_rows):
        converted, reason = convert_row(row, idx, input_path)
        if converted is None:
            skip_counts[reason or "unknown"] += 1
            continue
        bucket = bucket_for(converted)
        if bucket not in TARGET_PLAN:
            skip_counts[f"bucket_not_targeted_{bucket}"] += 1
            continue
        buckets[bucket].append(converted)

    available_counts = Counter({bucket: len(rows) for bucket, rows in buckets.items()})
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for bucket, target in TARGET_PLAN.items():
        rows = list(buckets.get(bucket, []))
        rng.shuffle(rows)
        remaining = max(args.max_total - len(selected), 0)
        take = min(target, len(rows), remaining)
        selected.extend(rows[:take])
        selected_counts[bucket] = take
        if len(selected) >= args.max_total:
            break

    rng.shuffle(selected)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    write_jsonl(output_path, selected)
    write_summary(summary_path, input_path, output_path, selected, skip_counts, available_counts, selected_counts)
    print(f"wrote {len(selected)} rows to {output_path}")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
