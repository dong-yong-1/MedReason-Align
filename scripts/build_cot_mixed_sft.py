#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a Direct + CoT mixed SFT dataset for controlled comparison."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed-size Direct + CoT mixed SFT dataset.")
    parser.add_argument("--direct-train", required=True, help="Direct-answer SFT train JSONL.")
    parser.add_argument("--cot-train", required=True, help="Teacher CoT SFT train JSONL.")
    parser.add_argument("--val", required=True, help="Validation JSONL, usually the direct-answer val set.")
    parser.add_argument("--output-dir", required=True, help="Output directory for train/val files.")
    parser.add_argument("--target-total", type=int, default=5000, help="Target mixed train size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-name", default="cmb_sft_cot_mixed.jsonl")
    parser.add_argument("--val-name", default="cmb_sft_cot_mixed_val.jsonl")
    parser.add_argument("--summary-name", default="cmb_sft_cot_mixed_summary.md")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
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


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("question") or "")


def assistant_text(row: dict[str, Any]) -> str:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ""
    return str(conversations[-1].get("value") or "")


def source_format(row: dict[str, Any]) -> str:
    fmt = str(row.get("sft_format") or "")
    if fmt:
        return fmt
    text = assistant_text(row)
    if "分析：" in text and "答案：" in text:
        return "cot_teacher_deepseek"
    if text.startswith("答案："):
        return "direct_answer"
    return "unknown"


def distribution(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    return Counter(str(row.get(field) or "unknown") for row in rows)


def format_counter(counter: Counter[str], total: int) -> list[str]:
    lines = []
    for key, count in counter.most_common(12):
        pct = count / total * 100 if total else 0.0
        lines.append(f"| {key} | {count} | {pct:.2f}% |")
    return lines


def write_summary(path: Path, args: argparse.Namespace, train_rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]], cot_rows: list[dict[str, Any]]) -> None:
    total = len(train_rows)
    format_counts = Counter(source_format(row) for row in train_rows)
    cot_count = format_counts.get("cot_teacher_deepseek", 0)
    direct_count = format_counts.get("direct_answer", 0)
    lines = [
        "# CMB Direct + CoT Mixed SFT Summary",
        "",
        "## Inputs",
        f"- direct_train: `{args.direct_train}`",
        f"- cot_train: `{args.cot_train}`",
        f"- val: `{args.val}`",
        "",
        "## Sampling",
        f"- seed: {args.seed}",
        f"- target_total: {args.target_total}",
        f"- direct_selected: {len(direct_rows)}",
        f"- cot_selected: {len(cot_rows)}",
        f"- mixed_total: {total}",
        f"- cot_ratio: {cot_count / total * 100:.2f}%" if total else "- cot_ratio: 0.00%",
        "",
        "## Output Format",
        "| format | count | ratio |",
        "|---|---:|---:|",
        f"| direct_answer | {direct_count} | {direct_count / total * 100:.2f}% |" if total else "| direct_answer | 0 | 0.00% |",
        f"| cot_teacher_deepseek | {cot_count} | {cot_count / total * 100:.2f}% |" if total else "| cot_teacher_deepseek | 0 | 0.00% |",
        "",
        "## Question Type",
        "| question_type | count | ratio |",
        "|---|---:|---:|",
        *format_counter(distribution(train_rows, "question_type"), total),
        "",
        "## Top Exam Subjects",
        "| exam_subject | count | ratio |",
        "|---|---:|---:|",
        *format_counter(distribution(train_rows, "exam_subject"), total),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    direct_all = read_jsonl(Path(args.direct_train))
    cot_all = read_jsonl(Path(args.cot_train))
    val_rows = read_jsonl(Path(args.val))

    cot_count = min(len(cot_all), args.target_total)
    direct_count = max(args.target_total - cot_count, 0)
    if direct_count > len(direct_all):
        raise ValueError(f"Need {direct_count} direct rows, but only found {len(direct_all)}")

    cot_rows = list(cot_all[:cot_count])
    cot_ids = {row_id(row) for row in cot_rows}
    direct_pool = [row for row in direct_all if row_id(row) not in cot_ids]
    if direct_count > len(direct_pool):
        raise ValueError(f"Need {direct_count} non-overlapping direct rows, but only found {len(direct_pool)}")

    direct_rows = rng.sample(direct_pool, direct_count)
    train_rows = direct_rows + cot_rows
    rng.shuffle(train_rows)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / args.train_name, train_rows)
    write_jsonl(output_dir / args.val_name, val_rows)
    write_summary(output_dir / args.summary_name, args, train_rows, direct_rows, cot_rows)

    print(
        f"wrote train={len(train_rows)} direct={len(direct_rows)} cot={len(cot_rows)} "
        f"val={len(val_rows)} -> {output_dir}"
    )


if __name__ == "__main__":
    main()
