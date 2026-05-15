#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose duplicate and overlap rates in CMB SFT JSONL files.

This script performs low-cost exact/normalized duplicate checks without
embedding dependencies. It is intended as the first quality-control pass before
optional semantic deduplication.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


DEFAULT_DATASETS = {
    "baseline_train": "data/sft/cmb_baseline/cmb_sft_baseline.jsonl",
    "baseline_val": "data/sft/cmb_baseline/cmb_sft_baseline_val.jsonl",
    "quality_train": "data/sft/cmb_quality/cmb_sft_quality_only.jsonl",
    "quality_val": "data/sft/cmb_quality/cmb_sft_quality_only_val.jsonl",
    "optimized_train": "data/sft/cmb_optimized/cmb_sft_optimized.jsonl",
    "optimized_val": "data/sft/cmb_optimized/cmb_sft_optimized_val.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose duplicate rates for CMB SFT data.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec in name=path format. Can be repeated. Defaults to known CMB SFT files.",
    )
    parser.add_argument("--output-md", default="data/diagnostics/sft_duplicate_report.md")
    parser.add_argument("--output-json", default="data/diagnostics/sft_duplicate_report.json")
    parser.add_argument("--max-examples", type=int, default=5)
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
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[，。、“”‘’：:；;,.!?？！（）()\[\]【】{}<>《》\"'`~\-—_·|/\\]", "", value)
    return value


def options_signature(options: Any) -> str:
    if not isinstance(options, dict):
        return normalize_text(options)
    parts = []
    for key in sorted(options):
        parts.append(f"{normalize_text(key)}={normalize_text(options[key])}")
    return "|".join(parts)


def row_options(row: dict[str, Any]) -> Any:
    if "options" in row:
        return row.get("options")
    return row.get("option")


def raw_signature(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "")
    options = row_options(row) or {}
    return json.dumps({"q": question, "o": options}, ensure_ascii=False, sort_keys=True)


def normalized_signature(row: dict[str, Any]) -> str:
    return f"{normalize_text(row.get('question'))}||{options_signature(row_options(row))}"


def question_only_signature(row: dict[str, Any]) -> str:
    return normalize_text(row.get("question"))


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def duplicate_groups(rows: list[dict[str, Any]], sig_fn) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[digest(sig_fn(row))].append(row)
    return {key: group for key, group in groups.items() if len(group) > 1}


def summarize_groups(groups: dict[str, list[dict[str, Any]]], max_examples: int) -> dict[str, Any]:
    duplicate_rows = sum(len(group) - 1 for group in groups.values())
    examples = []
    for group in sorted(groups.values(), key=len, reverse=True)[:max_examples]:
        examples.append(
            {
                "count": len(group),
                "sample_ids": [row.get("sample_id") for row in group[:8]],
                "line_nos": [row.get("_line_no") for row in group[:8]],
                "question": str(group[0].get("question") or "")[:180],
                "answer_values": sorted({str(row.get("answer") or "") for row in group}),
            }
        )
    return {
        "group_count": len(groups),
        "duplicate_rows": duplicate_rows,
        "examples": examples,
    }


def dataset_summary(name: str, path: Path, rows: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    raw_groups = duplicate_groups(rows, raw_signature)
    norm_groups = duplicate_groups(rows, normalized_signature)
    question_groups = duplicate_groups(rows, question_only_signature)
    return {
        "name": name,
        "path": str(path),
        "rows": len(rows),
        "raw_exact": summarize_groups(raw_groups, max_examples),
        "normalized_exact": summarize_groups(norm_groups, max_examples),
        "question_only": summarize_groups(question_groups, max_examples),
    }


def overlap_summary(left_name: str, left_rows: list[dict[str, Any]], right_name: str, right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    right_norm = defaultdict(list)
    right_question = defaultdict(list)
    for row in right_rows:
        right_norm[digest(normalized_signature(row))].append(row)
        right_question[digest(question_only_signature(row))].append(row)

    norm_hits = []
    question_hits = []
    for row in left_rows:
        norm_key = digest(normalized_signature(row))
        question_key = digest(question_only_signature(row))
        if norm_key in right_norm:
            norm_hits.append((row, right_norm[norm_key][0]))
        if question_key in right_question:
            question_hits.append((row, right_question[question_key][0]))

    return {
        "left": left_name,
        "right": right_name,
        "left_rows": len(left_rows),
        "normalized_overlap_rows": len(norm_hits),
        "question_overlap_rows": len(question_hits),
        "normalized_examples": [
            {
                "left_sample_id": a.get("sample_id"),
                "right_sample_id": b.get("sample_id"),
                "question": str(a.get("question") or "")[:180],
            }
            for a, b in norm_hits[:5]
        ],
        "question_examples": [
            {
                "left_sample_id": a.get("sample_id"),
                "right_sample_id": b.get("sample_id"),
                "question": str(a.get("question") or "")[:180],
            }
            for a, b in question_hits[:5]
        ],
    }


def parse_dataset_specs(specs: list[str]) -> dict[str, str]:
    if not specs:
        return DEFAULT_DATASETS
    parsed = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --dataset spec: {spec}. Expected name=path.")
        name, path = spec.split("=", 1)
        parsed[name.strip()] = path.strip()
    return parsed


def pct(value: int, total: int) -> str:
    return f"{value / total * 100:.2f}%" if total else "0.00%"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SFT Duplicate Diagnosis", ""]
    lines.append("## Dataset Duplicate Rates")
    lines.append("")
    lines.append("| Dataset | Rows | Raw dup rows | Norm dup rows | Question-only dup rows |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in report["datasets"]:
        rows = item["rows"]
        lines.append(
            f"| {item['name']} | {rows} | "
            f"{item['raw_exact']['duplicate_rows']} ({pct(item['raw_exact']['duplicate_rows'], rows)}) | "
            f"{item['normalized_exact']['duplicate_rows']} ({pct(item['normalized_exact']['duplicate_rows'], rows)}) | "
            f"{item['question_only']['duplicate_rows']} ({pct(item['question_only']['duplicate_rows'], rows)}) |"
        )

    lines.append("")
    lines.append("## Train-Val Overlap")
    lines.append("")
    lines.append("| Pair | Left rows | Norm overlap | Question-only overlap |")
    lines.append("|---|---:|---:|---:|")
    for item in report["overlaps"]:
        left_rows = item["left_rows"]
        pair = f"{item['left']} -> {item['right']}"
        lines.append(
            f"| {pair} | {left_rows} | "
            f"{item['normalized_overlap_rows']} ({pct(item['normalized_overlap_rows'], left_rows)}) | "
            f"{item['question_overlap_rows']} ({pct(item['question_overlap_rows'], left_rows)}) |"
        )

    lines.append("")
    lines.append("## Example Duplicate Groups")
    for item in report["datasets"]:
        examples = item["normalized_exact"]["examples"]
        if not examples:
            continue
        lines.append("")
        lines.append(f"### {item['name']} normalized duplicates")
        for ex in examples:
            lines.append(f"- count={ex['count']}, sample_ids={ex['sample_ids']}, answers={ex['answer_values']}")
            lines.append(f"  - question: {ex['question']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    specs = parse_dataset_specs(args.dataset)
    loaded = {}
    dataset_reports = []
    for name, path_str in specs.items():
        path = Path(path_str)
        if not path.exists():
            print(f"skip missing dataset: {name} -> {path}")
            continue
        rows = read_jsonl(path)
        loaded[name] = rows
        dataset_reports.append(dataset_summary(name, path, rows, args.max_examples))

    overlaps = []
    for prefix in ("baseline", "quality", "optimized"):
        train_name = f"{prefix}_train"
        val_name = f"{prefix}_val"
        if train_name in loaded and val_name in loaded:
            overlaps.append(overlap_summary(train_name, loaded[train_name], val_name, loaded[val_name]))

    report = {
        "datasets": dataset_reports,
        "overlaps": overlaps,
        "notes": {
            "raw_exact": "Exact JSON signature over question and options.",
            "normalized_exact": "NFKC + lower + whitespace/punctuation removal over question and options.",
            "question_only": "Normalized question text only; this may over-count duplicates when options differ.",
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.output_md))
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
