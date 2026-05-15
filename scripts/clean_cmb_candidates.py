#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean CMB-Exam candidate pool before SFT sampling.

The cleaner is intentionally conservative: it rejects only clear data errors
and exact normalized duplicates. Ambiguous cases, such as very short but valid
definition questions or empty non-answer options, are counted but retained.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


DEFAULT_INPUT = "data/raw/hf/CMB/CMB-Exam/CMB-train/CMB-train-merge.json"
DEFAULT_OUTPUT = "data/processed/cmb_clean/cmb_train_clean.jsonl"
DEFAULT_REJECTED = "data/processed/cmb_clean/cmb_train_rejected.jsonl"
DEFAULT_SUMMARY = "data/processed/cmb_clean/cmb_clean_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean CMB-Exam train candidates.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--rejected-output", default=DEFAULT_REJECTED)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--min-options", type=int, default=3)
    parser.add_argument("--max-question-length", type=int, default=800)
    parser.add_argument("--max-special-ratio", type=float, default=0.60)
    parser.add_argument("--max-examples", type=int, default=8)
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


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[，。、“”‘’：:；;,.!?？！（）()\[\]【】{}<>《》\"'`~\-—_·|/\\]", "", value)
    return value


def clean_text(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def options_signature(options: dict[str, Any]) -> str:
    return "|".join(f"{normalize_text(key)}={normalize_text(options[key])}" for key in sorted(options))


def normalized_signature(row: dict[str, Any]) -> str:
    return f"{normalize_text(row.get('question'))}||{options_signature(row.get('option') or {})}"


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def garbled_score(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    for ch in text:
        code = ord(ch)
        if ch == "\ufffd" or (code < 32 and ch not in "\n\t"):
            bad += 1
    return bad / len(text)


def special_ratio(text: str) -> float:
    if not text:
        return 0.0
    special = 0
    for ch in text:
        is_cjk = "\u4e00" <= ch <= "\u9fff"
        if not is_cjk and not ch.isalnum() and not ch.isspace():
            special += 1
    return special / len(text)


def normalize_answer(answer: Any, option_keys: set[str]) -> str:
    text = clean_text(answer).upper()
    if text in option_keys:
        return text
    chars = [ch for ch in text if ch in option_keys]
    if chars and "".join(chars) == text:
        return "".join(dict.fromkeys(chars))
    return ""


def sorted_options(options: dict[str, Any]) -> dict[str, str]:
    return {str(key).strip().upper(): clean_text(value) for key, value in sorted(options.items())}


def reject(row: dict[str, Any], reason: str, detail: str = "") -> dict[str, Any]:
    out = dict(row)
    out["reject_reason"] = reason
    out["reject_detail"] = detail
    return out


def validate_and_clean(row: dict[str, Any], args: argparse.Namespace, seen_signatures: set[str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    flags: list[str] = []

    question = clean_text(row.get("question"))
    if not question:
        return None, reject(row, "missing_question"), flags

    options_raw = row.get("option")
    if not isinstance(options_raw, dict):
        return None, reject(row, "option_not_dict"), flags

    options = sorted_options(options_raw)
    if len(options) < args.min_options:
        return None, reject(row, "options_too_few", f"option_count={len(options)}"), flags

    answer = clean_text(row.get("answer")).upper()
    if not answer:
        return None, reject(row, "missing_answer"), flags

    option_keys = set(options)
    normalized_answer = normalize_answer(answer, option_keys)
    if not normalized_answer:
        return None, reject(row, "invalid_answer", f"answer={answer}; option_keys={sorted(option_keys)}"), flags

    for answer_key in normalized_answer:
        if not options.get(answer_key, ""):
            return None, reject(row, "empty_correct_option", f"answer_key={answer_key}"), flags

    if garbled_score(question) > 0.02:
        return None, reject(row, "garbled_question"), flags

    if len(question) > args.max_question_length:
        return None, reject(row, "question_too_long", f"length={len(question)}"), flags

    if special_ratio(question) > args.max_special_ratio and len(question) > 20:
        return None, reject(row, "special_ratio_too_high", f"ratio={special_ratio(question):.3f}"), flags

    if len(question) < 5:
        flags.append("short_question_retained")

    empty_non_answer = [key for key, value in options.items() if not value and key not in normalized_answer]
    if empty_non_answer:
        flags.append("empty_non_answer_option_retained")

    cleaned = dict(row)
    cleaned["question"] = question
    cleaned["option"] = options
    cleaned["answer"] = normalized_answer
    cleaned["clean_flags"] = flags

    sig = digest(normalized_signature(cleaned))
    if sig in seen_signatures:
        return None, reject(cleaned, "duplicate_normalized_question_options"), flags
    seen_signatures.add(sig)

    return cleaned, None, flags


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    args: argparse.Namespace,
    total: int,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    reason_counts: Counter[str],
    flag_counts: Counter[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CMB Candidate Cleaning Summary",
        "",
        f"- Input: `{args.input}`",
        f"- Accepted output: `{args.output}`",
        f"- Rejected output: `{args.rejected_output}`",
        f"- Raw total: {total}",
        f"- Accepted: {len(accepted)} ({len(accepted) / total * 100:.2f}%)" if total else "- Accepted: 0",
        f"- Rejected: {len(rejected)} ({len(rejected) / total * 100:.2f}%)" if total else "- Rejected: 0",
        "",
        "## Reject Reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"| {reason} | {count} |")

    lines.extend(["", "## Retained Warning Flags", "", "| Flag | Count |", "|---|---:|"])
    for flag, count in flag_counts.most_common():
        lines.append(f"| {flag} | {count} |")

    lines.extend(["", "## Rejected Examples"])
    examples_by_reason: dict[str, list[dict[str, Any]]] = {}
    for row in rejected:
        examples_by_reason.setdefault(str(row.get("reject_reason")), []).append(row)
    for reason, rows in examples_by_reason.items():
        lines.extend(["", f"### {reason}"])
        for row in rows[: args.max_examples]:
            question = str(row.get("question") or "")[:160]
            answer = str(row.get("answer") or "")[:120]
            detail = str(row.get("reject_detail") or "")
            lines.append(f"- question={question}; answer={answer}; detail={detail}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_json_or_jsonl(Path(args.input))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    seen_signatures: set[str] = set()

    for row in rows:
        cleaned, rejected_row, flags = validate_and_clean(row, args, seen_signatures)
        flag_counts.update(flags)
        if rejected_row is not None:
            rejected.append(rejected_row)
            reason_counts[str(rejected_row.get("reject_reason"))] += 1
        elif cleaned is not None:
            accepted.append(cleaned)

    write_jsonl(Path(args.output), accepted)
    write_jsonl(Path(args.rejected_output), rejected)
    write_summary(Path(args.summary_output), args, len(rows), accepted, rejected, reason_counts, flag_counts)
    print(f"raw={len(rows)} accepted={len(accepted)} rejected={len(rejected)}")
    print(f"wrote {args.output}")
    print(f"wrote {args.rejected_output}")
    print(f"wrote {args.summary_output}")


if __name__ == "__main__":
    main()
