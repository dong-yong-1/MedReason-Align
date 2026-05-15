#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build direct-answer SFT data from CMB SFT JSONL files.

The transformation keeps the user prompt and metadata unchanged, and replaces
the assistant response with only the answer label, e.g. "答案：A".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CMB SFT data to direct-answer format.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    return parser.parse_args()


def normalize_answer(value: Any, option_keys: set[str]) -> str:
    answer = str(value or "").strip().upper()
    if answer in option_keys:
        return answer
    letters = [ch for ch in answer if ch in option_keys]
    if letters and all(ch in option_keys for ch in letters):
        return "".join(dict.fromkeys(letters))
    return ""


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    options = row.get("options") or {}
    option_keys = {str(key).strip().upper() for key in options if str(key).strip()}
    answer = normalize_answer(row.get("answer"), option_keys)
    if not answer:
        raise ValueError(f"Missing or invalid answer for sample_id={row.get('sample_id')}")

    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError(f"Invalid conversations for sample_id={row.get('sample_id')}")

    converted = dict(row)
    converted["conversations"] = [dict(turn) for turn in conversations]
    converted["conversations"][-1]["from"] = "gpt"
    converted["conversations"][-1]["value"] = f"答案：{answer}"
    converted["answer"] = answer
    converted["sft_format"] = "direct_answer"
    return converted


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    converted_rows = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            try:
                converted_rows.append(convert_row(row))
            except ValueError:
                skipped += 1

    with output_path.open("w", encoding="utf-8") as f:
        for row in converted_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"converted {len(converted_rows)} / {total} rows -> {output_path}; skipped={skipped}")


if __name__ == "__main__":
    main()
