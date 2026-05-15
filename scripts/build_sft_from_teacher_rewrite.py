#!/usr/bin/env python3
"""Build MedicalGPT SFT JSONL from accepted teacher rewrite outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = "data/medreason/day10_rewrite_pilot_outputs_v1.jsonl"
DEFAULT_OUTPUT_PATH = "data/sft/medreason_schema_sft_pilot_v1.jsonl"
DEFAULT_REJECTED_OUTPUT_PATH = "data/medreason/day10_rewrite_rejected_invalid_v1.jsonl"
DEFAULT_SUMMARY_PATH = "data/medreason/day10_sft_build_summary_v1.md"

SFT_USER_SUFFIX = (
    "请基于以上病例信息，按合法 JSON 输出结构化诊断推理结果。"
    "字段固定为 primary_diagnosis、diagnostic_basis、differential_diagnoses、"
    "recommended_actions、risk_flags。不要输出 Markdown 或额外解释。"
)

SCHEMA_FIELDS = (
    "primary_diagnosis",
    "diagnostic_basis",
    "differential_diagnoses",
    "recommended_actions",
    "risk_flags",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build schema-v1 SFT data from teacher rewrite outputs.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--rejected-output-path", default=DEFAULT_REJECTED_OUTPUT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Write only the conversations field. Default keeps metadata for traceability.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
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


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def nonempty_string_list(value: Any, min_len: int = 1) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= min_len
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def validate_schema_target(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return ["schema_target_not_object"]
    errors: list[str] = []
    if not nonempty_string(schema.get("primary_diagnosis")):
        errors.append("primary_diagnosis_empty")
    if not nonempty_string_list(schema.get("diagnostic_basis"), min_len=2):
        errors.append("diagnostic_basis_too_short")
    if not nonempty_string_list(schema.get("differential_diagnoses"), min_len=1):
        errors.append("differential_diagnoses_empty")
    if not nonempty_string_list(schema.get("recommended_actions"), min_len=1):
        errors.append("recommended_actions_empty")
    if not nonempty_string_list(schema.get("risk_flags"), min_len=1):
        errors.append("risk_flags_empty")
    for field in schema:
        if field not in SCHEMA_FIELDS:
            errors.append(f"unknown_schema_field:{field}")
    return errors


def normalize_schema_target(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_diagnosis": str(schema["primary_diagnosis"]).strip(),
        "diagnostic_basis": [str(item).strip() for item in schema["diagnostic_basis"] if str(item).strip()],
        "differential_diagnoses": [
            str(item).strip() for item in schema["differential_diagnoses"] if str(item).strip()
        ],
        "recommended_actions": [str(item).strip() for item in schema["recommended_actions"] if str(item).strip()],
        "risk_flags": [str(item).strip() for item in schema["risk_flags"] if str(item).strip()],
    }


def build_user_message(case_text: str) -> str:
    return f"{case_text.strip()}\n\n{SFT_USER_SUFFIX}"


def build_sft_row(row: dict[str, Any], minimal: bool) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    teacher_response = row.get("teacher_response")
    if not isinstance(teacher_response, dict):
        rejected = dict(row)
        rejected["sft_reject_reason"] = "teacher_response_not_object"
        return None, rejected
    if teacher_response.get("decision") != "accept":
        rejected = dict(row)
        rejected["sft_reject_reason"] = f"teacher_decision:{teacher_response.get('decision')}"
        return None, rejected

    case_text = teacher_response.get("case_text")
    schema_target = teacher_response.get("schema_target")
    if not nonempty_string(case_text):
        rejected = dict(row)
        rejected["sft_reject_reason"] = "case_text_empty"
        return None, rejected

    schema_errors = validate_schema_target(schema_target)
    if schema_errors:
        rejected = dict(row)
        rejected["sft_reject_reason"] = ",".join(schema_errors)
        return None, rejected

    normalized_schema = normalize_schema_target(schema_target)
    sft_row: dict[str, Any] = {
        "conversations": [
            {"from": "human", "value": build_user_message(case_text)},
            {"from": "gpt", "value": json.dumps(normalized_schema, ensure_ascii=False, indent=2)},
        ]
    }
    if not minimal:
        sft_row.update(
            {
                "sample_id": row.get("sample_id"),
                "route": row.get("route"),
                "source": row.get("source"),
                "case_text": case_text,
                "schema_target": normalized_schema,
                "quality_tags": teacher_response.get("quality_tags") or [],
                "model": row.get("model"),
            }
        )
    return sft_row, None


def write_summary(
    path: Path,
    input_path: str,
    output_path: str,
    rejected_output_path: str,
    sft_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final_status_counts = Counter(str(row.get("final_status")) for row in all_rows)
    route_counts = Counter(str(row.get("route")) for row in sft_rows)
    rejected_reasons = Counter(str(row.get("sft_reject_reason")) for row in rejected_rows)

    lines = [
        "# Day 10 SFT Build Summary v1",
        "",
        "## Inputs",
        f"- teacher_outputs: `{input_path}`",
        "",
        "## Outputs",
        f"- sft_jsonl: `{output_path}`",
        f"- rejected_or_invalid: `{rejected_output_path}`",
        f"- summary: `{path}`",
        "",
        "## Counts",
        f"- teacher_rows: {len(all_rows)}",
        f"- sft_rows: {len(sft_rows)}",
        f"- rejected_or_invalid_rows: {len(rejected_rows)}",
        "",
        "## Teacher Final Statuses",
        "| final_status | count |",
        "|---|---:|",
    ]
    for status, count in sorted(final_status_counts.items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(["", "## SFT Routes", "| route | count |", "|---|---:|"])
    for route, count in sorted(route_counts.items()):
        lines.append(f"| {route} | {count} |")

    lines.extend(["", "## Rejection Reasons", "| reason | count |", "|---|---:|"])
    for reason, count in sorted(rejected_reasons.items()):
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "## SFT Format",
            "- Output uses MedicalGPT ShareGPT format: `conversations: [{from: human}, {from: gpt}]`.",
            "- Assistant value is only schema_v1 JSON, without decision metadata.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    rows = read_jsonl(input_path)
    sft_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for row in rows:
        sft_row, rejected_row = build_sft_row(row, minimal=args.minimal)
        if sft_row is not None:
            sft_rows.append(sft_row)
        if rejected_row is not None:
            rejected_rows.append(rejected_row)

    write_jsonl(Path(args.output_path), sft_rows)
    write_jsonl(Path(args.rejected_output_path), rejected_rows)
    write_summary(
        Path(args.summary_path),
        args.input_path,
        args.output_path,
        args.rejected_output_path,
        sft_rows,
        rejected_rows,
        rows,
    )


if __name__ == "__main__":
    main()
