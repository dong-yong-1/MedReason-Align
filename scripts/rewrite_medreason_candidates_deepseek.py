#!/usr/bin/env python3
"""Rewrite Day 9 selected MedReason candidates with DeepSeek V4.

Default mode is dry-run: it only prepares the pilot input file. Add
`--call-api` to call the OpenAI-compatible DeepSeek chat completions endpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import linecache
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib import error, request


DEFAULT_INPUT_PATH = "data/medreason/day9_selected_for_rewrite_sample50k_v1.jsonl"
DEFAULT_PILOT_INPUT_PATH = "data/medreason/day10_rewrite_pilot_input_v1.jsonl"
DEFAULT_OUTPUT_PATH = "data/medreason/day10_rewrite_pilot_outputs_v1.jsonl"
DEFAULT_SUMMARY_PATH = "data/medreason/day10_rewrite_pilot_summary_v1.md"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

ROUTE_LIMIT_DEFAULTS = {
    "structured_block_candidate": 74,
    "exam_rewrite_candidate": 20,
    "rewrite_candidate": 20,
}

SCHEMA_FIELDS = (
    "primary_diagnosis",
    "diagnostic_basis",
    "differential_diagnoses",
    "recommended_actions",
    "risk_flags",
)

SYSTEM_PROMPT = """你是一名严谨的中文医疗数据标注老师。你的任务是把候选医疗样本改写为结构化病例诊断推理 schema。

必须遵守：
1. 只基于输入材料归纳，不得编造输入中没有的年龄、性别、检查结果、既往史、体征或病程。
2. 如果病例信息不足，必须 reject，不要硬凑字段。
3. 诊断表达要保守，使用“可能性大”“首先考虑”“倾向于”等措辞，除非原文已明确确诊。
4. recommended_actions 以检查、线下就医、进一步评估、风险升级建议为主，不给危险处方或剂量。
5. risk_flags 必须具体说明什么情况需要急诊或尽快线下就医。
6. 只输出合法 JSON 对象，不要输出 Markdown，不要输出解释文字。
"""

USER_PROMPT_TEMPLATE = """请根据下面候选样本做 teacher rewrite。

输出必须是以下二选一 JSON 格式。

接受样本：
{{
  "decision": "accept",
  "reject_reason": "",
  "case_text": "保留或轻度清理后的病例输入，不要包含考试选项格式",
  "schema_target": {{
    "primary_diagnosis": "首要诊断方向",
    "diagnostic_basis": ["至少2条，必须能被输入材料支持"],
    "differential_diagnoses": ["至少1条"],
    "recommended_actions": ["至少1条"],
    "risk_flags": ["至少1条"]
  }},
  "quality_tags": ["case_like", "schema_complete"]
}}

拒绝样本：
{{
  "decision": "reject",
  "reject_reason": "简要说明拒绝原因",
  "case_text": "",
  "schema_target": null,
  "quality_tags": ["insufficient_case_context"]
}}

候选样本：
{candidate_json}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek V4 teacher rewrite for MedReason candidates.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--pilot-input-path", default=DEFAULT_PILOT_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--call-api", action="store_true", help="Actually call DeepSeek API. Default is dry-run.")
    parser.add_argument("--env-file", default=".env", help="Optional dotenv file to load before API calls.")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--thinking-mode",
        choices=("disabled", "enabled", "omit"),
        default=os.environ.get("DEEPSEEK_THINKING_MODE", "disabled"),
        help="DeepSeek V4 thinking mode. Use 'omit' to avoid sending this field.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--structured-limit", type=int, default=ROUTE_LIMIT_DEFAULTS["structured_block_candidate"])
    parser.add_argument("--exam-limit", type=int, default=ROUTE_LIMIT_DEFAULTS["exam_rewrite_candidate"])
    parser.add_argument("--rewrite-limit", type=int, default=ROUTE_LIMIT_DEFAULTS["rewrite_candidate"])
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Optional total target across rewrite routes. "
            "When set, rows are sampled round-robin from structured, exam, and rewrite candidates. "
            "0 keeps the small pilot route defaults."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already present in output.")
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without requiring python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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


def strip_prefix(text: str) -> str:
    text = str(text or "").strip()
    for prefix in ("问：", "问:", "答：", "答:", "Q:", "Q：", "A:", "A："):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def source_line_json(row: dict[str, Any]) -> dict[str, Any] | None:
    source = row.get("source") or {}
    path = source.get("path")
    line_no = source.get("line_no")
    if not path or not line_no:
        return None
    line = linecache.getline(str(path), int(line_no))
    if not line:
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def hydrate_raw_answer(row: dict[str, Any]) -> str:
    if row.get("raw_answer"):
        return str(row["raw_answer"])
    source = row.get("source") or {}
    dataset = source.get("dataset")
    source_row = source_line_json(row)
    if not source_row:
        return ""
    if dataset == "FreedomIntelligence/HuatuoGPT-sft-data-v1":
        data = source_row.get("data")
        if isinstance(data, list) and len(data) >= 2:
            return strip_prefix(data[1])
    if dataset == "shibing624/medical":
        return str(source_row.get("output") or "").strip()
    return ""


def route_limits(args: argparse.Namespace) -> dict[str, int]:
    return {
        "structured_block_candidate": args.structured_limit,
        "exam_rewrite_candidate": args.exam_limit,
        "rewrite_candidate": args.rewrite_limit,
    }


def choose_pilot_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    route_order = ("structured_block_candidate", "exam_rewrite_candidate", "rewrite_candidate")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        route = str(row.get("route") or "")
        if route in route_order:
            buckets[route].append(row)

    chosen: list[dict[str, Any]] = []
    if args.limit:
        indexes = {route: 0 for route in route_order}
        while len(chosen) < args.limit:
            added = False
            for route in route_order:
                idx = indexes[route]
                route_rows = buckets.get(route, [])
                if idx < len(route_rows):
                    chosen.append(route_rows[idx])
                    indexes[route] += 1
                    added = True
                    if len(chosen) >= args.limit:
                        break
            if not added:
                break
        return chosen

    limits = route_limits(args)
    for route in route_order:
        chosen.extend(buckets.get(route, [])[: limits[route]])
    return chosen


def candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "sample_id": row.get("sample_id"),
        "route": row.get("route"),
        "case_text": row.get("case_text"),
        "source": row.get("source"),
        "quality_flags": row.get("quality_flags") or [],
        "day9_scores": row.get("day9_scores"),
    }
    if row.get("raw_blocks"):
        payload["raw_blocks"] = row["raw_blocks"]
    if row.get("answer_text"):
        payload["answer_text"] = row["answer_text"]
    raw_answer = hydrate_raw_answer(row)
    if raw_answer:
        payload["raw_answer"] = raw_answer
    return payload


def build_prompt_record(row: dict[str, Any]) -> dict[str, Any]:
    payload = candidate_payload(row)
    candidate_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "sample_id": row.get("sample_id"),
        "route": row.get("route"),
        "source": row.get("source"),
        "candidate": payload,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(candidate_json=candidate_json)},
        ],
    }


def chat_completion_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def call_deepseek(prompt_record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} before using --call-api")

    payload = {
        "model": args.model,
        "messages": prompt_record["messages"],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    if args.thinking_mode != "omit":
        payload["thinking"] = {"type": args.thinking_mode}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    req = request.Request(chat_completion_url(args.base_url), data=data, headers=headers, method="POST")
    last_error = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            with request.urlopen(req, timeout=args.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt >= args.max_retries:
                break
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"DeepSeek API request failed after {args.max_retries} attempts: {last_error}")


def extract_message_content(api_response: dict[str, Any]) -> str:
    choices = api_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


def parse_json_content(content: str) -> dict[str, Any] | None:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def nonempty_string_list(value: Any, min_len: int = 1) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= min_len
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def validate_teacher_json(parsed: dict[str, Any] | None) -> tuple[str, list[str]]:
    if not isinstance(parsed, dict):
        return "invalid", ["not_json_object"]
    decision = parsed.get("decision")
    if decision == "reject":
        if not nonempty_string(parsed.get("reject_reason")):
            return "invalid", ["reject_missing_reason"]
        return "rejected_by_teacher", []
    if decision != "accept":
        return "invalid", ["decision_not_accept_or_reject"]

    schema = parsed.get("schema_target")
    if not isinstance(schema, dict):
        return "invalid", ["schema_target_not_object"]

    errors: list[str] = []
    if not nonempty_string(parsed.get("case_text")):
        errors.append("case_text_empty")
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
    return ("accepted" if not errors else "invalid"), errors


def processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if sample_id:
            ids.add(str(sample_id))
    return ids


def write_summary(
    path: Path,
    args: argparse.Namespace,
    prompt_records: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_routes = Counter(str(row.get("route")) for row in prompt_records)
    statuses = Counter(str(row.get("final_status")) for row in output_rows)
    output_routes = Counter(str(row.get("route")) for row in output_rows)
    lines = [
        "# Day 10 Teacher Rewrite Pilot Summary v1",
        "",
        "## Scope",
        "- Input: Day 9 selected candidates.",
        "- Teacher: DeepSeek V4 OpenAI-compatible API.",
        f"- Model: `{args.model}`",
        f"- API called: {'yes' if args.call_api else 'no'}",
        "",
        "## Outputs",
        f"- pilot_input: `{args.pilot_input_path}`",
        f"- rewrite_outputs: `{args.output_path}`",
        f"- summary: `{args.summary_path}`",
        "",
        "## Pilot Input Routes",
        "| route | count |",
        "|---|---:|",
    ]
    for route, count in sorted(prompt_routes.items()):
        lines.append(f"| {route} | {count} |")

    lines.extend(["", "## Output Statuses", "| status | count |", "|---|---:|"])
    for status, count in sorted(statuses.items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(["", "## Output Routes", "| route | count |", "|---|---:|"])
    for route, count in sorted(output_routes.items()):
        lines.append(f"| {route} | {count} |")

    lines.extend(
        [
            "",
            "## Rules",
            "- Accept only if the sample can support schema_v1 without inventing facts.",
            "- Reject pure knowledge QA, drug instruction, or samples with insufficient case context.",
            "- Keep benchmark-only data out of this rewrite flow.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    load_env_file(Path(".env"))
    args = parse_args()
    load_env_file(Path(args.env_file))
    selected_rows = read_jsonl(Path(args.input_path))
    pilot_rows = choose_pilot_rows(selected_rows, args)
    prompt_records = [build_prompt_record(row) for row in pilot_rows]
    write_jsonl(Path(args.pilot_input_path), prompt_records)

    output_rows: list[dict[str, Any]] = []
    if args.call_api:
        already_done = processed_ids(Path(args.output_path)) if args.resume else set()
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.resume and output_path.exists() else "w"
        with output_path.open(mode, encoding="utf-8") as f:
            for prompt_record in prompt_records:
                sample_id = str(prompt_record["sample_id"])
                if sample_id in already_done:
                    continue
                try:
                    api_response = call_deepseek(prompt_record, args)
                    content = extract_message_content(api_response)
                    parsed = parse_json_content(content)
                    status, validation_errors = validate_teacher_json(parsed)
                    out_row = {
                        "sample_id": sample_id,
                        "route": prompt_record.get("route"),
                        "source": prompt_record.get("source"),
                        "final_status": status,
                        "validation_errors": validation_errors,
                        "teacher_response": parsed,
                        "raw_content": content,
                        "model": args.model,
                    }
                except Exception as exc:  # Keep batch jobs resumable.
                    out_row = {
                        "sample_id": sample_id,
                        "route": prompt_record.get("route"),
                        "source": prompt_record.get("source"),
                        "final_status": "api_error",
                        "validation_errors": [str(exc)],
                        "teacher_response": None,
                        "raw_content": "",
                        "model": args.model,
                    }
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                f.flush()
                output_rows.append(out_row)
                time.sleep(args.sleep_seconds)
        if args.resume and Path(args.output_path).exists():
            output_rows = read_jsonl(Path(args.output_path))
    elif Path(args.output_path).exists():
        output_rows = read_jsonl(Path(args.output_path))

    write_summary(Path(args.summary_path), args, prompt_records, output_rows)


if __name__ == "__main__":
    main()
