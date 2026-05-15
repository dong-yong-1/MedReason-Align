#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek second-stage screening for CoT candidate questions.

Stage 1 uses local rules to rank CoT-suitable questions. This script performs
stage 2 teacher screening on a rule-filtered candidate slice, asking DeepSeek
to judge whether each item is truly worth explicit CoT supervision.

The script does not generate rationales. It only scores "CoT-worthiness" and
keeps enough audit metadata to compare rule-only selection vs. DeepSeek
selection later.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import statistics
import time
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_INPUT = "data/processed/cmb_clean/cmb_cot_candidates_scored.jsonl"
DEFAULT_AUDIT_OUTPUT = "data/teacher/cot_candidate_deepseek_audit.jsonl"
DEFAULT_SELECTED_OUTPUT = "data/processed/cmb_clean/cmb_cot_candidates_deepseek_selected.jsonl"
DEFAULT_SUMMARY_OUTPUT = "data/processed/cmb_clean/cmb_cot_candidates_deepseek_summary.md"


SYSTEM_PROMPT = """你是一名严谨的中文医学考试数据策划老师。

任务：判断一道医学考试题是否适合作为 CoT-style SFT 子集中的样本。

这里的“适合 CoT”不是指题目难不难，而是指它是否值得模型显式写出分析过程。
优先考虑：
1. 需要多步推理、比较、排除、鉴别或决策的题；
2. 病例信息较丰富，需要综合人群、症状、检查和病程的题；
3. 选项之间容易混淆、需要逐项分析的题；
4. 多选题。

降低评分：
1. 纯定义/概念识别题；
2. 纯知识点 recall 题；
3. 题干极短且没有病例信息、几乎不需要展开分析的题。

不要解题，不要输出答案解释，只评估“是否适合做 CoT 监督”。
只输出合法 JSON，不要输出 Markdown。
"""


USER_PROMPT_TEMPLATE = """请评估下面这道医学考试题是否适合作为 CoT-style SFT 子集样本。

输出 JSON，字段必须完整：
{{
  "decision": "keep" 或 "drop",
  "cot_worthiness_score": 0 到 10 的整数，
  "reasoning_need_score": 0 到 4 的整数，
  "case_richness_score": 0 到 2 的整数，
  "option_confusion_score": 0 到 2 的整数，
  "rote_recall_penalty": 0 到 2 的整数，
  "definition_penalty": 0 到 1 的整数，
  "reason": "40到120字的中文简述，说明为什么适合或不适合做CoT",
  "tags": ["从 case/multi_choice/option_confusing/definition/rote_recall/diagnosis/decision/exam_interpretation 中选若干个"]
}}

打分参考：
- reasoning_need_score：题目是否真的需要显式分析过程；
- case_richness_score：病例/检查/病程信息是否丰富；
- option_confusion_score：选项是否相近、需要区分；
- rote_recall_penalty：是否更像直接背知识点；
- definition_penalty：是否明显是定义/概念识别题。

题型：{question_type}
题目：{question}

选项：
{options_text}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek second-stage CoT candidate screening.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Stage-1 scored candidate JSONL.")
    parser.add_argument("--audit-output", default=DEFAULT_AUDIT_OUTPUT, help="Append-only DeepSeek audit JSONL.")
    parser.add_argument("--selected-output", default=DEFAULT_SELECTED_OUTPUT, help="Selected candidates after DeepSeek screening.")
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--prompt-output", default="", help="Optional prompt records JSONL for audit or dry run.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--candidate-top-k", type=int, default=3000, help="Screen only the top-k stage-1 candidates. 0 means all.")
    parser.add_argument("--select-top-k", type=int, default=1000, help="Keep top-k candidates after DeepSeek scoring. 0 means all valid rows.")
    parser.add_argument("--min-rule-score", type=float, default=None, help="Optional minimum stage-1 score before screening.")
    parser.add_argument("--min-deepseek-score", type=float, default=None, help="Optional minimum DeepSeek score for final selection.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit after offset/top-k filtering, for pilot runs.")
    parser.add_argument("--call-api", action="store_true", help="Actually call DeepSeek. Default only writes prompts and selected rows from existing audit.")
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already present in audit-output.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N API attempts. 0 disables progress logs.")
    parser.add_argument("--max-consecutive-errors", type=int, default=20, help="Stop early after this many consecutive request/runtime errors. 0 disables.")
    parser.add_argument(
        "--thinking-mode",
        choices=("disabled", "enabled", "omit"),
        default=os.environ.get("DEEPSEEK_THINKING_MODE", "disabled"),
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
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
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: Any) -> str:
    return str(text or "").strip()


def options_text(options: dict[str, Any]) -> str:
    return "\n".join(f"{key}. {clean_text(options[key])}" for key in sorted(options))


def extract_options(row: dict[str, Any]) -> dict[str, str]:
    options = row.get("options") if "options" in row else row.get("option")
    if not isinstance(options, dict):
        return {}
    return {str(key).strip().upper(): clean_text(value) for key, value in sorted(options.items())}


def sample_id(row: dict[str, Any], idx: int) -> str:
    if row.get("sample_id") is not None:
        return str(row.get("sample_id"))
    if row.get("id") is not None:
        return str(row.get("id"))
    # Keep backward compatibility with already-written audit rows that used
    # zero-based rank position inside the top-k candidate slice.
    if row.get("cot_candidate_rank") is not None:
        try:
            return str(int(row["cot_candidate_rank"]) - 1)
        except (TypeError, ValueError):
            pass
    return str(idx)


def build_prompt_record(row: dict[str, Any], idx: int) -> dict[str, Any]:
    options = extract_options(row)
    prompt = USER_PROMPT_TEMPLATE.format(
        question_type=clean_text(row.get("question_type")),
        question=clean_text(row.get("question")),
        options_text=options_text(options),
    )
    return {
        "sample_id": sample_id(row, idx),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }


def chat_completion_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def call_deepseek(prompt_record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} before using --call-api")
    payload: dict[str, Any] = {
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

    last_error = ""
    for attempt in range(1, args.max_retries + 1):
        req = request.Request(chat_completion_url(args.base_url), data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=args.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
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


def normalize_tag(value: str) -> str:
    return re.sub(r"[^a-z_]+", "", value.strip().lower())


def clamp_int(parsed: dict[str, Any], key: str, minimum: int, maximum: int, errors: list[str]) -> int:
    raw = parsed.get(key)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        errors.append(f"{key} is not an int")
        return minimum
    if value < minimum or value > maximum:
        errors.append(f"{key} out of range: {value}")
    return max(minimum, min(value, maximum))


def validate_screening(parsed: dict[str, Any] | None) -> tuple[str, list[str], dict[str, Any]]:
    errors: list[str] = []
    normalized: dict[str, Any] = {}
    if not isinstance(parsed, dict):
        return "invalid", ["response is not a JSON object"], normalized

    decision = clean_text(parsed.get("decision")).lower()
    if decision not in {"keep", "drop"}:
        errors.append(f"invalid decision: {decision}")
    normalized["decision"] = decision if decision in {"keep", "drop"} else "drop"

    normalized["cot_worthiness_score"] = clamp_int(parsed, "cot_worthiness_score", 0, 10, errors)
    normalized["reasoning_need_score"] = clamp_int(parsed, "reasoning_need_score", 0, 4, errors)
    normalized["case_richness_score"] = clamp_int(parsed, "case_richness_score", 0, 2, errors)
    normalized["option_confusion_score_model"] = clamp_int(parsed, "option_confusion_score", 0, 2, errors)
    normalized["rote_recall_penalty"] = clamp_int(parsed, "rote_recall_penalty", 0, 2, errors)
    normalized["definition_penalty_model"] = clamp_int(parsed, "definition_penalty", 0, 1, errors)

    reason = clean_text(parsed.get("reason"))
    if len(reason) < 10:
        errors.append("reason too short")
    normalized["reason"] = reason

    raw_tags = parsed.get("tags")
    tags: list[str] = []
    if not isinstance(raw_tags, list):
        errors.append("tags is not a list")
    else:
        for value in raw_tags:
            tag = normalize_tag(str(value))
            if tag and tag not in tags:
                tags.append(tag)
    normalized["tags"] = tags

    derived_score = (
        normalized["reasoning_need_score"]
        + normalized["case_richness_score"]
        + normalized["option_confusion_score_model"]
        - normalized["rote_recall_penalty"]
        - normalized["definition_penalty_model"]
    )
    normalized["deepseek_cot_score"] = derived_score

    if errors:
        return "invalid", errors, normalized
    return "ok", [], normalized


def processed_ids(path: Path) -> tuple[set[str], Counter[str]]:
    ids: set[str] = set()
    status_counts: Counter[str] = Counter()
    for row in read_jsonl(path):
        sample = row.get("sample_id")
        status = str(row.get("final_status") or "unknown")
        status_counts[status] += 1
        # Resume should skip terminal rows, but retry transient request/runtime failures.
        if sample and status in {"ok", "invalid"}:
            ids.add(str(sample))
    return ids, status_counts


def audit_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row.get("sample_id")): row for row in rows if row.get("sample_id")}


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(
    run_stats: Counter[str],
    pending_total: int,
    start_time: float,
    prefix: str = "progress",
) -> None:
    attempted = run_stats.get("attempted", 0)
    completed = run_stats.get("ok", 0) + run_stats.get("invalid", 0) + run_stats.get("request_error", 0)
    elapsed = max(time.time() - start_time, 1e-6)
    rate = attempted / elapsed if attempted > 0 else 0.0
    remaining = max(pending_total - attempted, 0)
    eta = remaining / rate if rate > 0 else 0.0
    print(
        f"[{prefix}] attempted={attempted}/{pending_total} completed={completed} "
        f"ok={run_stats.get('ok', 0)} invalid={run_stats.get('invalid', 0)} "
        f"errors={run_stats.get('request_error', 0)} skipped_resume={run_stats.get('skipped_resume', 0)} "
        f"rate={rate:.2f}/s elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
    )


def filter_candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    filtered = rows
    if args.min_rule_score is not None:
        filtered = [row for row in filtered if float(row.get("cot_candidate_score", 0.0)) >= args.min_rule_score]
    if args.candidate_top_k > 0:
        filtered = filtered[: args.candidate_top_k]
    if args.offset > 0:
        filtered = filtered[args.offset :]
    if args.limit > 0:
        filtered = filtered[: args.limit]
    return filtered


def merge_selected_row(source_row: dict[str, Any], audit_row: dict[str, Any]) -> dict[str, Any]:
    merged = dict(source_row)
    merged["sample_id"] = audit_row.get("sample_id") or source_row.get("sample_id")
    merged["deepseek_decision"] = audit_row.get("decision")
    merged["deepseek_cot_score"] = audit_row.get("deepseek_cot_score")
    merged["deepseek_cot_worthiness_score"] = audit_row.get("cot_worthiness_score")
    merged["deepseek_reasoning_need_score"] = audit_row.get("reasoning_need_score")
    merged["deepseek_case_richness_score"] = audit_row.get("case_richness_score")
    merged["deepseek_option_confusion_score"] = audit_row.get("option_confusion_score_model")
    merged["deepseek_rote_recall_penalty"] = audit_row.get("rote_recall_penalty")
    merged["deepseek_definition_penalty"] = audit_row.get("definition_penalty_model")
    merged["deepseek_reason"] = audit_row.get("reason")
    merged["deepseek_tags"] = audit_row.get("tags", [])
    merged["deepseek_model"] = audit_row.get("model")
    merged["deepseek_status"] = audit_row.get("final_status")
    return merged


def select_rows(
    candidate_rows: list[dict[str, Any]],
    audit_rows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for idx, row in enumerate(candidate_rows):
        sid = sample_id(row, idx)
        audit_row = audit_rows.get(sid)
        if not audit_row or audit_row.get("final_status") != "ok":
            continue
        merged = merge_selected_row(row, audit_row)
        if args.min_deepseek_score is not None and float(merged.get("deepseek_cot_score", -999)) < args.min_deepseek_score:
            continue
        selected.append(merged)

    selected.sort(
        key=lambda row: (
            float(row.get("deepseek_cot_score", -999)),
            float(row.get("deepseek_cot_worthiness_score", -999)),
            float(row.get("cot_candidate_score", -999)),
        ),
        reverse=True,
    )
    for rank, row in enumerate(selected, start=1):
        row["deepseek_cot_rank"] = rank
    if args.select_top_k > 0:
        selected = selected[: args.select_top_k]
    return selected


def write_summary(
    path: Path,
    args: argparse.Namespace,
    candidate_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    audit_rows: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = [row for row in audit_rows.values() if row.get("final_status") == "ok"]
    invalid = [row for row in audit_rows.values() if row.get("final_status") != "ok"]
    scores = [float(row.get("deepseek_cot_score", 0.0)) for row in valid]
    decisions = Counter(str(row.get("decision")) for row in valid)
    tag_counts = Counter()
    for row in valid:
        tag_counts.update(row.get("tags", []))

    lines = [
        "# DeepSeek CoT Candidate Screening Summary",
        "",
        f"- Input: `{args.input}`",
        f"- Candidate rows considered: {len(candidate_rows)}",
        f"- Audit rows total: {len(audit_rows)}",
        f"- Valid audit rows: {len(valid)}",
        f"- Invalid audit rows: {len(invalid)}",
        f"- Selected rows: {len(selected_rows)}",
        f"- candidate_top_k: {args.candidate_top_k}",
        f"- select_top_k: {args.select_top_k}",
        f"- min_rule_score: {args.min_rule_score}",
        f"- min_deepseek_score: {args.min_deepseek_score}",
        "",
        "## DeepSeek Score Distribution",
        "",
    ]

    if scores:
        lines.extend(
            [
                f"- min: {min(scores):.3f}",
                f"- p50: {statistics.median(scores):.3f}",
                f"- mean: {statistics.mean(scores):.3f}",
                f"- max: {max(scores):.3f}",
            ]
        )
    else:
        lines.extend(["- min: 0.000", "- p50: 0.000", "- mean: 0.000", "- max: 0.000"])

    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"- keep: {decisions.get('keep', 0)}",
            f"- drop: {decisions.get('drop', 0)}",
            "",
            "## Top Tags",
            "",
            "| Tag | Count |",
            "|---|---:|",
        ]
    )
    for tag, count in tag_counts.most_common(12):
        lines.append(f"| {tag} | {count} |")

    lines.extend(["", "## Top Selected Examples"])
    for row in selected_rows[:10]:
        question = clean_text(row.get("question"))[:180]
        reason = clean_text(row.get("deepseek_reason"))[:140]
        lines.append(
            f"- deepseek_score={row.get('deepseek_cot_score')}; rule_score={row.get('cot_candidate_score')}; question={question}; reason={reason}"
        )

    if invalid:
        lines.extend(["", "## Invalid Audit Examples"])
        for row in invalid[:8]:
            sample = row.get("sample_id")
            errors = "; ".join(row.get("validation_errors", [])[:4])
            lines.append(f"- sample_id={sample}; errors={errors}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file))

    input_rows = read_jsonl(Path(args.input))
    candidate_rows = filter_candidates(input_rows, args)

    if args.prompt_output:
        prompt_path = Path(args.prompt_output)
        prompt_rows = [build_prompt_record(row, idx) for idx, row in enumerate(candidate_rows)]
        write_jsonl(prompt_path, prompt_rows)

    audit_path = Path(args.audit_output)
    seen_ids, existing_status_counts = processed_ids(audit_path) if args.resume else (set(), Counter())

    pending_rows: list[tuple[int, dict[str, Any], str, dict[str, Any]]] = []
    skipped_resume = 0
    for idx, row in enumerate(candidate_rows):
        sid = sample_id(row, idx)
        if sid in seen_ids:
            skipped_resume += 1
            continue
        prompt_record = build_prompt_record(row, idx)
        pending_rows.append((idx, row, sid, prompt_record))

    print(
        f"[setup] candidates={len(candidate_rows)} resume_skipped={skipped_resume} "
        f"pending={len(pending_rows)} existing_audit_status={dict(existing_status_counts)}"
    )

    run_stats: Counter[str] = Counter()
    run_stats["skipped_resume"] = skipped_resume
    start_time = time.time()
    consecutive_errors = 0

    for idx, row, sid, prompt_record in pending_rows:
        if not args.call_api:
            break

        run_stats["attempted"] += 1
        try:
            api_response = call_deepseek(prompt_record, args)
            content = extract_message_content(api_response)
            parsed = parse_json_content(content)
            final_status, validation_errors, normalized = validate_screening(parsed)
            audit_row = {
                "sample_id": sid,
                "cot_candidate_rank": row.get("cot_candidate_rank"),
                "cot_candidate_score": row.get("cot_candidate_score"),
                "question": row.get("question"),
                "question_type": row.get("question_type"),
                "model": args.model,
                "prompt_record": prompt_record,
                "raw_content": content,
                "parsed": parsed,
                "final_status": final_status,
                "validation_errors": validation_errors,
            }
            audit_row.update(normalized)
            append_jsonl(audit_path, audit_row)
            run_stats[final_status] += 1
            consecutive_errors = 0
        except KeyboardInterrupt:
            print("[interrupt] interrupted by user; writing current selection and summary from existing audit.")
            break
        except Exception as exc:
            error_row = {
                "sample_id": sid,
                "cot_candidate_rank": row.get("cot_candidate_rank"),
                "cot_candidate_score": row.get("cot_candidate_score"),
                "question": row.get("question"),
                "question_type": row.get("question_type"),
                "model": args.model,
                "prompt_record": prompt_record,
                "final_status": "request_error",
                "validation_errors": [f"{type(exc).__name__}: {exc}"],
            }
            append_jsonl(audit_path, error_row)
            run_stats["request_error"] += 1
            consecutive_errors += 1

            if args.progress_every > 0:
                print(
                    f"[error] sample_id={sid} status=request_error consecutive_errors={consecutive_errors} "
                    f"detail={type(exc).__name__}: {exc}"
                )
            if args.max_consecutive_errors > 0 and consecutive_errors >= args.max_consecutive_errors:
                print(
                    f"[stop] hit max_consecutive_errors={args.max_consecutive_errors}; "
                    "stopping early and writing current selection/summary."
                )
                break
        finally:
            if args.progress_every > 0 and run_stats["attempted"] % args.progress_every == 0:
                print_progress(run_stats, len(pending_rows), start_time)

        time.sleep(args.sleep_seconds)

    audit_rows = audit_map(audit_path)
    selected_rows = select_rows(candidate_rows, audit_rows, args)
    write_jsonl(Path(args.selected_output), selected_rows)
    write_summary(Path(args.summary_output), args, candidate_rows, selected_rows, audit_rows)

    if args.call_api:
        print_progress(run_stats, len(pending_rows), start_time, prefix="done")
    print(
        f"candidates={len(candidate_rows)} audited={len(audit_rows)} selected={len(selected_rows)} "
        f"selected_output={args.selected_output}"
    )
    print(f"summary={args.summary_output}")


if __name__ == "__main__":
    main()
