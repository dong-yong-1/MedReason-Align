#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build CoT-style CMB SFT data with DeepSeek teacher rationales.

The teacher is only asked to explain the provided gold answer. It must not
change the label. Output rows keep the original prompt, options, answer, and
metadata, replacing the assistant message with "分析：...\n\n答案：X".
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"

SYSTEM_PROMPT = """你是一名严谨的中文医学考试解析老师。

任务：根据题目、选项和已给定的标准答案，写一段简洁、可信的解析。

必须遵守：
1. 标准答案已经给定，不得修改答案，不得质疑答案。
2. 解析只基于题干、选项和常识性医学考试知识，不得编造题目中没有的病例信息。
3. 单选题说明为什么正确选项最合适，并简要排除关键干扰项。
4. 多选题说明所有正确选项为什么成立，并指出未选项为什么不符合。
5. 解析控制在 80-220 个中文字符，避免啰嗦。
6. 只输出合法 JSON，不要输出 Markdown。
"""

USER_PROMPT_TEMPLATE = """请为下面医学考试题生成解析。

输出 JSON 格式：
{{
  "analysis": "简洁中文解析，不要包含最终答案标签",
  "answer": "{answer}"
}}

题型：{question_type}
题目：{question}

选项：
{options_text}

标准答案：{answer}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CoT-style CMB SFT data with DeepSeek.")
    parser.add_argument("--input", required=True, help="Input CMB SFT JSONL.")
    parser.add_argument("--output", required=True, help="Output CoT-style SFT JSONL.")
    parser.add_argument("--teacher-output", required=True, help="Raw teacher output JSONL for resume/audit.")
    parser.add_argument("--prompt-output", default="", help="Optional prompt records JSONL for dry-run inspection.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--max-consecutive-errors", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="Limit rows for pilot generation. 0 means all rows.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--call-api", action="store_true", help="Actually call DeepSeek. Default only writes prompts.")
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already in teacher-output.")
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


def normalize_answer(value: Any, option_keys: set[str]) -> str:
    answer = str(value or "").strip().upper()
    if answer in option_keys:
        return answer
    letters = [ch for ch in answer if ch in option_keys]
    if letters:
        return "".join(dict.fromkeys(letters))
    return ""


def get_options(row: dict[str, Any]) -> dict[str, Any]:
    options = row.get("options") or row.get("option") or {}
    return options if isinstance(options, dict) else {}


def options_text(options: dict[str, Any]) -> str:
    return "\n".join(f"{key}. {str(options[key]).strip()}" for key in sorted(options))


def build_prompt_record(row: dict[str, Any]) -> dict[str, Any]:
    options = get_options(row)
    option_keys = {str(key).strip().upper() for key in options if str(key).strip()}
    answer = normalize_answer(row.get("answer"), option_keys)
    if not answer:
        raise ValueError(f"Invalid answer for sample_id={row.get('sample_id')}")
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question_type=row.get("question_type", ""),
        question=row.get("question", ""),
        options_text=options_text(options),
        answer=answer,
    )
    return {
        "sample_id": row.get("sample_id"),
        "answer": answer,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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


def validate_teacher(parsed: dict[str, Any] | None, answer: str) -> tuple[str, list[str], str]:
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return "invalid", ["response is not a JSON object"], ""
    analysis = str(parsed.get("analysis") or "").strip()
    teacher_answer = str(parsed.get("answer") or "").strip().upper()
    if teacher_answer != answer:
        errors.append(f"teacher answer changed: {teacher_answer} != {answer}")
    if not analysis:
        errors.append("missing analysis")
    if "答案：" in analysis or "标准答案" in analysis:
        errors.append("analysis contains answer label")
    if len(analysis) < 20:
        errors.append("analysis too short")
    return ("ok" if not errors else "invalid"), errors, analysis


def processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("sample_id")
            if sample_id:
                ids.add(str(sample_id))
    return ids


def teacher_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path) if path.exists() else []
    return {str(row.get("sample_id")): row for row in rows if row.get("final_status") == "ok"}


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def build_sft_row(row: dict[str, Any], teacher_row: dict[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    options = get_options(row)
    answer = teacher_row["answer"]
    analysis = str(teacher_row["analysis"]).strip()
    conversations = row.get("conversations")
    if isinstance(conversations, list) and len(conversations) >= 2:
        converted["conversations"] = [dict(turn) for turn in conversations]
    else:
        converted["conversations"] = [
            {
                "from": "human",
                "value": (
                    "请回答以下医学考试题，先给出简洁分析，再给出答案选项。\n\n"
                    f"题目：{row.get('question', '')}\n\n"
                    f"选项：\n{options_text(options)}"
                ),
            },
            {"from": "gpt", "value": ""},
        ]
    converted["conversations"][-1]["from"] = "gpt"
    converted["conversations"][-1]["value"] = f"分析：{analysis}\n\n答案：{answer}"
    converted["options"] = options
    converted.pop("option", None)
    converted["answer"] = answer
    converted["sft_format"] = "cot_teacher_deepseek"
    converted["teacher_model"] = teacher_row.get("model")
    return converted


def main() -> None:
    load_env_file(Path(".env"))
    args = parse_args()
    load_env_file(Path(args.env_file))

    rows = read_jsonl(Path(args.input))
    if args.offset:
        rows = rows[args.offset :]
    if args.limit:
        rows = rows[: args.limit]

    prompt_records: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        try:
            prompt_records.append(build_prompt_record(row))
        except ValueError:
            skipped += 1

    if args.prompt_output:
        write_jsonl(Path(args.prompt_output), prompt_records)

    teacher_path = Path(args.teacher_output)
    if args.call_api:
        done = processed_ids(teacher_path) if args.resume else set()
        if not args.resume and teacher_path.exists():
            teacher_path.unlink()
        pending = [record for record in prompt_records if str(record["sample_id"]) not in done]
        start_time = time.time()
        consecutive_errors = 0
        for index, prompt_record in enumerate(pending, start=1):
            sample_id = str(prompt_record["sample_id"])
            try:
                api_response = call_deepseek(prompt_record, args)
                content = extract_message_content(api_response)
                parsed = parse_json_content(content)
                status, validation_errors, analysis = validate_teacher(parsed, prompt_record["answer"])
                out_row = {
                    "sample_id": sample_id,
                    "final_status": status,
                    "validation_errors": validation_errors,
                    "analysis": analysis,
                    "answer": prompt_record["answer"],
                    "raw_content": content,
                    "model": args.model,
                }
            except Exception as exc:
                status = "api_error"
                validation_errors = [str(exc)]
                consecutive_errors += 1
                out_row = {
                    "sample_id": sample_id,
                    "final_status": status,
                    "validation_errors": validation_errors,
                    "analysis": "",
                    "answer": prompt_record["answer"],
                    "raw_content": "",
                    "model": args.model,
                }
            else:
                if status == "api_error":
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0
            append_jsonl(teacher_path, out_row)
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(pending)):
                elapsed = time.time() - start_time
                speed = index / elapsed if elapsed > 0 else 0.0
                eta = (len(pending) - index) / speed if speed > 0 else 0.0
                print(
                    f"[cot-teacher] {index}/{len(pending)} "
                    f"status={out_row['final_status']} speed={speed:.2f}/s eta={format_eta(eta)}",
                    flush=True,
                )
            if consecutive_errors >= args.max_consecutive_errors:
                raise RuntimeError(
                    f"Stopped after {consecutive_errors} consecutive API/validation errors. "
                    f"Last error: {validation_errors}"
                )
            time.sleep(args.sleep_seconds)

    teachers = teacher_by_id(teacher_path)
    output_rows = [build_sft_row(row, teachers[str(row.get("sample_id"))]) for row in rows if str(row.get("sample_id")) in teachers]
    write_jsonl(Path(args.output), output_rows)
    print(
        f"prompts={len(prompt_records)} skipped_input={skipped} "
        f"teacher_ok={len(teachers)} wrote_sft={len(output_rows)} -> {args.output}"
    )


if __name__ == "__main__":
    main()
