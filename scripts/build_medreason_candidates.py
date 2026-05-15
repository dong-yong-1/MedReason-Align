#!/usr/bin/env python3
"""Build Day 8 MedReason candidate-pool artifacts.

Day 8 is about unifying raw public datasets into one candidate protocol. It is
not the teacher-rewrite step and does not claim that raw QA data is already
valid schema-v1 supervision.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable


DEFAULT_HUATUO_PATH = "data/raw/hf/HuatuoGPT-sft-data-v1/HuatuoGPT_sft_data_v1.jsonl"
DEFAULT_SHIBING_PATH = "data/raw/hf/shibing624-medical/finetune/train_zh_0.json"
DEFAULT_CMB_CLIN_PATH = "data/raw/hf/CMB/CMB-Clin/CMB-Clin-qa.json"
DEFAULT_CMB_EXAM_TRAIN_PATH = "data/raw/hf/CMB/CMB-Exam/CMB-train/CMB-train-merge.json"
DEFAULT_CMB_EXAM_VAL_PATH = "data/raw/hf/CMB/CMB-Exam/CMB-val/CMB-val-merge.json"
DEFAULT_CMB_EXAM_TEST_PATH = "data/raw/hf/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json"

DEFAULT_TRAIN_OUTPUT = "data/medreason/day8_train_candidates_v1.jsonl"
DEFAULT_BENCHMARK_OUTPUT = "data/medreason/day8_benchmark_only_v1.jsonl"
DEFAULT_REVIEW_OUTPUT = "data/medreason/day8_manual_review_sample_v1.jsonl"
DEFAULT_SUMMARY_OUTPUT = "data/medreason/day8_summary_v1.md"

ROUTE_REWRITE = "rewrite_candidate"
ROUTE_STRUCTURED_BLOCK = "structured_block_candidate"
ROUTE_EXAM_REWRITE = "exam_rewrite_candidate"
ROUTE_BENCHMARK = "benchmark_only"

SYMPTOM_HINTS = (
    "患者",
    "病人",
    "主诉",
    "现病史",
    "体格检查",
    "辅助检查",
    "发热",
    "疼痛",
    "咳",
    "呕吐",
    "腹",
    "胸",
    "头痛",
)
GENERIC_QA_HINTS = ("怎么办", "严重吗", "吃什么药", "用法用量", "简介", "是什么")
IMAGE_DEPENDENCY_HINTS = ("图片", "照片", "片子", "报告单", "上传")
SAFETY_HINTS = ("自杀", "去死", "不想活", "生不如死")
DIAGNOSIS_HINTS = ("诊断", "诊断依据", "诊断要点", "最可能")
DIFFERENTIAL_HINTS = ("鉴别", "相鉴别")
ACTION_HINTS = ("治疗", "处理", "检查", "决策", "原则")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Day 8 MedReason candidate-pool files.")
    parser.add_argument("--huatuo-path", default=DEFAULT_HUATUO_PATH)
    parser.add_argument("--shibing-path", default=DEFAULT_SHIBING_PATH)
    parser.add_argument("--cmb-clin-path", default=DEFAULT_CMB_CLIN_PATH)
    parser.add_argument("--cmb-exam-train-path", default=DEFAULT_CMB_EXAM_TRAIN_PATH)
    parser.add_argument("--cmb-exam-val-path", default=DEFAULT_CMB_EXAM_VAL_PATH)
    parser.add_argument("--cmb-exam-test-path", default=DEFAULT_CMB_EXAM_TEST_PATH)
    parser.add_argument("--out-train-candidates", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--out-benchmark-only", default=DEFAULT_BENCHMARK_OUTPUT)
    parser.add_argument("--out-review-sample", default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument("--out-summary", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--max-huatuo", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--max-shibing", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--max-cmb-clin", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--max-cmb-exam-train", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--max-cmb-exam-val", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--max-cmb-exam-test", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--sample-per-route", type=int, default=5)
    parser.add_argument(
        "--include-raw-answers",
        action="store_true",
        help="Include Huatuo/shibing raw answers. Default keeps output compact and traceable by source index.",
    )
    parser.add_argument(
        "--no-dedupe-case-text",
        action="store_true",
        help="Disable exact case_text hash deduplication for train candidates.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_prefix(text: str, prefixes: Iterable[str]) -> str:
    cleaned = normalize_text(text)
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            return normalize_text(cleaned[len(prefix):])
    return cleaned


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[tuple[int, dict[str, Any]]]:
    emitted = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield line_no, row
            emitted += 1
            if limit and emitted >= limit:
                return


def load_json_array(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    if limit:
        data = data[:limit]
    for idx, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{idx} is not a JSON object")
    return data


def infer_input_style(text: str, default: str = "patient_narrative") -> str:
    if "现病史" in text or "体格检查" in text or "辅助检查" in text:
        return "semi_structured_case"
    if re.search(r"[A-E][.．、]", text):
        return "exam_case"
    return default


def quality_flags(case_text: str) -> list[str]:
    flags: list[str] = []
    if len(case_text) < 12:
        flags.append("too_short")
    if "�" in case_text:
        flags.append("has_replacement_character")
    if any(hint in case_text for hint in IMAGE_DEPENDENCY_HINTS):
        flags.append("may_depend_on_image_or_attachment")
    if any(hint in case_text for hint in SAFETY_HINTS):
        flags.append("safety_boundary_signal")
    if len(case_text) < 40 and any(hint in case_text for hint in GENERIC_QA_HINTS):
        flags.append("generic_medical_qa")
    if not any(hint in case_text for hint in SYMPTOM_HINTS):
        flags.append("weak_case_context")
    return flags


def build_record(
    sample_id: str,
    split: str,
    route: str,
    case_text: str,
    source: dict[str, Any],
    meta: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case_text_norm = normalize_text(case_text)
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "route": route,
        "case_text": case_text_norm,
        "source": source,
        "meta": meta,
        "quality_flags": quality_flags(case_text_norm),
        "audit": {
            "case_text_hash": text_hash(case_text_norm),
            "case_text_chars": len(case_text_norm),
        },
    }
    if extra:
        record.update(extra)
    return record


def classify_cmb_clin_blocks(qas: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    blocks = {
        "diagnosis": [],
        "differential": [],
        "treatment_or_action": [],
        "other": [],
    }
    for qa in qas:
        question = normalize_text(qa.get("question"))
        answer = normalize_text(qa.get("answer"))
        if not question and not answer:
            continue
        block = {"question": question, "answer": answer}
        joined = f"{question}\n{answer}"
        if any(hint in joined for hint in DIFFERENTIAL_HINTS):
            blocks["differential"].append(block)
        elif any(hint in joined for hint in DIAGNOSIS_HINTS):
            blocks["diagnosis"].append(block)
        elif any(hint in joined for hint in ACTION_HINTS):
            blocks["treatment_or_action"].append(block)
        else:
            blocks["other"].append(block)
    return blocks


def cmb_exam_case_text(row: dict[str, Any]) -> str:
    question = normalize_text(row.get("question"))
    option = row.get("option", {})
    if not isinstance(option, dict):
        return question
    option_lines = [f"{key}. {normalize_text(value)}" for key, value in sorted(option.items())]
    return "\n".join([question, *option_lines]).strip()


def cmb_exam_answer_text(row: dict[str, Any]) -> str:
    answer_key = normalize_text(row.get("answer"))
    option = row.get("option", {})
    if isinstance(option, dict) and answer_key in option:
        return f"{answer_key}. {normalize_text(option[answer_key])}"
    return answer_key


class Day8Writer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stats: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()
        self.route_counts: Counter[str] = Counter()
        self.quality_counts: Counter[str] = Counter()
        self.review_samples: list[dict[str, Any]] = []
        self.sample_buckets: Counter[str] = Counter()
        self.seen_hashes: set[str] = set()

    def should_skip_train_record(self, record: dict[str, Any]) -> bool:
        case_hash = record["audit"]["case_text_hash"]
        if not self.args.no_dedupe_case_text and case_hash in self.seen_hashes:
            self.stats["skipped_duplicate_case_text"] += 1
            return True
        self.seen_hashes.add(case_hash)
        if "too_short" in record["quality_flags"] or "has_replacement_character" in record["quality_flags"]:
            self.stats["skipped_invalid_quality"] += 1
            return True
        return False

    def track(self, record: dict[str, Any]) -> None:
        source_name = record["source"]["dataset"]
        route = record["route"]
        self.source_counts[source_name] += 1
        self.route_counts[route] += 1
        for flag in record["quality_flags"]:
            self.quality_counts[flag] += 1
        sample_key = f"{source_name}|{route}"
        if self.sample_buckets[sample_key] < self.args.sample_per_route:
            self.review_samples.append(record)
            self.sample_buckets[sample_key] += 1

    def write_train_record(self, f, record: dict[str, Any]) -> None:
        if self.should_skip_train_record(record):
            return
        self.track(record)
        self.stats["train_candidates_written"] += 1
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_benchmark_record(self, f, record: dict[str, Any]) -> None:
        self.track(record)
        self.stats["benchmark_only_written"] += 1
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def emit_huatuo(writer: Day8Writer, f) -> None:
    for line_no, row in iter_jsonl(Path(writer.args.huatuo_path), limit=writer.args.max_huatuo):
        writer.stats["huatuo_input"] += 1
        data = row.get("data")
        if not isinstance(data, list) or len(data) < 2:
            writer.stats["huatuo_invalid"] += 1
            continue
        question = remove_prefix(data[0], ("问：", "问:", "Q:", "Q："))
        answer = remove_prefix(data[1], ("答：", "答:", "A:", "A："))
        extra = {
            "raw_answer_included": bool(writer.args.include_raw_answers),
            "raw_answer_ref": "data[1]",
        }
        if writer.args.include_raw_answers:
            extra["raw_answer"] = answer
        record = build_record(
            sample_id=f"huatuo_{line_no:07d}",
            split="train_candidate",
            route=ROUTE_REWRITE,
            case_text=question,
            source={
                "dataset": "FreedomIntelligence/HuatuoGPT-sft-data-v1",
                "subset": "default",
                "path": writer.args.huatuo_path,
                "line_no": line_no,
            },
            meta={
                "input_style": infer_input_style(question),
                "candidate_role": "raw_medical_qa_for_rewrite",
                "quality_tier": "bronze",
                "review_status": "unreviewed",
            },
            extra=extra,
        )
        writer.write_train_record(f, record)


def emit_shibing(writer: Day8Writer, f) -> None:
    for line_no, row in iter_jsonl(Path(writer.args.shibing_path), limit=writer.args.max_shibing):
        writer.stats["shibing_input"] += 1
        instruction = normalize_text(row.get("instruction"))
        input_text = normalize_text(row.get("input"))
        output = normalize_text(row.get("output"))
        case_text = instruction if not input_text else f"{instruction}\n\n{input_text}"
        if not case_text:
            writer.stats["shibing_invalid"] += 1
            continue
        extra = {
            "raw_answer_included": bool(writer.args.include_raw_answers),
            "raw_answer_ref": "output",
        }
        if writer.args.include_raw_answers:
            extra["raw_answer"] = output
        record = build_record(
            sample_id=f"shibing_{line_no:07d}",
            split="train_candidate",
            route=ROUTE_REWRITE,
            case_text=case_text,
            source={
                "dataset": "shibing624/medical",
                "subset": "finetune/train_zh_0",
                "path": writer.args.shibing_path,
                "line_no": line_no,
            },
            meta={
                "input_style": infer_input_style(case_text),
                "candidate_role": "raw_medical_qa_for_rewrite",
                "quality_tier": "bronze",
                "review_status": "unreviewed",
            },
            extra=extra,
        )
        writer.write_train_record(f, record)


def emit_cmb_clin(writer: Day8Writer, f) -> None:
    rows = load_json_array(Path(writer.args.cmb_clin_path), limit=writer.args.max_cmb_clin)
    for idx, row in enumerate(rows, start=1):
        writer.stats["cmb_clin_input"] += 1
        description = normalize_text(row.get("description"))
        qas = row.get("QA_pairs", [])
        if not description or not isinstance(qas, list):
            writer.stats["cmb_clin_invalid"] += 1
            continue
        raw_blocks = classify_cmb_clin_blocks(qas)
        record = build_record(
            sample_id=f"cmb_clin_{idx:05d}",
            split="train_candidate",
            route=ROUTE_STRUCTURED_BLOCK,
            case_text=description,
            source={
                "dataset": "FreedomIntelligence/CMB",
                "subset": "CMB-Clin",
                "path": writer.args.cmb_clin_path,
                "record_index": idx,
                "record_id": normalize_text(row.get("id")),
                "title": normalize_text(row.get("title")),
            },
            meta={
                "input_style": "semi_structured_case",
                "candidate_role": "structured_blocks_for_teacher_rewrite",
                "quality_tier": "silver",
                "review_status": "unreviewed",
            },
            extra={"raw_blocks": raw_blocks},
        )
        writer.write_train_record(f, record)


def emit_cmb_exam_train(writer: Day8Writer, f) -> None:
    rows = load_json_array(Path(writer.args.cmb_exam_train_path), limit=writer.args.max_cmb_exam_train)
    for idx, row in enumerate(rows, start=1):
        writer.stats["cmb_exam_train_input"] += 1
        case_text = cmb_exam_case_text(row)
        if not case_text:
            writer.stats["cmb_exam_train_invalid"] += 1
            continue
        record = build_record(
            sample_id=f"cmb_exam_train_{idx:07d}",
            split="train_candidate",
            route=ROUTE_EXAM_REWRITE,
            case_text=case_text,
            source={
                "dataset": "FreedomIntelligence/CMB",
                "subset": "CMB-Exam/train",
                "path": writer.args.cmb_exam_train_path,
                "record_index": idx,
                "exam_type": normalize_text(row.get("exam_type")),
                "exam_class": normalize_text(row.get("exam_class")),
                "exam_subject": normalize_text(row.get("exam_subject")),
                "question_type": normalize_text(row.get("question_type")),
            },
            meta={
                "input_style": "exam_case",
                "candidate_role": "exam_case_for_rewrite_or_weak_supervision",
                "quality_tier": "bronze",
                "review_status": "unreviewed",
            },
            extra={
                "answer_key": normalize_text(row.get("answer")),
                "answer_text": cmb_exam_answer_text(row),
            },
        )
        writer.write_train_record(f, record)


def emit_cmb_exam_benchmark(writer: Day8Writer, f, path_value: str, subset: str, limit: int) -> None:
    rows = load_json_array(Path(path_value), limit=limit)
    for idx, row in enumerate(rows, start=1):
        writer.stats[f"{subset}_input"] += 1
        case_text = cmb_exam_case_text(row)
        if not case_text:
            writer.stats[f"{subset}_invalid"] += 1
            continue
        record = build_record(
            sample_id=f"{subset.replace('/', '_')}_{idx:07d}",
            split="benchmark_only",
            route=ROUTE_BENCHMARK,
            case_text=case_text,
            source={
                "dataset": "FreedomIntelligence/CMB",
                "subset": subset,
                "path": path_value,
                "record_index": idx,
                "exam_type": normalize_text(row.get("exam_type")),
                "exam_class": normalize_text(row.get("exam_class")),
                "exam_subject": normalize_text(row.get("exam_subject")),
                "question_type": normalize_text(row.get("question_type")),
            },
            meta={
                "input_style": "exam_case",
                "candidate_role": "held_out_public_benchmark",
                "quality_tier": "benchmark",
                "review_status": "locked_out_of_training",
            },
            extra={
                "answer_key": normalize_text(row.get("answer")),
                "answer_text": cmb_exam_answer_text(row),
            },
        )
        writer.write_benchmark_record(f, record)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, args: argparse.Namespace, writer: Day8Writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Day 8 Candidate Pool Summary v1",
        "",
        "## Scope",
        "- Goal: unify raw public medical datasets into traceable train candidates for Day 9 similarity filtering.",
        "- Non-goal: produce final schema_target labels or teacher-rewritten SFT data.",
        "- LLM API used: no.",
        "",
        "## Outputs",
        f"- train_candidates: `{args.out_train_candidates}`",
        f"- benchmark_only: `{args.out_benchmark_only}`",
        f"- manual_review_sample: `{args.out_review_sample}`",
        f"- summary: `{args.out_summary}`",
        "",
        "## Counts",
        f"- train_candidates_written: {writer.stats.get('train_candidates_written', 0)}",
        f"- benchmark_only_written: {writer.stats.get('benchmark_only_written', 0)}",
        f"- skipped_duplicate_case_text: {writer.stats.get('skipped_duplicate_case_text', 0)}",
        f"- skipped_invalid_quality: {writer.stats.get('skipped_invalid_quality', 0)}",
        "",
        "## Routes",
        "| route | count |",
        "|---|---:|",
    ]
    for route, count in sorted(writer.route_counts.items()):
        lines.append(f"| {route} | {count} |")

    lines.extend(["", "## Sources", "| source | count |", "|---|---:|"])
    for source, count in sorted(writer.source_counts.items()):
        lines.append(f"| {source} | {count} |")

    lines.extend(["", "## Quality Flags", "| flag | count |", "|---|---:|"])
    for flag, count in sorted(writer.quality_counts.items()):
        lines.append(f"| {flag} | {count} |")

    lines.extend(["", "## Raw Input Stats", "| key | count |", "|---|---:|"])
    for key, count in sorted(writer.stats.items()):
        lines.append(f"| {key} | {count} |")

    lines.extend(
        [
            "",
            "## Day 8 Completion Check",
            "- All training candidates use a unified top-level protocol: sample_id, split, route, case_text, source, meta, quality_flags, audit.",
            "- CMB-Clin is preserved as structured raw blocks for later teacher rewrite, not treated as final schema labels.",
            "- HuatuoGPT and shibing624/medical are preserved as rewrite candidates with source pointers.",
            "- CMB-Exam train is separated from CMB-Exam val/test; val/test are locked into benchmark_only.",
            "- Exact duplicate train case_text values are removed unless --no-dedupe-case-text is set.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    Path(args.out_train_candidates).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_benchmark_only).parent.mkdir(parents=True, exist_ok=True)

    writer = Day8Writer(args)

    with Path(args.out_train_candidates).open("w", encoding="utf-8") as train_f:
        emit_cmb_clin(writer, train_f)
        emit_huatuo(writer, train_f)
        emit_shibing(writer, train_f)
        emit_cmb_exam_train(writer, train_f)

    with Path(args.out_benchmark_only).open("w", encoding="utf-8") as benchmark_f:
        emit_cmb_exam_benchmark(
            writer,
            benchmark_f,
            args.cmb_exam_val_path,
            "CMB-Exam/val",
            args.max_cmb_exam_val,
        )
        emit_cmb_exam_benchmark(
            writer,
            benchmark_f,
            args.cmb_exam_test_path,
            "CMB-Exam/test",
            args.max_cmb_exam_test,
        )

    write_jsonl(Path(args.out_review_sample), writer.review_samples)
    write_summary(Path(args.out_summary), args, writer)


if __name__ == "__main__":
    main()
