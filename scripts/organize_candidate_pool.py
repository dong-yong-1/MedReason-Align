#!/usr/bin/env python3
"""Organize MedReason train candidates with Day 6 cleaning rules v1."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any
import unicodedata


SCHEMA_FIELDS = (
    "primary_diagnosis",
    "diagnostic_basis",
    "differential_diagnoses",
    "recommended_actions",
    "risk_flags",
)

QUALITY_TIER_SCORE = {"gold": 3, "silver": 2, "bronze": 1}
COMPLETENESS_SCORE = {
    "complete": 4,
    "mostly_complete": 3,
    "partially_missing": 2,
    "critically_missing": 1,
}
OVERCONFIDENT_TERMS = ("肯定", "一定", "绝对", "无需检查即可确诊")
RECOMMENDATION_HINTS = (
    "检查",
    "就医",
    "急诊",
    "评估",
    "影像",
    "心电图",
    "复查",
    "转诊",
    "住院",
)
URGENT_HINTS = ("急诊", "立即", "尽快", "急救", "当天")
GENERIC_QUESTION_HINTS = ("怎么办", "吃什么药", "严重吗", "要不要紧")
VAGUE_BASIS_HINTS = ("根据患者情况", "结合临床", "综合考虑", "情况考虑")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize MedReason candidate pool with Day 6 rules v1."
    )
    parser.add_argument(
        "--input-paths",
        nargs="+",
        required=True,
        help="One or more candidate JSONL files using the MedReason sample structure.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for accepted/rejected/manual-review outputs.",
    )
    parser.add_argument(
        "--target-dev-path",
        default="data/medreason/target_dev_seed_v1.jsonl",
        help="Locked target_dev JSONL used for exact leakage blocking.",
    )
    parser.add_argument(
        "--holdout-path",
        default="",
        help="Optional final_holdout JSONL used for exact leakage blocking.",
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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalize_list(items: Any) -> list[str] | None:
    if not isinstance(items, list):
        return None
    normalized = [normalize_text(item) for item in items]
    return [item for item in normalized if item]


def normalize_schema_target(schema_target: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(schema_target, dict):
        return None, ["R-QUAL-001"]

    normalized: dict[str, Any] = {}
    hit_rules: list[str] = []

    primary = normalize_text(schema_target.get("primary_diagnosis"))
    if not primary:
        hit_rules.append("R-QUAL-001")
    normalized["primary_diagnosis"] = primary

    for field in SCHEMA_FIELDS[1:]:
        values = normalize_list(schema_target.get(field))
        if values is None:
            hit_rules.append("R-QUAL-001")
            values = []
        normalized[field] = values

    if hit_rules:
        return None, sorted(set(hit_rules))
    return normalized, []


def normalize_schema_target_text(schema_target: dict[str, Any]) -> str:
    return json.dumps(schema_target, ensure_ascii=False, sort_keys=True)


def canonical_diagnosis(text: str) -> str:
    normalized = normalize_text(text)
    for phrase in ("可能性大", "首先考虑", "倾向于", "倾向", "可能", "待排", "？", "?"):
        normalized = normalized.replace(phrase, "")
    return re.sub(r"\s+", "", normalized)


def has_noise(case_text_norm: str) -> bool:
    if "�" in case_text_norm:
        return True
    ascii_chars = sum(1 for char in case_text_norm if char.isascii() and char.isalnum())
    return ascii_chars > 0 and ascii_chars / max(len(case_text_norm), 1) > 0.6


def recommendation_is_informative(actions: list[str]) -> bool:
    action_text = " ".join(actions)
    return any(keyword in action_text for keyword in RECOMMENDATION_HINTS)


def needs_urgent_escalation(text: str) -> bool:
    return any(keyword in text for keyword in URGENT_HINTS)


def build_candidate(raw: dict[str, Any], fallback_idx: int) -> dict[str, Any]:
    candidate = dict(raw)
    sample_id = normalize_text(candidate.get("sample_id")) or f"cand_{fallback_idx:06d}"
    split = normalize_text(candidate.get("split")) or "train_candidate"
    case_text = normalize_text(candidate.get("case_text"))
    meta = candidate.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    schema_target, schema_hit_rules = normalize_schema_target(candidate.get("schema_target"))
    schema_target_norm = normalize_schema_target_text(schema_target) if schema_target else ""
    joint_text_norm = f"{case_text}\n{schema_target_norm}".strip() if case_text and schema_target_norm else ""
    quality_score = (
        QUALITY_TIER_SCORE.get(normalize_text(meta.get("quality_tier")), 0),
        COMPLETENESS_SCORE.get(normalize_text(meta.get("completeness_level")), 0),
        len(case_text),
        len(schema_target_norm),
    )

    candidate["sample_id"] = sample_id
    candidate["split"] = split
    candidate["case_text"] = candidate.get("case_text", "")
    candidate["meta"] = meta
    candidate["audit"] = {
        "case_text_norm": case_text,
        "schema_target_norm": schema_target_norm,
        "joint_text_norm": joint_text_norm,
        "quality_score": list(quality_score),
    }
    if schema_target is not None:
        candidate["schema_target"] = schema_target
    if schema_hit_rules:
        candidate["audit"]["precheck_rules"] = schema_hit_rules
    return candidate


def quality_review(candidate: dict[str, Any]) -> tuple[str, list[str]]:
    audit = candidate["audit"]
    case_text_norm = audit["case_text_norm"]
    schema_target = candidate.get("schema_target")
    meta = candidate.get("meta", {})

    if not case_text_norm or len(case_text_norm) < 18:
        return "rejected_low_quality", ["R-QUAL-002"]

    if has_noise(case_text_norm):
        return "rejected_low_quality", ["R-QUAL-008"]

    if "precheck_rules" in audit:
        return "rejected_low_quality", list(audit["precheck_rules"])

    if not isinstance(schema_target, dict):
        return "rejected_low_quality", ["R-QUAL-001"]

    primary = schema_target["primary_diagnosis"]
    basis = schema_target["diagnostic_basis"]
    differentials = schema_target["differential_diagnoses"]
    actions = schema_target["recommended_actions"]
    risk_flags = schema_target["risk_flags"]
    risk_level = normalize_text(meta.get("risk_level"))
    completeness = normalize_text(meta.get("completeness_level"))

    if len(basis) < 2 or any(item in VAGUE_BASIS_HINTS or len(item) < 4 for item in basis):
        return "rejected_low_quality", ["R-QUAL-005"]

    if not differentials:
        return "rejected_low_quality", ["R-QUAL-006"]

    primary_canonical = canonical_diagnosis(primary)
    if primary_canonical and any(canonical_diagnosis(item) == primary_canonical for item in differentials):
        return "needs_manual_review", ["R-QUAL-006", "R-REVIEW-005"]

    if risk_level in {"urgent", "emergent"} and not risk_flags:
        return "rejected_low_quality", ["R-QUAL-004"]

    if not actions:
        return "rejected_low_quality", ["R-QUAL-007"]

    if risk_level in {"urgent", "emergent"} and not recommendation_is_informative(actions):
        return "rejected_low_quality", ["R-QUAL-007"]

    action_and_risk_text = " ".join(actions + risk_flags)
    if risk_level in {"urgent", "emergent"} and not needs_urgent_escalation(action_and_risk_text):
        return "needs_manual_review", ["R-REVIEW-004"]

    if completeness in {"partially_missing", "critically_missing"} and any(
        term in primary for term in OVERCONFIDENT_TERMS
    ):
        return "rejected_low_quality", ["R-QUAL-009"]

    if (
        len(case_text_norm) < 32
        and any(token in case_text_norm for token in GENERIC_QUESTION_HINTS)
        and "患者" not in case_text_norm
        and "病例" not in case_text_norm
    ):
        return "rejected_low_quality", ["R-QUAL-010"]

    return "accepted", []


def build_exact_blocklist(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()

    blocked = set()
    for row in read_jsonl(path):
        candidate = build_candidate(row, fallback_idx=0)
        joint_text_norm = candidate["audit"]["joint_text_norm"]
        if joint_text_norm:
            blocked.add(joint_text_norm)
    return blocked


def attach_decision(
    candidate: dict[str, Any],
    final_status: str,
    hit_rules: list[str],
    extra_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = dict(candidate)
    audit = dict(record["audit"])
    if extra_audit:
        audit.update(extra_audit)
    record["audit"] = audit
    record["final_status"] = final_status
    record["hit_rules"] = sorted(set(hit_rules))
    return record


def generate_summary(
    output_path: Path,
    input_paths: list[str],
    rows: list[dict[str, Any]],
    target_dev_path: str,
    holdout_path: str,
) -> None:
    status_counter = Counter(row["final_status"] for row in rows)
    rule_counter = Counter(rule for row in rows for rule in row.get("hit_rules", []))

    lines = [
        "# candidate_pool_summary",
        "",
        "## Inputs",
        f"- input_paths: {', '.join(input_paths)}",
        f"- target_dev_path: {target_dev_path or 'none'}",
        f"- holdout_path: {holdout_path or 'none'}",
        "",
        "## Final Status Counts",
        f"- total: {len(rows)}",
        f"- accepted: {status_counter.get('accepted', 0)}",
        f"- needs_manual_review: {status_counter.get('needs_manual_review', 0)}",
        f"- rejected_duplicate: {status_counter.get('rejected_duplicate', 0)}",
        f"- rejected_leakage: {status_counter.get('rejected_leakage', 0)}",
        f"- rejected_low_quality: {status_counter.get('rejected_low_quality', 0)}",
        "",
        "## Rule Hits",
        "| rule | count |",
        "|---|---:|",
    ]

    for rule, count in sorted(rule_counter.items()):
        lines.append(f"| {rule} | {count} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def organize_candidates(
    input_paths: list[Path],
    target_dev_path: Path | None,
    holdout_path: Path | None,
) -> list[dict[str, Any]]:
    raw_candidates: list[dict[str, Any]] = []
    for input_path in input_paths:
        for row in read_jsonl(input_path):
            raw_candidates.append(build_candidate(row, fallback_idx=len(raw_candidates) + 1))

    blocked_by_target_dev = build_exact_blocklist(target_dev_path)
    blocked_by_holdout = build_exact_blocklist(holdout_path)

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, candidate in enumerate(raw_candidates):
        joint_text_norm = candidate["audit"]["joint_text_norm"]
        if joint_text_norm:
            grouped[joint_text_norm].append(idx)

    results: list[dict[str, Any] | None] = [None] * len(raw_candidates)

    for indices in grouped.values():
        if len(indices) < 2:
            continue
        winner_idx = max(indices, key=lambda idx: tuple(raw_candidates[idx]["audit"]["quality_score"]))
        for idx in indices:
            if idx == winner_idx:
                continue
            results[idx] = attach_decision(
                raw_candidates[idx],
                "rejected_duplicate",
                ["R-DUP-001"],
                {"duplicate_of": raw_candidates[winner_idx]["sample_id"]},
            )

    for idx, candidate in enumerate(raw_candidates):
        if results[idx] is not None:
            continue

        joint_text_norm = candidate["audit"]["joint_text_norm"]
        leakage_rules: list[str] = []
        leakage_source = ""
        if joint_text_norm and joint_text_norm in blocked_by_holdout:
            leakage_rules = ["R-LEAK-002"]
            leakage_source = "final_holdout"
        elif joint_text_norm and joint_text_norm in blocked_by_target_dev:
            leakage_rules = ["R-LEAK-004"]
            leakage_source = "target_dev"

        quality_status, quality_rules = quality_review(candidate)
        if leakage_rules:
            results[idx] = attach_decision(
                candidate,
                "rejected_leakage",
                leakage_rules,
                {"blocked_by": leakage_source},
            )
        else:
            results[idx] = attach_decision(candidate, quality_status, quality_rules)

    return [row for row in results if row is not None]


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.input_paths]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = organize_candidates(
        input_paths=input_paths,
        target_dev_path=Path(args.target_dev_path) if args.target_dev_path else None,
        holdout_path=Path(args.holdout_path) if args.holdout_path else None,
    )

    accepted = [row for row in results if row["final_status"] == "accepted"]
    manual_review = [row for row in results if row["final_status"] == "needs_manual_review"]
    rejected = [row for row in results if row["final_status"].startswith("rejected_")]

    write_jsonl(output_dir / "all_candidates.jsonl", results)
    write_jsonl(output_dir / "accepted.jsonl", accepted)
    write_jsonl(output_dir / "needs_manual_review.jsonl", manual_review)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    generate_summary(
        output_path=output_dir / "summary.md",
        input_paths=args.input_paths,
        rows=results,
        target_dev_path=args.target_dev_path,
        holdout_path=args.holdout_path,
    )


if __name__ == "__main__":
    main()
