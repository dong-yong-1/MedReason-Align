#!/usr/bin/env python3
"""Day 9 target-dev similarity filtering for MedReason candidates.

This script runs the 50k pilot filter without requiring an external API. The
default embedding backend is a deterministic hashed Chinese character n-gram
vectorizer, which keeps the pipeline runnable in a bare Python environment. A
stronger sentence-transformers backend can be added later without changing the
output contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
import heapq
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_CANDIDATE_PATH = "data/medreason/day8_train_candidates_v1.jsonl"
DEFAULT_TARGET_DEV_PATH = "data/medreason/target_dev_seed_v1.jsonl"
DEFAULT_RANKED_OUTPUT = "data/medreason/day9_ranked_candidates_sample50k_v1.jsonl"
DEFAULT_SELECTED_OUTPUT = "data/medreason/day9_selected_for_rewrite_sample50k_v1.jsonl"
DEFAULT_SUMMARY_OUTPUT = "data/medreason/day9_similarity_summary_sample50k_v1.md"

FORCE_INCLUDE_ROUTES = {"structured_block_candidate"}

SOURCE_QUALITY = {
    ("FreedomIntelligence/CMB", "CMB-Clin"): 1.00,
    ("FreedomIntelligence/CMB", "CMB-Exam/train"): 0.75,
    ("FreedomIntelligence/HuatuoGPT-sft-data-v1", "default"): 0.55,
    ("shibing624/medical", "finetune/train_zh_0"): 0.50,
}

ROUTE_PRIORITY = {
    "structured_block_candidate": 1.00,
    "exam_rewrite_candidate": 0.75,
    "rewrite_candidate": 0.50,
}

QUALITY_FLAG_PENALTIES = {
    "generic_medical_qa": 0.15,
    "weak_case_context": 0.20,
    "may_depend_on_image_or_attachment": 0.30,
    "safety_boundary_signal": 0.05,
    "too_short": 1.00,
    "has_replacement_character": 1.00,
}

PATTERN_PENALTIES = (
    (re.compile(r"(用法用量|注意事项|说明书|副作用|不良反应)"), 0.10),
    (re.compile(r"(简介是|是什么|什么是|发病原因|病因是什么|预防.*是什么)"), 0.08),
    (re.compile(r"^(.*)(严重吗|怎么办|吃什么药)[?？]?$"), 0.05),
)

CASE_MARKER_RE = re.compile(r"(患者|病人|主诉|现病史|体格检查|辅助检查)")
AGE_SEX_RE = re.compile(r"([男女].{0,4}\d{1,3}岁|\d{1,3}岁|男性|女性|男|女)")
DURATION_RE = re.compile(r"\d+(\.\d+)?\s*(小时|天|日|周|月|年|分钟)")
EXAM_LAB_RE = re.compile(r"(查体|体格检查|辅助检查|血常规|尿常规|CT|MRI|超声|心电图|胸片|影像|WBC|BP|血氧|体温|℃|°C)")
SYMPTOM_RE = re.compile(r"(发热|疼痛|胸痛|腹痛|咳|咳痰|呕吐|头痛|气短|气促|水肿|乏力|出血|黄疸|尿频|腹泻)")


SparseVector = dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Day 8 candidates with target-dev similarity.")
    parser.add_argument("--candidate-path", default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--target-dev-path", default=DEFAULT_TARGET_DEV_PATH)
    parser.add_argument("--out-ranked", default=DEFAULT_RANKED_OUTPUT)
    parser.add_argument("--out-selected", default=DEFAULT_SELECTED_OUTPUT)
    parser.add_argument("--out-summary", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--selected-size", type=int, default=5_000)
    parser.add_argument("--seed", default="day9_v1")
    parser.add_argument("--hash-dim", type=int, default=16_384)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--max-text-chars", type=int, default=1_200)
    parser.add_argument("--near-dup-hamming-threshold", type=int, default=3)
    parser.add_argument("--max-source-ratio", type=float, default=0.65)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_unit_float(seed: str, key: str) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def stable_hash_int(text: str, digest_size: int = 8) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=digest_size).digest(), "little")


def iter_ngrams(text: str, ngram_min: int, ngram_max: int) -> Iterable[str]:
    compact = re.sub(r"\s+", "", text)
    for n in range(ngram_min, ngram_max + 1):
        if len(compact) < n:
            continue
        for idx in range(0, len(compact) - n + 1):
            yield compact[idx : idx + n]


def embed_hash_char_ngrams(text: str, args: argparse.Namespace) -> SparseVector:
    text = text[: args.max_text_chars]
    counts: dict[int, float] = {}
    for ngram in iter_ngrams(text, args.ngram_min, args.ngram_max):
        raw_hash = stable_hash_int(ngram)
        index = raw_hash % args.hash_dim
        sign = 1.0 if ((raw_hash >> 8) & 1) == 0 else -1.0
        counts[index] = counts.get(index, 0.0) + sign
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm == 0:
        return {}
    return {index: value / norm for index, value in counts.items()}


def cosine_sparse(lhs: SparseVector, rhs: SparseVector) -> float:
    if not lhs or not rhs:
        return 0.0
    if len(lhs) > len(rhs):
        lhs, rhs = rhs, lhs
    return sum(value * rhs.get(index, 0.0) for index, value in lhs.items())


def simhash(text: str, args: argparse.Namespace, bits: int = 64) -> int:
    weights = [0.0] * bits
    for ngram in iter_ngrams(text[: args.max_text_chars], args.ngram_min, args.ngram_max):
        raw_hash = stable_hash_int(ngram)
        for bit in range(bits):
            weights[bit] += 1.0 if ((raw_hash >> bit) & 1) else -1.0
    value = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            value |= 1 << bit
    return value


def hamming_distance(lhs: int, rhs: int) -> int:
    return (lhs ^ rhs).bit_count()


def source_key(row: dict[str, Any]) -> tuple[str, str]:
    source = row.get("source") or {}
    return str(source.get("dataset") or ""), str(source.get("subset") or "")


def source_quality(row: dict[str, Any]) -> float:
    return SOURCE_QUALITY.get(source_key(row), 0.45)


def route_priority(row: dict[str, Any]) -> float:
    return ROUTE_PRIORITY.get(str(row.get("route") or ""), 0.0)


def length_score(text: str) -> float:
    length = len(text)
    if length <= 0:
        return 0.0
    if length < 40:
        return length / 80.0
    if length <= 1_200:
        return 1.0
    if length <= 2_500:
        return 0.85
    return 0.65


def case_richness(row: dict[str, Any]) -> float:
    text = str(row.get("case_text") or "")
    features = [
        1.0 if CASE_MARKER_RE.search(text) else 0.0,
        1.0 if AGE_SEX_RE.search(text) else 0.0,
        1.0 if DURATION_RE.search(text) else 0.0,
        1.0 if EXAM_LAB_RE.search(text) else 0.0,
        1.0 if SYMPTOM_RE.search(text) else 0.0,
        length_score(text),
    ]
    return sum(features) / len(features)


def penalties(row: dict[str, Any]) -> tuple[float, list[str]]:
    total = 0.0
    reasons: list[str] = []
    for flag in row.get("quality_flags") or []:
        value = QUALITY_FLAG_PENALTIES.get(flag, 0.0)
        if value:
            total += value
            reasons.append(flag)
    text = str(row.get("case_text") or "")
    for pattern, value in PATTERN_PENALTIES:
        if pattern.search(text):
            total += value
            reasons.append(f"pattern:{pattern.pattern}")
    if len(text) > 3_000:
        total += 0.05
        reasons.append("very_long_text")
    return total, reasons


def load_targets(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        text = str(row.get("case_text") or "")
        if not text:
            continue
        targets.append(
            {
                "sample_id": row.get("sample_id") or f"target_{len(targets) + 1:04d}",
                "case_text": text,
                "embedding": embed_hash_char_ngrams(text, args),
            }
        )
    if not targets:
        raise ValueError(f"No target_dev case_text rows found in {path}")
    return targets


def sample_candidates(path: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    forced: list[dict[str, Any]] = []
    heap: list[tuple[float, int, dict[str, Any]]] = []
    heap_limit = max(args.sample_size, 1)
    counter = 0

    for row in read_jsonl(path):
        stats["input_rows"] += 1
        route = str(row.get("route") or "")
        if route in FORCE_INCLUDE_ROUTES:
            forced.append(row)
            stats["force_included_rows"] += 1
            continue
        key = str(row.get("sample_id") or row.get("audit", {}).get("case_text_hash") or stats["input_rows"])
        sample_key = stable_unit_float(args.seed, key)
        counter += 1
        item = (-sample_key, counter, row)
        if len(heap) < heap_limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    remaining = max(args.sample_size - len(forced), 0)
    sampled_non_forced = [item[2] for item in sorted(heap, reverse=True)[:remaining]]
    sampled = forced[: args.sample_size] + sampled_non_forced
    stats["sampled_rows"] = len(sampled)
    return sampled, stats


def score_candidate(row: dict[str, Any], targets: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    text = str(row.get("case_text") or "")
    embedding = embed_hash_char_ngrams(text, args)
    similarities = [
        (cosine_sparse(embedding, target["embedding"]), target["sample_id"])
        for target in targets
    ]
    similarities.sort(reverse=True)
    top3 = similarities[:3]
    sim_max = top3[0][0] if top3 else 0.0
    sim_top3_mean = sum(value for value, _ in top3) / len(top3) if top3 else 0.0
    src_quality = source_quality(row)
    rich = case_richness(row)
    route = route_priority(row)
    penalty_value, penalty_reasons = penalties(row)
    final = (
        0.55 * sim_top3_mean
        + 0.25 * src_quality
        + 0.15 * rich
        + 0.05 * route
        - penalty_value
    )

    enriched = dict(row)
    enriched["day9_scores"] = {
        "embedding_backend": "hash_char_ngram",
        "target_similarity_max": round(sim_max, 6),
        "target_similarity_top3_mean": round(sim_top3_mean, 6),
        "matched_target_ids": [target_id for _, target_id in top3],
        "source_quality": round(src_quality, 6),
        "case_richness": round(rich, 6),
        "route_priority": round(route, 6),
        "penalties": round(penalty_value, 6),
        "penalty_reasons": penalty_reasons,
        "final_score": round(final, 6),
        "simhash64": f"{simhash(text, args):016x}",
    }
    return enriched


def select_for_rewrite(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    max_per_source = max(int(args.selected_size * args.max_source_ratio), 1)
    selected_simhashes: list[int] = []

    for row in rows:
        source_name = row.get("source", {}).get("dataset", "unknown")
        if source_counts[source_name] >= max_per_source:
            stats["skipped_source_cap"] += 1
            continue
        current_simhash = int(row["day9_scores"]["simhash64"], 16)
        if any(
            hamming_distance(current_simhash, prior) <= args.near_dup_hamming_threshold
            for prior in selected_simhashes
        ):
            stats["skipped_near_duplicate"] += 1
            continue
        selected.append(row)
        selected_simhashes.append(current_simhash)
        source_counts[source_name] += 1
        if len(selected) >= args.selected_size:
            break
    stats["selected_rows"] = len(selected)
    for source, count in source_counts.items():
        stats[f"selected_source::{source}"] = count
    return selected, stats


def write_summary(
    path: Path,
    args: argparse.Namespace,
    sample_stats: Counter[str],
    select_stats: Counter[str],
    ranked_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    route_counts = Counter(row.get("route", "unknown") for row in ranked_rows)
    source_counts = Counter(row.get("source", {}).get("dataset", "unknown") for row in ranked_rows)
    selected_route_counts = Counter(row.get("route", "unknown") for row in selected_rows)
    selected_source_counts = Counter(row.get("source", {}).get("dataset", "unknown") for row in selected_rows)

    scores = [row["day9_scores"]["final_score"] for row in ranked_rows]
    selected_scores = [row["day9_scores"]["final_score"] for row in selected_rows]

    def score_line(values: list[float]) -> str:
        if not values:
            return "n/a"
        values_sorted = sorted(values)
        return (
            f"min={values_sorted[0]:.4f}, "
            f"p50={values_sorted[len(values_sorted)//2]:.4f}, "
            f"max={values_sorted[-1]:.4f}"
        )

    lines = [
        "# Day 9 Similarity Filtering Summary sample50k v1",
        "",
        "## Scope",
        "- Input: Day 8 train candidates.",
        "- Target anchor: target_dev case_text.",
        "- Sample size requested: 50000.",
        "- LLM API used: no.",
        "- Embedding backend: hash_char_ngram.",
        "",
        "## Outputs",
        f"- ranked_candidates: `{args.out_ranked}`",
        f"- selected_for_rewrite: `{args.out_selected}`",
        f"- summary: `{args.out_summary}`",
        "",
        "## Counts",
        f"- input_rows_seen: {sample_stats.get('input_rows', 0)}",
        f"- force_included_rows: {sample_stats.get('force_included_rows', 0)}",
        f"- ranked_rows: {len(ranked_rows)}",
        f"- selected_rows: {len(selected_rows)}",
        f"- skipped_near_duplicate: {select_stats.get('skipped_near_duplicate', 0)}",
        f"- skipped_source_cap: {select_stats.get('skipped_source_cap', 0)}",
        "",
        "## Score Distribution",
        f"- ranked_final_score: {score_line(scores)}",
        f"- selected_final_score: {score_line(selected_scores)}",
        "",
        "## Ranked Routes",
        "| route | count |",
        "|---|---:|",
    ]
    for route, count in sorted(route_counts.items()):
        lines.append(f"| {route} | {count} |")

    lines.extend(["", "## Selected Routes", "| route | count |", "|---|---:|"])
    for route, count in sorted(selected_route_counts.items()):
        lines.append(f"| {route} | {count} |")

    lines.extend(["", "## Ranked Sources", "| source | count |", "|---|---:|"])
    for source, count in sorted(source_counts.items()):
        lines.append(f"| {source} | {count} |")

    lines.extend(["", "## Selected Sources", "| source | count |", "|---|---:|"])
    for source, count in sorted(selected_source_counts.items()):
        lines.append(f"| {source} | {count} |")

    lines.extend(
        [
            "",
            "## Top 10 Selected",
            "| rank | sample_id | route | source | score | top_match |",
            "|---:|---|---|---|---:|---|",
        ]
    )
    for rank, row in enumerate(selected_rows[:10], start=1):
        scores_obj = row["day9_scores"]
        source = row.get("source", {}).get("dataset", "unknown")
        top_match = ",".join(scores_obj.get("matched_target_ids", [])[:1])
        lines.append(
            f"| {rank} | {row.get('sample_id')} | {row.get('route')} | {source} | "
            f"{scores_obj['final_score']:.4f} | {top_match} |"
        )

    lines.extend(
        [
            "",
            "## Formula",
            "`final_score = 0.55 * target_similarity_top3_mean + 0.25 * source_quality + 0.15 * case_richness + 0.05 * route_priority - penalties`",
            "",
            "## Notes",
            "- This is a pilot filter for Day 9; it ranks candidate inputs, not final schema labels.",
            "- The current backend is dependency-free. Replace it with bge-small/base embeddings later for the stronger run.",
            "- `structured_block_candidate` rows are force-included in the 50k scoring pool.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    targets = load_targets(Path(args.target_dev_path), args)
    sampled_rows, sample_stats = sample_candidates(Path(args.candidate_path), args)
    ranked_rows = [score_candidate(row, targets, args) for row in sampled_rows]
    ranked_rows.sort(key=lambda row: row["day9_scores"]["final_score"], reverse=True)
    selected_rows, select_stats = select_for_rewrite(ranked_rows, args)

    write_jsonl(Path(args.out_ranked), ranked_rows)
    write_jsonl(Path(args.out_selected), selected_rows)
    write_summary(Path(args.out_summary), args, sample_stats, select_stats, ranked_rows, selected_rows)


if __name__ == "__main__":
    main()
