#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vector-based near-duplicate detection for CMB-style datasets.

This is a lightweight fallback when sentence-transformers is unavailable. It
uses character n-gram TF-IDF vectors over question + options and cosine
similarity to find near duplicates. It is not a replacement for semantic
embeddings, but works well for near-identical Chinese exam questions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF vector dedup for CMB-style JSONL/JSON files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Deduplicated JSONL output.")
    parser.add_argument("--pairs-output", required=True, help="Near-duplicate pair JSONL output.")
    parser.add_argument("--clusters-output", required=True, help="Near-duplicate cluster JSON output.")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for pilot runs.")
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--question-repeat", type=int, default=4, help="Repeat question text to reduce option-dominated false positives.")
    parser.add_argument("--allow-answer-mismatch", action="store_true", help="Allow near-duplicate pairs with different gold answers.")
    parser.add_argument(
        "--bucket-fields",
        default="exam_type,exam_class,exam_subject",
        help="Comma-separated fields used to restrict vector search within the same data bucket. Use 'none' to disable.",
    )
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        data = json.loads(text)
        return [row for row in data if isinstance(row, dict)]
    rows = []
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


def get_options(row: dict[str, Any]) -> dict[str, Any]:
    options = row.get("options") if "options" in row else row.get("option")
    return options if isinstance(options, dict) else {}


def vector_text(row: dict[str, Any], question_repeat: int) -> str:
    options = get_options(row)
    opt_text = "".join(f"{key}{options[key]}" for key in sorted(options))
    question_text = str(row.get("question", ""))
    weighted_question = question_text * max(question_repeat, 1)
    return normalize_text(f"{weighted_question}{opt_text}")


def sample_id(row: dict[str, Any], idx: int) -> str:
    return str(row.get("sample_id") or row.get("id") or idx)


def quality_score(row: dict[str, Any]) -> float:
    if isinstance(row.get("quality_score"), (int, float)):
        return float(row["quality_score"])
    question_len = len(str(row.get("question") or ""))
    options = get_options(row)
    nonempty_options = sum(1 for value in options.values() if str(value or "").strip())
    score = 0.0
    if 10 <= question_len <= 200:
        score += 0.4
    elif 5 <= question_len <= 400:
        score += 0.2
    score += min(nonempty_options, 5) / 5 * 0.4
    answer = str(row.get("answer") or "").strip().upper()
    if answer:
        score += 0.2
    return score


def normalized_answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or "").strip().upper()


def parse_bucket_fields(value: str) -> list[str]:
    if value.strip().lower() in {"", "none", "null", "-"}:
        return []
    return [field.strip() for field in value.split(",") if field.strip()]


def bucket_key(row: dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    if not fields:
        return ("__all__",)
    return tuple(str(row.get(field) or "__missing__").strip() or "__missing__" for field in fields)


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rows = read_json_or_jsonl(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]

    pairs = []
    dsu = DSU(len(rows))
    seen_pairs = set()
    bucket_fields = parse_bucket_fields(args.bucket_fields)
    buckets: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        buckets[bucket_key(row, bucket_fields)].append(idx)

    for bucket, row_indexes in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(row_indexes) < 2:
            continue
        bucket_rows = [rows[idx] for idx in row_indexes]
        texts = [vector_text(row, args.question_repeat) for row in bucket_rows]
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(args.ngram_min, args.ngram_max),
            max_features=args.max_features,
            min_df=2 if len(row_indexes) >= 4 else 1,
            dtype=np.float32,
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            continue

        n_neighbors = min(args.top_k + 1, len(row_indexes))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(matrix)
        distances, indices = nn.kneighbors(matrix, return_distance=True)

        for local_i in range(len(row_indexes)):
            i = row_indexes[local_i]
            for distance, local_j in zip(distances[local_i], indices[local_i]):
                j = row_indexes[int(local_j)]
                if i == j:
                    continue
                sim = 1.0 - float(distance)
                if sim < args.threshold:
                    continue
                answer_mismatch = normalized_answer(rows[i]) != normalized_answer(rows[j])
                if answer_mismatch and not args.allow_answer_mismatch:
                    continue
                key = tuple(sorted((i, j)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                dsu.union(i, j)
                pairs.append(
                    {
                        "bucket": dict(zip(bucket_fields, bucket)) if bucket_fields else {"all": True},
                        "left_index": i,
                        "right_index": j,
                        "similarity": round(sim, 6),
                        "left_id": sample_id(rows[i], i),
                        "right_id": sample_id(rows[j], j),
                        "left_answer": rows[i].get("answer"),
                        "right_answer": rows[j].get("answer"),
                        "answer_mismatch": answer_mismatch,
                        "left_question": str(rows[i].get("question") or "")[:220],
                        "right_question": str(rows[j].get("question") or "")[:220],
                    }
                )

    clusters_by_root: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        clusters_by_root[dsu.find(i)].append(i)
    clusters = [members for members in clusters_by_root.values() if len(members) > 1]

    keep_indexes = set(range(len(rows)))
    cluster_records = []
    for members in clusters:
        ranked = sorted(members, key=lambda idx: (quality_score(rows[idx]), -idx), reverse=True)
        keep = ranked[0]
        for idx in ranked[1:]:
            keep_indexes.discard(idx)
        cluster_records.append(
            {
                "bucket": dict(zip(bucket_fields, bucket_key(rows[keep], bucket_fields))) if bucket_fields else {"all": True},
                "keep_index": keep,
                "keep_id": sample_id(rows[keep], keep),
                "member_count": len(members),
                "members": [
                    {
                        "index": idx,
                        "id": sample_id(rows[idx], idx),
                        "quality_score": quality_score(rows[idx]),
                        "answer": rows[idx].get("answer"),
                        "question": str(rows[idx].get("question") or "")[:180],
                    }
                    for idx in ranked
                ],
            }
        )

    deduped = [row for idx, row in enumerate(rows) if idx in keep_indexes]
    write_jsonl(Path(args.output), deduped)
    write_jsonl(Path(args.pairs_output), pairs)
    Path(args.clusters_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.clusters_output).write_text(json.dumps(cluster_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rows={len(rows)} pairs={len(pairs)} clusters={len(clusters)} removed={len(rows)-len(deduped)} kept={len(deduped)}")
    print(f"bucket_fields={bucket_fields or ['__all__']} buckets={len(buckets)}")
    print(f"wrote {args.output}")
    print(f"wrote {args.pairs_output}")
    print(f"wrote {args.clusters_output}")


if __name__ == "__main__":
    main()
