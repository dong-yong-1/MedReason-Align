#!/usr/bin/env python3
"""Run hand-written sanity checks for answer-boundary GRPO v2 reward."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grpo_med_choice_rewards_v2 import score_choice_completion_v2


OUTPUT_MD = Path("data/grpo/reward_sanity_check_v2.md")

CASES = [
    {
        "group": "multi",
        "gold": "ABDE",
        "valid_options": list("ABCDE"),
        "completions": [
            "答案：ABDE",
            "答案：ABCDE",
            "答案：ABE",
            "答案：AB",
            "答案：C",
            "没有答案",
            "答案：ABDE\n分析：这是因为……",
            "答案：ABCDE\n答案：ABDE",
        ],
    },
    {
        "group": "single",
        "gold": "B",
        "valid_options": list("ABCDE"),
        "completions": [
            "答案：B",
            "答案：BD",
            "答案：A",
            "答案：ABCDE",
            "没有答案",
        ],
    },
]

FIELDS = [
    "group",
    "gold",
    "completion",
    "pred",
    "valid_extract",
    "exact_match",
    "precision",
    "recall",
    "f1",
    "extra_count",
    "missing_count",
    "answer_label_count",
    "format_reward",
    "exact_reward",
    "precision_reward",
    "recall_reward",
    "extra_penalty",
    "missing_penalty",
    "answer_label_reward",
    "length_penalty",
    "extra_text_penalty",
    "invalid_penalty",
    "final_reward",
]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "Y" if value else "N"
    return str(value).replace("\n", "<br>")


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        for completion in case["completions"]:
            result = score_choice_completion_v2(completion, case["gold"], case["valid_options"]).to_dict()
            result["group"] = case["group"]
            result["completion"] = completion
            rows.append(result)
    return rows


def sanity_passed(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    by_key = {(row["gold"], row["completion"]): row for row in rows}
    issues: list[str] = []

    multi_exact = by_key[("ABDE", "答案：ABDE")]["final_reward"]
    multi_extra = by_key[("ABDE", "答案：ABCDE")]["final_reward"]
    multi_invalid = by_key[("ABDE", "没有答案")]["final_reward"]
    multi_long = by_key[("ABDE", "答案：ABDE\n分析：这是因为……")]["final_reward"]
    multi_multi_label = by_key[("ABDE", "答案：ABCDE\n答案：ABDE")]["final_reward"]
    single_exact = by_key[("B", "答案：B")]["final_reward"]
    single_extra = by_key[("B", "答案：BD")]["final_reward"]
    single_invalid = by_key[("B", "没有答案")]["final_reward"]

    if not multi_exact > multi_extra:
        issues.append("multi exact reward is not greater than extra-option reward")
    if not single_exact > single_extra:
        issues.append("single exact reward is not greater than extra-option reward")
    if not multi_extra - multi_invalid > 0.5:
        issues.append("invalid multi output is not clearly worse than extra-option output")
    if not single_extra - single_invalid > 0.5:
        issues.append("invalid single output is not clearly worse than extra-option output")
    if not multi_exact > multi_long:
        issues.append("long/trailing output is not penalized below clean exact output")
    if not multi_exact > multi_multi_label:
        issues.append("multiple answer-label output is not penalized below clean exact output")

    return not issues, issues


def write_markdown(rows: list[dict[str, Any]], passed: bool, issues: list[str]) -> None:
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GRPO v2 Reward Sanity Check",
        "",
        f"- status: {'PASS' if passed else 'FAIL'}",
        "- reward intent: answer-boundary / precision-oriented; extra options are penalized more strongly than missing options.",
        "",
    ]
    if issues:
        lines.extend(["## Issues", ""])
        lines.extend(f"- {issue}" for issue in issues)
        lines.append("")

    lines.extend(
        [
            "## Cases",
            "",
            "| " + " | ".join(FIELDS) + " |",
            "| " + " | ".join("---" for _ in FIELDS) + " |",
        ]
    )
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in FIELDS) + " |")
    lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    passed, issues = sanity_passed(rows)
    write_markdown(rows, passed, issues)
    print(f"wrote {OUTPUT_MD}")
    print(f"status={'PASS' if passed else 'FAIL'}")
    if issues:
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
