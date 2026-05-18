#!/usr/bin/env python3
"""Reward helpers for medical single/multiple-choice GRPO."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

VALID_LETTER_RE = re.compile(r"[A-Z]")
ANSWER_LABEL_RE = re.compile(r"答案\s*[:：]\s*([A-Za-z,\s，、/]+)")


@dataclass
class ChoiceRewardResult:
    reward: float
    pred: str
    gold: str
    valid: bool
    exact: bool
    option_f1: float
    extra_count: int
    missing_count: int
    format_reward: float
    exact_match_reward: float
    invalid_penalty: float
    too_long_penalty: float
    precision: float = 0.0
    recall: float = 0.0
    no_answer_label_penalty: float = 0.0
    multiple_answer_label_penalty: float = 0.0
    trailing_text_penalty: float = 0.0


def normalize_letters(value: Any, valid_options: list[str] | set[str] | None = None) -> str:
    valid = {str(v).strip().upper() for v in valid_options} if valid_options else set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    letters: list[str] = []
    for ch in str(value or "").upper():
        if "A" <= ch <= "Z" and ch in valid and ch not in letters:
            letters.append(ch)
    return "".join(sorted(letters))


def extract_choice_answer(text: Any, valid_options: list[str] | set[str] | None = None) -> str:
    content = str(text or "").strip()
    if not content:
        return ""
    matches = ANSWER_LABEL_RE.findall(content)
    if matches:
        return normalize_letters(matches[-1], valid_options)

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return ""
    last_line = lines[-1]
    if len(last_line) <= 32:
        return normalize_letters(last_line, valid_options)
    return ""


def option_f1(pred: str, gold: str) -> float:
    pred_set = set(pred)
    gold_set = set(gold)
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def option_precision_recall(pred: str, gold: str) -> tuple[float, float]:
    pred_set = set(pred)
    gold_set = set(gold)
    if not pred_set or not gold_set:
        return 0.0, 0.0
    tp = len(pred_set & gold_set)
    return tp / len(pred_set), tp / len(gold_set)


def too_long_penalty(text: Any) -> float:
    content = str(text or "")
    penalty = 0.0
    if len(content) > 512:
        penalty -= 0.1
    if len(ANSWER_LABEL_RE.findall(content)) > 1:
        penalty -= 0.1
    if content.count("答案") > 2:
        penalty -= 0.1
    return penalty


def score_choice_completion(
    completion: Any,
    gold_answer: Any,
    valid_options: list[str] | set[str] | None = None,
) -> ChoiceRewardResult:
    valid_options = valid_options or list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    gold = normalize_letters(gold_answer, valid_options)
    pred = extract_choice_answer(completion, valid_options)
    valid = bool(pred)
    pred_set = set(pred)
    gold_set = set(gold)
    extra_count = len(pred_set - gold_set) if valid else 0
    missing_count = len(gold_set - pred_set) if valid else len(gold_set)
    f1 = option_f1(pred, gold)
    exact = valid and pred == gold
    fmt_reward = 0.2 if valid else -0.5
    exact_reward = 1.0 if exact else 0.0
    invalid = 0.0 if valid else -0.5
    long_penalty = too_long_penalty(completion)
    reward = (
        fmt_reward
        + exact_reward
        + 0.5 * f1
        - 0.3 * extra_count
        - 0.2 * missing_count
        + invalid
        + long_penalty
    )
    return ChoiceRewardResult(
        reward=reward,
        pred=pred,
        gold=gold,
        valid=valid,
        exact=exact,
        option_f1=f1,
        extra_count=extra_count,
        missing_count=missing_count,
        format_reward=fmt_reward,
        exact_match_reward=exact_reward,
        invalid_penalty=invalid,
        too_long_penalty=long_penalty,
        precision=option_precision_recall(pred, gold)[0],
        recall=option_precision_recall(pred, gold)[1],
    )


def _answer_label_stats(text: Any) -> tuple[int, bool]:
    content = str(text or "")
    matches = list(ANSWER_LABEL_RE.finditer(content))
    if not matches:
        return 0, False
    last = matches[-1]
    trailing = content[last.end():].strip()
    has_trailing_text = bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", trailing))
    return len(matches), has_trailing_text


def too_long_penalty_v2(text: Any) -> float:
    content = str(text or "").strip()
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", content))
    if chinese_chars > 40 or len(content) > 80:
        return -0.3
    return 0.0


def score_choice_completion_v2(
    completion: Any,
    gold_answer: Any,
    valid_options: list[str] | set[str] | None = None,
) -> ChoiceRewardResult:
    valid_options = valid_options or list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    text = str(completion or "")
    gold = normalize_letters(gold_answer, valid_options)
    pred = extract_choice_answer(text, valid_options)
    valid = bool(pred)
    pred_set = set(pred)
    gold_set = set(gold)
    extra_count = len(pred_set - gold_set) if valid else 0
    missing_count = len(gold_set - pred_set) if valid else len(gold_set)
    precision, recall = option_precision_recall(pred, gold)
    f1 = option_f1(pred, gold)
    exact = valid and pred == gold
    answer_label_count, has_trailing_text = _answer_label_stats(text)
    fmt_reward = 0.2 if valid else 0.0
    exact_reward = 2.0 if exact else 0.0
    invalid = 0.0 if valid else -1.0
    no_label_penalty = -0.3 if answer_label_count == 0 else 0.0
    multi_label_penalty = -0.5 if answer_label_count > 1 else 0.0
    trailing_penalty = -0.3 if has_trailing_text else 0.0
    long_penalty = too_long_penalty_v2(text)
    reward = (
        fmt_reward
        + exact_reward
        + 0.8 * precision
        + 0.2 * recall
        - 0.7 * extra_count
        - 0.2 * missing_count
        + invalid
        + no_label_penalty
        + multi_label_penalty
        + trailing_penalty
        + long_penalty
    )
    return ChoiceRewardResult(
        reward=reward,
        pred=pred,
        gold=gold,
        valid=valid,
        exact=exact,
        option_f1=f1,
        extra_count=extra_count,
        missing_count=missing_count,
        format_reward=fmt_reward,
        exact_match_reward=exact_reward,
        invalid_penalty=invalid,
        too_long_penalty=long_penalty,
        precision=precision,
        recall=recall,
        no_answer_label_penalty=no_label_penalty,
        multiple_answer_label_penalty=multi_label_penalty,
        trailing_text_penalty=trailing_penalty,
    )
    return ChoiceRewardResult(
        reward=reward,
        pred=pred,
        gold=gold,
        valid=valid,
        exact=exact,
        option_f1=f1,
        extra_count=extra_count,
        missing_count=missing_count,
        format_reward=fmt_reward,
        exact_match_reward=exact_reward,
        invalid_penalty=invalid,
        too_long_penalty=long_penalty,
    )


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content") or "")
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or "")
    return str(completion or "")


def med_choice_reward(completions: list[Any], answer: list[Any], valid_options: list[Any] | None = None, **_: Any) -> list[float]:
    rewards: list[float] = []
    valid_options = valid_options or [None] * len(completions)
    for completion, gold, valid in zip(completions, answer, valid_options):
        rewards.append(score_choice_completion(_completion_text(completion), gold, valid).reward)
    return rewards


def med_choice_reward_v2(completions: list[Any], answer: list[Any], valid_options: list[Any] | None = None, **_: Any) -> list[float]:
    rewards: list[float] = []
    valid_options = valid_options or [None] * len(completions)
    for completion, gold, valid in zip(completions, answer, valid_options):
        rewards.append(score_choice_completion_v2(_completion_text(completion), gold, valid).reward)
    return rewards


if __name__ == "__main__":
    examples = [
        ("分析...\n答案：ABDE", "ABDE", list("ABCDE")),
        ("答案：ABCDE", "ABDE", list("ABCDE")),
        ("BE", "ABDE", list("ABCDE")),
    ]
    for text, gold, valid in examples:
        print(score_choice_completion(text, gold, valid))
