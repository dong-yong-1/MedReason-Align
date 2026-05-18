#!/usr/bin/env python3
"""Answer-boundary reward helpers for medical-choice GRPO v2."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

ANSWER_LABEL_RE = re.compile(r"答案\s*[:：]\s*([^\n\r]*)")


@dataclass
class ChoiceRewardV2Result:
    final_reward: float
    pred: str
    gold: str
    valid_extract: bool
    exact_match: bool
    precision: float
    recall: float
    f1: float
    extra_count: int
    missing_count: int
    answer_label_count: int
    format_reward: float
    exact_reward: float
    precision_reward: float
    recall_reward: float
    extra_penalty: float
    missing_penalty: float
    answer_label_reward: float
    length_penalty: float
    extra_text_penalty: float
    invalid_penalty: float
    used_fallback: bool
    extra_text_after_answer: bool
    completion_length: int
    chinese_char_count: int

    @property
    def reward(self) -> float:
        return self.final_reward

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_letters(value: Any, valid_options: list[str] | set[str] | None = None) -> str:
    valid = {str(v).strip().upper() for v in valid_options} if valid_options else set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    letters: list[str] = []
    for ch in str(value or "").upper():
        if "A" <= ch <= "Z" and ch in valid and ch not in letters:
            letters.append(ch)
    return "".join(sorted(letters))


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _contains_extra_text_after_last_answer(text: str, last_match: re.Match[str] | None) -> bool:
    if last_match is None:
        return False
    answer_line_tail = last_match.group(1)
    letters = "".join(re.findall(r"[A-Za-z]", answer_line_tail))
    if letters:
        last_letter_pos = max(answer_line_tail.rfind(ch) for ch in letters)
        same_line_tail = answer_line_tail[last_letter_pos + 1 :]
    else:
        same_line_tail = answer_line_tail
    trailing = same_line_tail + text[last_match.end() :]
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", trailing))


def extract_choice_answer_v2(
    text: Any,
    valid_options: list[str] | set[str] | None = None,
) -> tuple[str, bool, int, bool, bool]:
    """Extract final answer.

    Returns: pred, valid_extract, answer_label_count, used_fallback, extra_text_after_answer.
    """
    content = str(text or "").strip()
    if not content:
        return "", False, 0, False, False

    matches = list(ANSWER_LABEL_RE.finditer(content))
    if matches:
        pred = normalize_letters(matches[-1].group(1), valid_options)
        extra_text = _contains_extra_text_after_last_answer(content, matches[-1])
        return pred, bool(pred), len(matches), False, extra_text

    last_line = _last_nonempty_line(content)
    if last_line and len(last_line) <= 32:
        pred = normalize_letters(last_line, valid_options)
        return pred, bool(pred), 0, True, False
    return "", False, 0, False, False


def option_precision_recall_f1(pred: str, gold: str) -> tuple[float, float, float]:
    pred_set = set(pred)
    gold_set = set(gold)
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_choice_completion_v2(
    completion: Any,
    gold_answer: Any,
    valid_options: list[str] | set[str] | None = None,
) -> ChoiceRewardV2Result:
    valid_options = valid_options or list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    text = str(completion or "")
    gold = normalize_letters(gold_answer, valid_options)
    pred, valid_extract, answer_label_count, used_fallback, extra_text_after_answer = extract_choice_answer_v2(
        text, valid_options
    )

    pred_set = set(pred)
    gold_set = set(gold)
    extra_count = len(pred_set - gold_set) if valid_extract else 0
    missing_count = len(gold_set - pred_set) if valid_extract else len(gold_set)
    precision, recall, f1 = option_precision_recall_f1(pred, gold)
    exact_match = valid_extract and pred == gold

    format_reward = 0.2 if valid_extract else 0.0
    invalid_penalty = 0.0 if valid_extract else -1.0
    exact_reward = 2.0 if exact_match else 0.0
    precision_reward = 0.8 * precision
    recall_reward = 0.2 * recall
    extra_penalty = -0.7 * extra_count
    missing_penalty = -0.2 * missing_count
    if answer_label_count == 1:
        answer_label_reward = 0.2
    elif answer_label_count == 0:
        answer_label_reward = -0.3
    else:
        answer_label_reward = -0.5
    chinese_char_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    length_penalty = -0.3 if chinese_char_count > 40 or len(text) > 80 else 0.0
    extra_text_penalty = -0.3 if extra_text_after_answer else 0.0

    final_reward = (
        format_reward
        + invalid_penalty
        + exact_reward
        + precision_reward
        + recall_reward
        + extra_penalty
        + missing_penalty
        + answer_label_reward
        + length_penalty
        + extra_text_penalty
    )

    return ChoiceRewardV2Result(
        final_reward=final_reward,
        pred=pred,
        gold=gold,
        valid_extract=valid_extract,
        exact_match=exact_match,
        precision=precision,
        recall=recall,
        f1=f1,
        extra_count=extra_count,
        missing_count=missing_count,
        answer_label_count=answer_label_count,
        format_reward=format_reward,
        exact_reward=exact_reward,
        precision_reward=precision_reward,
        recall_reward=recall_reward,
        extra_penalty=extra_penalty,
        missing_penalty=missing_penalty,
        answer_label_reward=answer_label_reward,
        length_penalty=length_penalty,
        extra_text_penalty=extra_text_penalty,
        invalid_penalty=invalid_penalty,
        used_fallback=used_fallback,
        extra_text_after_answer=extra_text_after_answer,
        completion_length=len(text),
        chinese_char_count=chinese_char_count,
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


def med_choice_reward_v2(
    completions: list[Any],
    answer: list[Any],
    valid_options: list[Any] | None = None,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    valid_options = valid_options or [None] * len(completions)
    for completion, gold, valid in zip(completions, answer, valid_options):
        rewards.append(score_choice_completion_v2(_completion_text(completion), gold, valid).final_reward)
    return rewards


if __name__ == "__main__":
    examples = [
        ("答案：ABDE", "ABDE", list("ABCDE")),
        ("答案：ABCDE", "ABDE", list("ABCDE")),
        ("答案：BE", "ABDE", list("ABCDE")),
        ("没有答案", "ABDE", list("ABCDE")),
    ]
    for text, gold, valid in examples:
        print(score_choice_completion_v2(text, gold, valid))
