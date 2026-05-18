#!/usr/bin/env python3
"""Adaptive reasoning reward helpers for medical-choice GRPO v2.

The reward keeps the answer-boundary scoring from v2, and adds a difficulty-aware
reasoning signal driven by the sample-level ``difficulty`` field:

- direct: prefer concise direct-answer outputs, e.g. ``答案：A``.
- brief: prefer short rationale + final answer, e.g. ``分析：...\n答案：A``.
- cot: prefer structured CoT-style reasoning + final answer.

This file is intentionally framework-light so it can be imported both by pytest
and by TRL GRPOTrainer reward functions.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

ANSWER_LABEL_RE = re.compile(r"答案\s*[:：]\s*([^\n\r]*)")
ANALYSIS_LABEL_RE = re.compile(r"(?:分析|解析|思路|理由|推理)\s*[:：]")
REASONING_MARKER_RE = re.compile(
    r"因为|因此|所以|首先|其次|再者|另外|提示|考虑|诊断|鉴别|排除|符合|不符合|机制|病因|治疗|首选|禁忌|表现|症状|体征|检查|选项|可见|可知"
)
REPETITION_RE = re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,12})\1{2,}")


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
    difficulty: str
    reasoning_reward: float
    reasoning_length_reward: float
    reasoning_structure_reward: float
    reasoning_marker_reward: float
    reasoning_absence_penalty: float
    reasoning_length_penalty: float
    repetition_penalty: float

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


def normalize_difficulty(value: Any) -> str:
    difficulty = str(value or "brief").strip().lower()
    if difficulty not in {"direct", "brief", "cot"}:
        return "brief"
    return difficulty


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


def _split_reasoning_before_answer(text: str) -> str:
    matches = list(ANSWER_LABEL_RE.finditer(text))
    if matches:
        return text[: matches[-1].start()].strip()
    return text.strip()


def _count_chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def _score_reasoning_by_difficulty(text: str, difficulty: str) -> dict[str, float]:
    difficulty = normalize_difficulty(difficulty)
    reasoning = _split_reasoning_before_answer(text)
    reasoning_zh_len = _count_chinese_chars(reasoning)
    has_analysis_label = bool(ANALYSIS_LABEL_RE.search(reasoning))
    marker_count = len(REASONING_MARKER_RE.findall(reasoning))
    has_repetition = bool(REPETITION_RE.search(text))

    reasoning_length_reward = 0.0
    reasoning_structure_reward = 0.0
    reasoning_marker_reward = 0.0
    reasoning_absence_penalty = 0.0
    reasoning_length_penalty = 0.0
    repetition_penalty = -0.25 if has_repetition else 0.0

    if difficulty == "direct":
        # Direct samples should not be trained to produce long CoT. Reward compactness.
        if reasoning_zh_len == 0:
            reasoning_length_reward = 0.25
        elif reasoning_zh_len <= 16:
            reasoning_length_reward = 0.10
        else:
            reasoning_length_penalty = -min(0.60, 0.12 + 0.01 * (reasoning_zh_len - 16))
        if has_analysis_label:
            reasoning_structure_reward = -0.20
        if marker_count >= 2:
            reasoning_marker_reward = -0.15

    elif difficulty == "brief":
        # Brief samples should contain a compact explanation, not a long chain.
        if 8 <= reasoning_zh_len <= 80:
            reasoning_length_reward = 0.35
        elif reasoning_zh_len == 0:
            reasoning_absence_penalty = -0.20
        elif reasoning_zh_len > 120:
            reasoning_length_penalty = -min(0.50, 0.10 + 0.004 * (reasoning_zh_len - 120))
        if has_analysis_label:
            reasoning_structure_reward = 0.15
        if marker_count >= 1:
            reasoning_marker_reward = 0.15

    else:  # cot
        # CoT samples need enough reasoning structure, while still avoiding rambling.
        if 40 <= reasoning_zh_len <= 260:
            reasoning_length_reward = 0.45
        elif reasoning_zh_len < 20:
            reasoning_absence_penalty = -0.45
        elif reasoning_zh_len > 360:
            reasoning_length_penalty = -min(0.70, 0.15 + 0.002 * (reasoning_zh_len - 360))
        if has_analysis_label:
            reasoning_structure_reward = 0.25
        else:
            reasoning_structure_reward = -0.10
        reasoning_marker_reward = min(0.30, 0.10 * marker_count)

    reasoning_reward = (
        reasoning_length_reward
        + reasoning_structure_reward
        + reasoning_marker_reward
        + reasoning_absence_penalty
        + reasoning_length_penalty
        + repetition_penalty
    )
    return {
        "reasoning_reward": reasoning_reward,
        "reasoning_length_reward": reasoning_length_reward,
        "reasoning_structure_reward": reasoning_structure_reward,
        "reasoning_marker_reward": reasoning_marker_reward,
        "reasoning_absence_penalty": reasoning_absence_penalty,
        "reasoning_length_penalty": reasoning_length_penalty,
        "repetition_penalty": repetition_penalty,
    }


def score_choice_completion_v2(
    completion: Any,
    gold_answer: Any,
    valid_options: list[str] | set[str] | None = None,
    difficulty: Any = "brief",
) -> ChoiceRewardV2Result:
    valid_options = valid_options or list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    difficulty = normalize_difficulty(difficulty)
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

    chinese_char_count = _count_chinese_chars(text)
    if difficulty == "direct":
        length_penalty = -0.3 if chinese_char_count > 40 or len(text) > 80 else 0.0
    elif difficulty == "brief":
        length_penalty = -0.3 if chinese_char_count > 140 or len(text) > 260 else 0.0
    else:
        length_penalty = -0.3 if chinese_char_count > 420 or len(text) > 780 else 0.0
    extra_text_penalty = -0.3 if extra_text_after_answer else 0.0
    reasoning_scores = _score_reasoning_by_difficulty(text, difficulty)

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
        + reasoning_scores["reasoning_reward"]
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
        difficulty=difficulty,
        reasoning_reward=reasoning_scores["reasoning_reward"],
        reasoning_length_reward=reasoning_scores["reasoning_length_reward"],
        reasoning_structure_reward=reasoning_scores["reasoning_structure_reward"],
        reasoning_marker_reward=reasoning_scores["reasoning_marker_reward"],
        reasoning_absence_penalty=reasoning_scores["reasoning_absence_penalty"],
        reasoning_length_penalty=reasoning_scores["reasoning_length_penalty"],
        repetition_penalty=reasoning_scores["repetition_penalty"],
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


def _expand_or_default(values: list[Any] | None, length: int, default: Any) -> list[Any]:
    if values is None:
        return [default] * length
    if len(values) < length:
        return list(values) + [default] * (length - len(values))
    return list(values)


def med_choice_reward_v2(
    completions: list[Any],
    answer: list[Any],
    valid_options: list[Any] | None = None,
    difficulty: list[Any] | None = None,
    **_: Any,
) -> list[float]:
    """TRL-compatible reward function.

    TRL passes dataset columns as keyword lists. This function consumes ``answer``,
    optional ``valid_options`` and optional ``difficulty``. Missing difficulty is
    treated as ``brief`` for backward compatibility.
    """
    rewards: list[float] = []
    valid_options = _expand_or_default(valid_options, len(completions), None)
    difficulty = _expand_or_default(difficulty, len(completions), "brief")
    for completion, gold, valid, diff in zip(completions, answer, valid_options, difficulty):
        rewards.append(score_choice_completion_v2(_completion_text(completion), gold, valid, diff).final_reward)
    return rewards


# Alias with a task-focused name for new GRPO scripts.
adaptive_reasoning_reward = med_choice_reward_v2


if __name__ == "__main__":
    examples = [
        ("答案：ABDE", "ABDE", list("ABCDE"), "direct"),
        ("分析：题干提示典型表现，因此选择ABDE。\n答案：ABDE", "ABDE", list("ABCDE"), "brief"),
        ("分析：首先根据题干症状考虑该病。其次选项A、B、D、E均符合诊断和治疗原则，C不符合常见机制，因此排除C。\n答案：ABDE", "ABDE", list("ABCDE"), "cot"),
        ("没有答案", "ABDE", list("ABCDE"), "cot"),
    ]
    for text, gold, valid, diff in examples:
        print(score_choice_completion_v2(text, gold, valid, diff))
