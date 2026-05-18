from scripts.grpo_med_choice_rewards import extract_choice_answer, score_choice_completion, score_choice_completion_v2


VALID = list("ABCDE")


def test_exact_multichoice_reward_is_highest():
    result = score_choice_completion("分析：...\n答案：ABDE", "ABDE", VALID)
    assert result.pred == "ABDE"
    assert result.exact
    assert result.extra_count == 0
    assert result.missing_count == 0
    assert result.reward > 1.6


def test_extra_option_penalized():
    exact = score_choice_completion("答案：ABDE", "ABDE", VALID)
    extra = score_choice_completion("答案：ABCDE", "ABDE", VALID)
    assert extra.pred == "ABCDE"
    assert extra.extra_count == 1
    assert not extra.exact
    assert extra.reward < exact.reward


def test_missing_options_penalized():
    result = score_choice_completion("答案：BE", "ABDE", VALID)
    assert result.pred == "BE"
    assert result.extra_count == 0
    assert result.missing_count == 2
    assert not result.exact


def test_single_choice_exact():
    result = score_choice_completion("答案：B", "B", VALID)
    assert result.pred == "B"
    assert result.exact


def test_single_choice_extra_penalized():
    exact = score_choice_completion("答案：B", "B", VALID)
    extra = score_choice_completion("答案：BD", "B", VALID)
    assert extra.extra_count == 1
    assert extra.reward < exact.reward


def test_invalid_answer():
    result = score_choice_completion("我不知道。", "ABDE", VALID)
    assert result.pred == ""
    assert not result.valid
    assert result.reward < 0


def test_extract_last_answer_label():
    assert extract_choice_answer("答案：A\n修正，答案：BD", VALID) == "BD"


def test_v2_extra_is_penalized_more_than_missing():
    extra = score_choice_completion_v2("答案：ABCDE", "ABDE", VALID)
    missing = score_choice_completion_v2("答案：BE", "ABDE", VALID)
    assert extra.extra_count == 1
    assert missing.missing_count == 2
    assert extra.reward < missing.reward


def test_v2_requires_answer_label_and_short_output():
    no_label = score_choice_completion_v2("ABDE", "ABDE", VALID)
    clean = score_choice_completion_v2("答案：ABDE", "ABDE", VALID)
    long = score_choice_completion_v2("答案：ABDE。这里继续输出很多很多没有必要的解释文字，导致答案后面还有明显多余解释。", "ABDE", VALID)
    assert no_label.no_answer_label_penalty < 0
    assert long.trailing_text_penalty < 0
    assert long.too_long_penalty < 0
    assert clean.reward > no_label.reward
    assert clean.reward > long.reward
