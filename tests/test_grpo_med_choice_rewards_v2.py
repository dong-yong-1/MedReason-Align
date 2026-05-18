from scripts.grpo_med_choice_rewards_v2 import score_choice_completion_v2


VALID = list("ABCDE")


def test_direct_prefers_short_answer():
    concise = score_choice_completion_v2(
        "答案：A",
        "A",
        VALID,
        difficulty="direct",
    )

    verbose = score_choice_completion_v2(
        "分析：首先考虑病因，其次考虑症状，因此综合分析后选择A。\n答案：A",
        "A",
        VALID,
        difficulty="direct",
    )

    assert concise.exact_match
    assert concise.final_reward > verbose.final_reward


def test_brief_prefers_compact_reasoning():
    brief_reasoning = score_choice_completion_v2(
        "分析：题干提示典型表现，因此选择B。\n答案：B",
        "B",
        VALID,
        difficulty="brief",
    )

    no_reasoning = score_choice_completion_v2(
        "答案：B",
        "B",
        VALID,
        difficulty="brief",
    )

    assert brief_reasoning.reasoning_reward > no_reasoning.reasoning_reward
    assert brief_reasoning.final_reward > no_reasoning.final_reward


def test_cot_prefers_structured_reasoning():
    cot = score_choice_completion_v2(
        "分析：首先根据患者症状考虑感染。其次结合实验室检查和影像学表现，可排除其他疾病。选项A、C符合诊断与治疗原则，因此选择AC。\n答案：AC",
        "AC",
        VALID,
        difficulty="cot",
    )

    direct = score_choice_completion_v2(
        "答案：AC",
        "AC",
        VALID,
        difficulty="cot",
    )

    assert cot.reasoning_structure_reward > 0
    assert cot.reasoning_marker_reward > 0
    assert cot.final_reward > direct.final_reward


def test_invalid_answer_still_penalized():
    invalid = score_choice_completion_v2(
        "我不确定",
        "B",
        VALID,
        difficulty="cot",
    )

    assert invalid.valid_extract is False
    assert invalid.final_reward < 0
