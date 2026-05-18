#!/usr/bin/env python3
"""Create a compact GRPO pilot report from four CMExam multi sample files."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-sft", required=True)
    parser.add_argument("--cot-sft", required=True)
    parser.add_argument("--direct-grpo", required=True)
    parser.add_argument("--cot-grpo", required=True)
    parser.add_argument("--gold", default="data/eval/lm_eval/cmexam_test_multi.jsonl")
    parser.add_argument("--output-json", default="data/analysis/grpo_pilot_multichoice_boundary.json")
    parser.add_argument("--output-md", default="docs/grpo_pilot_diagnostics.md")
    return parser.parse_args()


def run_boundary(name: str, direct_pred: str, cot_pred: str, gold: str, out_json: Path, out_md: Path) -> dict:
    cmd = [
        "python",
        "scripts/analyze_cot_multichoice_boundary.py",
        "--direct-pred",
        direct_pred,
        "--cot-pred",
        cot_pred,
        "--gold",
        gold,
        "--dataset-name",
        name,
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def metrics(report: dict, label: str) -> dict:
    return report["metrics"][label]


def table_row(name: str, m: dict) -> str:
    keys = [
        "exact_match",
        "answer_extraction_success_rate",
        "invalid_answer_rate",
        "missing_option_rate",
        "extra_option_rate",
        "option_precision",
        "option_recall",
        "option_f1",
    ]
    return "| " + name + " | " + " | ".join(str(m.get(k, "NA")) for k in keys) + " |"


def main() -> None:
    args = parse_args()
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_json.parent / "grpo_pilot_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sft_report = run_boundary(
        "grpo_pilot_sft_baseline",
        args.direct_sft,
        args.cot_sft,
        args.gold,
        tmp_dir / "sft_boundary.json",
        tmp_dir / "sft_boundary.md",
    )
    grpo_report = run_boundary(
        "grpo_pilot_after_grpo",
        args.direct_grpo,
        args.cot_grpo,
        args.gold,
        tmp_dir / "grpo_boundary.json",
        tmp_dir / "grpo_boundary.md",
    )

    combined = {
        "inputs": vars(args),
        "sft_boundary": sft_report,
        "grpo_boundary": grpo_report,
    }
    out_json.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    direct_sft = metrics(sft_report, "Direct SFT")
    cot_sft = metrics(sft_report, "CoT/Mixed")
    direct_grpo = metrics(grpo_report, "Direct SFT")
    cot_grpo = metrics(grpo_report, "CoT/Mixed")

    lines = [
        "# GRPO Pilot 诊断报告",
        "",
        "## 1. 实验目的",
        "",
        "验证 GRPO 是否能降低 CoT/Mixed SFT 在 CMExam-test-multi 上的 `extra_option_rate`，同时避免 `missed_gold_rate` 明显上升。",
        "",
        "## 2. 数据构造方式",
        "",
        "- 训练数据：`data/grpo/cmb_grpo_pilot_train.jsonl`",
        "- 数据摘要：`data/grpo/cmb_grpo_pilot_summary.md`",
        "- prompt 不包含 gold answer，也不包含 teacher analysis。",
        "",
        "## 3. Reward 设计",
        "",
        "- reward 文件：`scripts/grpo_med_choice_rewards.py`",
        "- 重点：`extra_option_penalty=-0.3`，`missing_option_penalty=-0.2`。",
        "",
        "## 4. 两组 GRPO 初始化 checkpoint",
        "",
        "| 实验 | init/reference adapter | output |",
        "|---|---|---|",
        "| Direct + GRPO | `outputs/sft/cmb_optimized` | `outputs/grpo/cmb_optimized_grpo_pilot` |",
        "| CoT/Mixed + GRPO | `outputs/sft/cmb_cot_mixed` | `outputs/grpo/cmb_cot_mixed_grpo_pilot` |",
        "",
        "## 5. 四组模型评估表",
        "",
        "| 模型 | exact_match | answer_extraction_success_rate | invalid_answer_rate | missed_gold_rate | extra_option_rate | option_precision | option_recall | option_f1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        table_row("Direct SFT", direct_sft),
        table_row("CoT/Mixed SFT", cot_sft),
        table_row("Direct + GRPO", direct_grpo),
        table_row("CoT/Mixed + GRPO", cot_grpo),
        "",
        "## 6. extra_option_rate 是否下降",
        "",
        f"- CoT/Mixed SFT extra_option_rate: `{cot_sft.get('extra_option_rate')}`",
        f"- CoT/Mixed + GRPO extra_option_rate: `{cot_grpo.get('extra_option_rate')}`",
        "",
        "## 7. missed_gold_rate 是否上升",
        "",
        f"- CoT/Mixed SFT missed_gold_rate: `{cot_sft.get('missing_option_rate')}`",
        f"- CoT/Mixed + GRPO missed_gold_rate: `{cot_grpo.get('missing_option_rate')}`",
        "",
        "## 8. exact_match 是否回升",
        "",
        f"- CoT/Mixed SFT exact_match: `{cot_sft.get('exact_match')}`",
        f"- CoT/Mixed + GRPO exact_match: `{cot_grpo.get('exact_match')}`",
        "",
        "## 9. 典型 case",
        "",
        "详见 JSON 中 `grpo_boundary.direct_vs_cot_overlap.representative_error_cases`。",
        "",
        "## 10. 下一步建议",
        "",
        "- 如果 extra 下降但 missing 明显上升，降低 extra penalty 或提高 option F1 权重。",
        "- 如果格式不稳定，提高 format reward 或缩短 max completion。",
        "- 保留 Direct + GRPO 对照，避免把通用 GRPO 收益误归因于 CoT/Mixed。",
    ]
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
