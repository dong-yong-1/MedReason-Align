import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "rewrite_medreason_candidates_deepseek.py"
SPEC = importlib.util.spec_from_file_location("rewrite_medreason_candidates_deepseek", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
sys.modules.setdefault("rewrite_medreason_candidates_deepseek", MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def selected_row(sample_id: str, route: str, case_text: str, source: dict, extra=None) -> dict:
    row = {
        "sample_id": sample_id,
        "route": route,
        "case_text": case_text,
        "source": source,
        "quality_flags": [],
        "day9_scores": {"final_score": 0.3},
    }
    if extra:
        row.update(extra)
    return row


class RewriteMedReasonCandidatesDeepSeekTest(unittest.TestCase):
    def test_dry_run_builds_route_balanced_pilot_input_and_hydrates_raw_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            huatuo_path = tmp_path / "huatuo.jsonl"
            selected_path = tmp_path / "selected.jsonl"
            pilot_input_path = tmp_path / "pilot.jsonl"
            summary_path = tmp_path / "summary.md"
            output_path = tmp_path / "outputs.jsonl"

            write_jsonl(
                huatuo_path,
                [{"data": ["问：患者胸痛2小时怎么办？", "答：建议尽快急诊就医，完善心电图和肌钙蛋白。"]}],
            )
            write_jsonl(
                selected_path,
                [
                    selected_row(
                        "cmb_clin_001",
                        "structured_block_candidate",
                        "现病史\n病人，男，49岁，右下腹痛并包块3小时。",
                        {"dataset": "FreedomIntelligence/CMB", "subset": "CMB-Clin"},
                        extra={
                            "raw_blocks": {
                                "diagnosis": [{"question": "诊断？", "answer": "嵌顿性疝。"}],
                                "differential": [],
                                "treatment_or_action": [],
                                "other": [],
                            }
                        },
                    ),
                    selected_row(
                        "cmb_exam_001",
                        "exam_rewrite_candidate",
                        "患者，女，45岁。咳嗽发热。A. 肺炎 B. 胃炎",
                        {"dataset": "FreedomIntelligence/CMB", "subset": "CMB-Exam/train"},
                        extra={"answer_text": "A. 肺炎"},
                    ),
                    selected_row(
                        "huatuo_001",
                        "rewrite_candidate",
                        "患者胸痛2小时怎么办？",
                        {
                            "dataset": "FreedomIntelligence/HuatuoGPT-sft-data-v1",
                            "subset": "default",
                            "path": str(huatuo_path),
                            "line_no": 1,
                        },
                    ),
                ],
            )

            argv = [
                "rewrite_medreason_candidates_deepseek.py",
                "--input-path",
                str(selected_path),
                "--pilot-input-path",
                str(pilot_input_path),
                "--output-path",
                str(output_path),
                "--summary-path",
                str(summary_path),
                "--structured-limit",
                "1",
                "--exam-limit",
                "1",
                "--rewrite-limit",
                "1",
            ]
            with patch.object(sys, "argv", argv):
                MODULE.main()

            prompt_rows = [json.loads(line) for line in pilot_input_path.read_text(encoding="utf-8").splitlines()]
            summary = summary_path.read_text(encoding="utf-8")

            self.assertEqual(len(prompt_rows), 3)
            by_id = {row["sample_id"]: row for row in prompt_rows}
            self.assertIn("raw_blocks", by_id["cmb_clin_001"]["candidate"])
            self.assertEqual(by_id["cmb_exam_001"]["candidate"]["answer_text"], "A. 肺炎")
            self.assertIn("急诊就医", by_id["huatuo_001"]["candidate"]["raw_answer"])
            self.assertIn("API called: no", summary)

    def test_validate_teacher_json_accept_reject_and_invalid(self):
        accepted = {
            "decision": "accept",
            "reject_reason": "",
            "case_text": "患者胸痛2小时。",
            "schema_target": {
                "primary_diagnosis": "急性冠脉综合征可能性大",
                "diagnostic_basis": ["胸痛2小时", "需要排查心血管急症"],
                "differential_diagnoses": ["主动脉夹层"],
                "recommended_actions": ["建议急诊就医并完善心电图"],
                "risk_flags": ["若胸痛持续不缓解应立即急诊处理"],
            },
            "quality_tags": ["case_like"],
        }
        rejected = {
            "decision": "reject",
            "reject_reason": "纯知识问答",
            "case_text": "",
            "schema_target": None,
            "quality_tags": ["pure_knowledge_qa"],
        }
        invalid = {
            "decision": "accept",
            "case_text": "患者胸痛。",
            "schema_target": {
                "primary_diagnosis": "胸痛待查",
                "diagnostic_basis": ["胸痛"],
                "differential_diagnoses": [],
                "recommended_actions": [],
                "risk_flags": [],
            },
        }

        self.assertEqual(MODULE.validate_teacher_json(accepted), ("accepted", []))
        self.assertEqual(MODULE.validate_teacher_json(rejected), ("rejected_by_teacher", []))
        status, errors = MODULE.validate_teacher_json(invalid)
        self.assertEqual(status, "invalid")
        self.assertIn("diagnostic_basis_too_short", errors)

    def test_limit_samples_round_robin_across_routes(self):
        rows = []
        for index in range(3):
            rows.append(
                selected_row(
                    f"structured_{index}",
                    "structured_block_candidate",
                    "患者腹痛并发热。",
                    {"dataset": "FreedomIntelligence/CMB", "subset": "CMB-Clin"},
                )
            )
            rows.append(
                selected_row(
                    f"exam_{index}",
                    "exam_rewrite_candidate",
                    "患者咳嗽发热，考虑肺炎。",
                    {"dataset": "FreedomIntelligence/CMB", "subset": "CMB-Exam/train"},
                )
            )
            rows.append(
                selected_row(
                    f"rewrite_{index}",
                    "rewrite_candidate",
                    "患者胸痛2小时怎么办？",
                    {"dataset": "FreedomIntelligence/HuatuoGPT-sft-data-v1"},
                )
            )

        args = SimpleNamespace(
            limit=7,
            structured_limit=MODULE.ROUTE_LIMIT_DEFAULTS["structured_block_candidate"],
            exam_limit=MODULE.ROUTE_LIMIT_DEFAULTS["exam_rewrite_candidate"],
            rewrite_limit=MODULE.ROUTE_LIMIT_DEFAULTS["rewrite_candidate"],
        )
        chosen = MODULE.choose_pilot_rows(rows, args)

        self.assertEqual(len(chosen), 7)
        self.assertEqual(
            [row["route"] for row in chosen],
            [
                "structured_block_candidate",
                "exam_rewrite_candidate",
                "rewrite_candidate",
                "structured_block_candidate",
                "exam_rewrite_candidate",
                "rewrite_candidate",
                "structured_block_candidate",
            ],
        )

    def test_load_env_file_sets_missing_values_without_overriding_existing_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "DEEPSEEK_API_KEY=from_file",
                        "DEEPSEEK_MODEL='deepseek-v4-flash'",
                        "IGNORED_LINE",
                    ]
                ),
                encoding="utf-8",
            )

            old_key = os.environ.get("DEEPSEEK_API_KEY")
            old_model = os.environ.get("DEEPSEEK_MODEL")
            try:
                os.environ["DEEPSEEK_API_KEY"] = "from_env"
                os.environ.pop("DEEPSEEK_MODEL", None)

                MODULE.load_env_file(env_path)

                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "from_env")
                self.assertEqual(os.environ["DEEPSEEK_MODEL"], "deepseek-v4-flash")
            finally:
                if old_key is None:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                else:
                    os.environ["DEEPSEEK_API_KEY"] = old_key
                if old_model is None:
                    os.environ.pop("DEEPSEEK_MODEL", None)
                else:
                    os.environ["DEEPSEEK_MODEL"] = old_model


if __name__ == "__main__":
    unittest.main()
