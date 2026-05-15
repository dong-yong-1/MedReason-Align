import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_sft_from_teacher_rewrite.py"
SPEC = importlib.util.spec_from_file_location("build_sft_from_teacher_rewrite", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
sys.modules.setdefault("build_sft_from_teacher_rewrite", MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class BuildSftFromTeacherRewriteTest(unittest.TestCase):
    def test_builds_sharegpt_sft_rows_from_accepted_teacher_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "teacher_outputs.jsonl"
            output_path = tmp_path / "sft.jsonl"
            rejected_path = tmp_path / "rejected.jsonl"
            summary_path = tmp_path / "summary.md"

            write_jsonl(
                input_path,
                [
                    {
                        "sample_id": "case_001",
                        "route": "structured_block_candidate",
                        "source": {"dataset": "FreedomIntelligence/CMB"},
                        "final_status": "accepted",
                        "teacher_response": {
                            "decision": "accept",
                            "reject_reason": "",
                            "case_text": "患者，男，61岁。胸痛2小时，伴出汗恶心。",
                            "schema_target": {
                                "primary_diagnosis": "急性冠脉综合征可能性大",
                                "diagnostic_basis": ["胸痛2小时", "伴出汗恶心支持急性心血管事件可能"],
                                "differential_diagnoses": ["主动脉夹层"],
                                "recommended_actions": ["建议急诊就医并完善心电图和肌钙蛋白检查"],
                                "risk_flags": ["若胸痛持续不缓解或出现呼吸困难，应立即急诊处理"],
                            },
                            "quality_tags": ["case_like"],
                        },
                        "model": "deepseek-v4-flash",
                    },
                    {
                        "sample_id": "case_002",
                        "route": "rewrite_candidate",
                        "final_status": "rejected_by_teacher",
                        "teacher_response": {
                            "decision": "reject",
                            "reject_reason": "纯知识问答",
                            "case_text": "",
                            "schema_target": None,
                            "quality_tags": ["pure_knowledge_qa"],
                        },
                    },
                    {
                        "sample_id": "case_003",
                        "route": "rewrite_candidate",
                        "final_status": "invalid",
                        "teacher_response": {
                            "decision": "accept",
                            "case_text": "患者胸痛。",
                            "schema_target": {
                                "primary_diagnosis": "胸痛待查",
                                "diagnostic_basis": ["胸痛"],
                                "differential_diagnoses": [],
                                "recommended_actions": [],
                                "risk_flags": [],
                            },
                        },
                    },
                ],
            )

            argv = [
                "build_sft_from_teacher_rewrite.py",
                "--input-path",
                str(input_path),
                "--output-path",
                str(output_path),
                "--rejected-output-path",
                str(rejected_path),
                "--summary-path",
                str(summary_path),
            ]
            with patch.object(sys, "argv", argv):
                MODULE.main()

            sft_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            rejected_rows = [json.loads(line) for line in rejected_path.read_text(encoding="utf-8").splitlines()]
            summary = summary_path.read_text(encoding="utf-8")

            self.assertEqual(len(sft_rows), 1)
            self.assertEqual(len(rejected_rows), 2)
            self.assertEqual(sft_rows[0]["conversations"][0]["from"], "human")
            self.assertEqual(sft_rows[0]["conversations"][1]["from"], "gpt")
            parsed_schema = json.loads(sft_rows[0]["conversations"][1]["value"])
            self.assertEqual(parsed_schema["primary_diagnosis"], "急性冠脉综合征可能性大")
            self.assertIn("sft_rows: 1", summary)
            self.assertIn("rejected_or_invalid_rows: 2", summary)


if __name__ == "__main__":
    unittest.main()
