import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "filter_medreason_candidates_day9.py"
SPEC = importlib.util.spec_from_file_location("filter_medreason_candidates_day9", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
sys.modules.setdefault("filter_medreason_candidates_day9", MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate(sample_id: str, route: str, text: str, dataset: str, subset: str, flags=None) -> dict:
    return {
        "sample_id": sample_id,
        "split": "train_candidate",
        "route": route,
        "case_text": text,
        "source": {"dataset": dataset, "subset": subset},
        "meta": {"input_style": "semi_structured_case"},
        "quality_flags": flags or [],
        "audit": {"case_text_hash": sample_id, "case_text_chars": len(text)},
    }


class FilterMedReasonCandidatesDay9Test(unittest.TestCase):
    def test_main_scores_ranks_and_selects_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target_path = tmp_path / "target.jsonl"
            candidate_path = tmp_path / "candidates.jsonl"
            out_ranked = tmp_path / "ranked.jsonl"
            out_selected = tmp_path / "selected.jsonl"
            out_summary = tmp_path / "summary.md"

            write_jsonl(
                target_path,
                [
                    {
                        "sample_id": "dev_cardio_001",
                        "case_text": "患者，男，61岁。胸闷胸痛2小时，疼痛放射至左肩，伴出汗恶心。既往高血压。",
                    },
                    {
                        "sample_id": "dev_resp_001",
                        "case_text": "患者，女，45岁。咳嗽咳痰7天，发热，右下肺湿啰音，血氧96%。",
                    },
                ],
            )
            write_jsonl(
                candidate_path,
                [
                    candidate(
                        "cmb_clin_001",
                        "structured_block_candidate",
                        "现病史\n病人，男，60岁，胸痛2小时，伴出汗。体格检查 BP 160/90mmHg。",
                        "FreedomIntelligence/CMB",
                        "CMB-Clin",
                    ),
                    candidate(
                        "huatuo_001",
                        "rewrite_candidate",
                        "患者胸闷胸痛2小时，伴恶心出汗，既往高血压，应该做什么检查？",
                        "FreedomIntelligence/HuatuoGPT-sft-data-v1",
                        "default",
                    ),
                    candidate(
                        "shibing_001",
                        "rewrite_candidate",
                        "曲匹地尔片的用法用量是什么？",
                        "shibing624/medical",
                        "finetune/train_zh_0",
                        flags=["generic_medical_qa", "weak_case_context"],
                    ),
                    candidate(
                        "cmb_exam_001",
                        "exam_rewrite_candidate",
                        "患者，女，45岁。咳嗽咳痰7天，发热。最可能的诊断是\nA. 肺炎\nB. 胃炎",
                        "FreedomIntelligence/CMB",
                        "CMB-Exam/train",
                    ),
                ],
            )

            argv = [
                "filter_medreason_candidates_day9.py",
                "--candidate-path",
                str(candidate_path),
                "--target-dev-path",
                str(target_path),
                "--out-ranked",
                str(out_ranked),
                "--out-selected",
                str(out_selected),
                "--out-summary",
                str(out_summary),
                "--sample-size",
                "4",
                "--selected-size",
                "3",
                "--seed",
                "test",
            ]
            with patch.object(sys, "argv", argv):
                MODULE.main()

            ranked = [json.loads(line) for line in out_ranked.read_text(encoding="utf-8").splitlines()]
            selected = [json.loads(line) for line in out_selected.read_text(encoding="utf-8").splitlines()]
            summary = out_summary.read_text(encoding="utf-8")

            self.assertEqual(len(ranked), 4)
            self.assertEqual(len(selected), 3)
            self.assertTrue(all("day9_scores" in row for row in ranked))
            self.assertIn("cmb_clin_001", {row["sample_id"] for row in ranked})
            self.assertGreater(
                ranked[0]["day9_scores"]["final_score"],
                ranked[-1]["day9_scores"]["final_score"],
            )
            self.assertIn("LLM API used: no", summary)
            self.assertIn("Embedding backend: hash_char_ngram", summary)


if __name__ == "__main__":
    unittest.main()
