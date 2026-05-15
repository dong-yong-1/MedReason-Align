import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_medreason_candidates.py"
SPEC = importlib.util.spec_from_file_location("build_medreason_candidates", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
sys.modules.setdefault("build_medreason_candidates", MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


class BuildMedReasonCandidatesTest(unittest.TestCase):
    def test_main_writes_day8_train_benchmark_review_and_summary_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            huatuo_path = tmp_path / "huatuo.jsonl"
            shibing_path = tmp_path / "shibing.jsonl"
            cmb_clin_path = tmp_path / "cmb_clin.json"
            cmb_train_path = tmp_path / "cmb_train.json"
            cmb_val_path = tmp_path / "cmb_val.json"
            cmb_test_path = tmp_path / "cmb_test.json"

            write_jsonl(
                huatuo_path,
                [{"data": ["问：患者咳嗽发热3天怎么办？", "答：建议完善血常规和胸片检查，并及时就医。"]}],
            )
            write_jsonl(
                shibing_path,
                [{"instruction": "患者腹痛伴呕吐2天，今天疼痛加重怎么办？", "input": "", "output": "建议尽快线下评估。"}],
            )
            write_json(
                cmb_clin_path,
                [
                    {
                        "id": "0",
                        "title": "案例分析-腹外疝",
                        "description": "现病史\n病人，男，49岁，右下腹痛并自扪及包块3小时。\n体格检查\n右侧腹股沟区包块。",
                        "QA_pairs": [
                            {
                                "question": "简述该病人的诊断及诊断依据。",
                                "answer": "诊断：嵌顿性腹股沟斜疝合并肠梗阻。诊断依据：①右下腹痛并自扪及包块3小时。",
                            },
                            {
                                "question": "简述该病人的鉴别诊断。",
                                "answer": "睾丸鞘膜积液、交通性鞘膜积液等。",
                            },
                            {
                                "question": "简述该病人的治疗原则。",
                                "answer": "嵌顿性疝原则上需要紧急手术治疗。",
                            },
                        ],
                    }
                ],
            )
            exam_row = {
                "exam_type": "医师考试",
                "exam_class": "规培结业",
                "exam_subject": "临床病理科",
                "question": "HIV患者最常感染的是下列哪种肺炎",
                "answer": "D",
                "question_type": "单项选择题",
                "option": {
                    "A": "大叶性肺炎",
                    "B": "小叶性肺炎",
                    "C": "非典型肺炎",
                    "D": "卡氏囊虫性肺炎",
                    "E": "病毒性肺炎",
                },
            }
            write_json(cmb_train_path, [exam_row])
            write_json(cmb_val_path, [exam_row])
            write_json(cmb_test_path, [exam_row])

            out_train = tmp_path / "train.jsonl"
            out_benchmark = tmp_path / "benchmark.jsonl"
            out_review = tmp_path / "review.jsonl"
            out_summary = tmp_path / "summary.md"

            argv = [
                "build_medreason_candidates.py",
                "--huatuo-path",
                str(huatuo_path),
                "--shibing-path",
                str(shibing_path),
                "--cmb-clin-path",
                str(cmb_clin_path),
                "--cmb-exam-train-path",
                str(cmb_train_path),
                "--cmb-exam-val-path",
                str(cmb_val_path),
                "--cmb-exam-test-path",
                str(cmb_test_path),
                "--out-train-candidates",
                str(out_train),
                "--out-benchmark-only",
                str(out_benchmark),
                "--out-review-sample",
                str(out_review),
                "--out-summary",
                str(out_summary),
            ]

            with patch.object(sys, "argv", argv):
                MODULE.main()

            train_rows = [json.loads(line) for line in out_train.read_text(encoding="utf-8").splitlines()]
            benchmark_rows = [json.loads(line) for line in out_benchmark.read_text(encoding="utf-8").splitlines()]
            review_rows = [json.loads(line) for line in out_review.read_text(encoding="utf-8").splitlines()]
            summary = out_summary.read_text(encoding="utf-8")

            self.assertEqual(len(train_rows), 4)
            self.assertEqual(len(benchmark_rows), 2)
            self.assertGreaterEqual(len(review_rows), 4)
            self.assertEqual(train_rows[0]["route"], "structured_block_candidate")
            self.assertIn("raw_blocks", train_rows[0])
            self.assertEqual(train_rows[1]["route"], "rewrite_candidate")
            self.assertFalse(train_rows[1]["raw_answer_included"])
            self.assertEqual(train_rows[3]["route"], "exam_rewrite_candidate")
            self.assertTrue(all(row["split"] == "benchmark_only" for row in benchmark_rows))
            self.assertIn("LLM API used: no", summary)
            self.assertIn("train_candidates_written: 4", summary)
            self.assertIn("benchmark_only_written: 2", summary)


if __name__ == "__main__":
    unittest.main()
