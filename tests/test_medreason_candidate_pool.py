import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "organize_candidate_pool.py"
SPEC = importlib.util.spec_from_file_location("organize_candidate_pool", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
sys.modules.setdefault("organize_candidate_pool", MODULE)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _candidate(sample_id, case_text, schema_target, meta=None):
    return {
        "sample_id": sample_id,
        "split": "train_candidate",
        "case_text": case_text,
        "schema_target": schema_target,
        "meta": meta
        or {
            "primary_system": "cardiovascular",
            "risk_level": "urgent",
            "difficulty_level": "common_confusable",
            "input_style": "semi_structured_case",
            "completeness_level": "mostly_complete",
            "reasoning_type": ["diagnosis", "differential", "risk_detection"],
            "quality_tier": "silver",
            "review_status": "draft",
        },
    }


def test_organize_candidates_applies_duplicate_leakage_quality_and_review_rules(tmp_path):
    candidate_path = tmp_path / "candidates.jsonl"
    target_dev_path = tmp_path / "target_dev.jsonl"

    duplicate_winner = _candidate(
        "cand_keep",
        "患者，男，61岁。胸闷胸痛2小时，放射至左肩，伴出汗恶心。既往高血压和吸烟史。",
        {
            "primary_diagnosis": "急性冠脉综合征可能性大",
            "diagnostic_basis": [
                "胸骨后压榨样疼痛伴出汗恶心，符合缺血性胸痛特点",
                "高血压和长期吸烟史提示心血管危险因素较高",
            ],
            "differential_diagnoses": ["主动脉夹层", "肺栓塞"],
            "recommended_actions": ["建议尽快急诊就医并完善心电图和肌钙蛋白检查"],
            "risk_flags": ["持续胸痛不缓解时应立即急诊处理"],
        },
        meta={
            "primary_system": "cardiovascular",
            "risk_level": "emergent",
            "difficulty_level": "common_confusable",
            "input_style": "semi_structured_case",
            "completeness_level": "mostly_complete",
            "reasoning_type": ["diagnosis", "differential", "risk_detection"],
            "quality_tier": "gold",
            "review_status": "reviewed",
        },
    )
    duplicate_loser = _candidate(
        "cand_dup",
        "患者，男，61岁。胸闷胸痛2小时，放射至左肩，伴出汗恶心。既往高血压和吸烟史。",
        {
            "primary_diagnosis": "急性冠脉综合征可能性大",
            "diagnostic_basis": [
                "胸骨后压榨样疼痛伴出汗恶心，符合缺血性胸痛特点",
                "高血压和长期吸烟史提示心血管危险因素较高",
            ],
            "differential_diagnoses": ["主动脉夹层", "肺栓塞"],
            "recommended_actions": ["建议尽快急诊就医并完善心电图和肌钙蛋白检查"],
            "risk_flags": ["持续胸痛不缓解时应立即急诊处理"],
        },
    )
    leakage_candidate = _candidate(
        "cand_leak",
        "患者，女，45岁。咳嗽咳痰7天，近3天发热，痰黄，伴轻度胸闷。右下肺可闻及湿啰音。",
        {
            "primary_diagnosis": "社区获得性肺炎可能性大",
            "diagnostic_basis": [
                "发热咳嗽黄痰提示下呼吸道感染",
                "右下肺湿啰音支持肺部感染可能",
            ],
            "differential_diagnoses": ["急性支气管炎", "流行性感冒"],
            "recommended_actions": ["建议尽快线下就诊并完善血常规和胸片检查"],
            "risk_flags": ["若出现呼吸困难或持续高热，应及时急诊处理"],
        },
        meta={
            "primary_system": "respiratory",
            "risk_level": "urgent",
            "difficulty_level": "common_confusable",
            "input_style": "semi_structured_case",
            "completeness_level": "mostly_complete",
            "reasoning_type": ["diagnosis", "differential", "risk_detection"],
            "quality_tier": "silver",
            "review_status": "draft",
        },
    )
    low_quality_candidate = _candidate(
        "cand_low_quality",
        "患者，男，67岁。高热伴心率快，今天精神差、尿量减少。",
        {
            "primary_diagnosis": "重症感染可能",
            "diagnostic_basis": ["高热提示感染"],
            "differential_diagnoses": ["病毒感染"],
            "recommended_actions": ["注意休息"],
            "risk_flags": [],
        },
        meta={
            "primary_system": "infection",
            "risk_level": "emergent",
            "difficulty_level": "common_confusable",
            "input_style": "patient_narrative",
            "completeness_level": "partially_missing",
            "reasoning_type": ["diagnosis", "triage", "risk_detection"],
            "quality_tier": "silver",
            "review_status": "draft",
        },
    )
    review_candidate = _candidate(
        "cand_review",
        "患者，女，36岁。发热、尿频尿急尿痛3天，近1天左侧腰痛。",
        {
            "primary_diagnosis": "急性肾盂肾炎可能性大",
            "diagnostic_basis": [
                "尿频尿急尿痛提示泌尿系统感染",
                "发热伴腰痛支持上尿路感染可能",
            ],
            "differential_diagnoses": ["急性肾盂肾炎", "急性膀胱炎"],
            "recommended_actions": ["建议尽快线下就医并完善尿常规和尿培养"],
            "risk_flags": ["若高热不退或寒战，应及时急诊处理"],
        },
        meta={
            "primary_system": "renal_urologic",
            "risk_level": "urgent",
            "difficulty_level": "common_confusable",
            "input_style": "semi_structured_case",
            "completeness_level": "mostly_complete",
            "reasoning_type": ["diagnosis", "differential", "risk_detection"],
            "quality_tier": "silver",
            "review_status": "draft",
        },
    )

    _write_jsonl(
        candidate_path,
        [duplicate_winner, duplicate_loser, leakage_candidate, low_quality_candidate, review_candidate],
    )
    _write_jsonl(target_dev_path, [leakage_candidate])

    results = MODULE.organize_candidates([candidate_path], target_dev_path, None)
    by_id = {row["sample_id"]: row for row in results}

    assert by_id["cand_keep"]["final_status"] == "accepted"
    assert by_id["cand_dup"]["final_status"] == "rejected_duplicate"
    assert by_id["cand_dup"]["audit"]["duplicate_of"] == "cand_keep"
    assert by_id["cand_leak"]["final_status"] == "rejected_leakage"
    assert by_id["cand_leak"]["hit_rules"] == ["R-LEAK-004"]
    assert by_id["cand_low_quality"]["final_status"] == "rejected_low_quality"
    assert "R-QUAL-005" in by_id["cand_low_quality"]["hit_rules"] or "R-QUAL-004" in by_id["cand_low_quality"]["hit_rules"]
    assert by_id["cand_review"]["final_status"] == "needs_manual_review"
    assert "R-REVIEW-005" in by_id["cand_review"]["hit_rules"]


def test_main_writes_expected_output_files(tmp_path, monkeypatch):
    candidate_path = tmp_path / "candidates.jsonl"
    output_dir = tmp_path / "organized"

    _write_jsonl(
        candidate_path,
        [
            _candidate(
                "cand_ok",
                "患者，男，58岁。近1天黑便2次，伴头晕乏力。既往胃溃疡史。",
                {
                    "primary_diagnosis": "上消化道出血可能性大",
                    "diagnostic_basis": [
                        "黑便提示消化道出血可能",
                        "头晕乏力结合胃溃疡史支持失血风险",
                    ],
                    "differential_diagnoses": ["消化性溃疡活动", "下消化道出血"],
                    "recommended_actions": ["建议立即急诊就医并完善血常规和胃镜评估"],
                    "risk_flags": ["若黑便增多或血压下降，应立即急诊处理"],
                },
                meta={
                    "primary_system": "gastrointestinal",
                    "risk_level": "emergent",
                    "difficulty_level": "typical_single",
                    "input_style": "semi_structured_case",
                    "completeness_level": "mostly_complete",
                    "reasoning_type": ["diagnosis", "triage", "risk_detection"],
                    "quality_tier": "silver",
                    "review_status": "draft",
                },
            )
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "organize_candidate_pool.py",
            "--input-paths",
            str(candidate_path),
            "--output-dir",
            str(output_dir),
            "--target-dev-path",
            "",
        ],
    )

    MODULE.main()

    assert (output_dir / "all_candidates.jsonl").exists()
    assert (output_dir / "accepted.jsonl").exists()
    assert (output_dir / "needs_manual_review.jsonl").exists()
    assert (output_dir / "rejected.jsonl").exists()
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "accepted: 1" in summary
