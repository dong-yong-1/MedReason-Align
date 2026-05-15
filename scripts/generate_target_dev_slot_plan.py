#!/usr/bin/env python3
"""Generate the MedReason target-dev slot plan v1."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


OUT_DIR = Path("data/medreason")
JSONL_PATH = OUT_DIR / "target_dev_slot_plan_v1.jsonl"
SUMMARY_PATH = OUT_DIR / "target_dev_slot_plan_v1_summary.md"


SYSTEM_DIFFS = {
    "respiratory": ["急性支气管炎", "社区获得性肺炎", "肺结核"],
    "cardiovascular": ["急性冠脉综合征", "主动脉夹层", "肺栓塞"],
    "gastrointestinal": ["消化性溃疡", "急性胆囊炎", "急性胰腺炎"],
    "neurology": ["脑卒中", "短暂性脑缺血发作", "低血糖发作"],
    "infection": ["细菌感染", "病毒感染", "脓毒症"],
    "endocrine_metabolic": ["糖尿病急症", "甲状腺功能异常", "电解质紊乱"],
    "renal_urologic": ["急性膀胱炎", "急性肾盂肾炎", "泌尿系结石"],
    "hematology_rheumatology": ["贫血", "血栓性疾病", "风湿免疫病活动"],
    "drug_related": ["药物不良反应", "药物相互作用", "原发疾病进展"],
    "general_internal_medicine": ["心肺疾病加重", "感染", "代谢或药物相关问题"],
}


RISK_FLAGS = {
    "routine": ["当前不应出现明显红旗信号；若出现症状快速加重、持续高热、胸痛、气促或意识异常，应及时就医"],
    "urgent": ["需要尽快线下就医；若出现生命体征异常、症状进行性加重或关键功能受损，应升级急诊评估"],
    "emergent": ["存在急诊红旗风险；若症状持续或伴低血压、意识改变、严重气促、神经功能缺损或活动性出血，应立即急诊处理"],
}


SYSTEM_PLANS = [
    {
        "system": "respiratory",
        "prefix": "resp",
        "risk": {"routine": 9, "urgent": 7, "emergent": 2},
        "difficulty": {"typical_single": 7, "common_confusable": 7, "atypical_presentation": 3, "noisy_or_incomplete": 1},
        "input_style": {"patient_narrative": 7, "semi_structured_case": 7, "exam_case": 4},
        "completeness": {"complete": 5, "mostly_complete": 9, "partially_missing": 4},
        "scenarios": [
            ("普通病毒性上呼吸道感染", ["流涕咽痛", "轻微咳嗽", "无明显气促"]),
            ("过敏性鼻炎与感冒鉴别", ["鼻痒喷嚏", "季节或接触诱因", "无发热"]),
            ("感染后咳嗽", ["咳嗽迁延", "上感后出现", "无明显肺部红旗"]),
            ("急性支气管炎低风险", ["咳嗽咳痰", "短病程", "肺部体征轻"]),
            ("轻症哮喘控制不佳", ["喘息反复", "诱因相关", "吸入药使用史"]),
            ("慢性咳嗽待查", ["咳嗽超过数周", "反酸或鼻后滴漏线索", "无咯血消瘦"]),
            ("急性咽炎或扁桃体炎", ["咽痛", "吞咽不适", "咽部体征"]),
            ("低风险胸闷伴呼吸不适", ["胸闷", "活动耐量基本正常", "血氧正常"]),
            ("轻症鼻窦炎", ["鼻塞流脓涕", "面部胀痛", "病程超过数日"]),
            ("社区获得性肺炎", ["发热咳痰", "肺部湿啰音", "影像待查"]),
            ("慢阻肺急性加重", ["基础慢阻肺", "咳痰气促加重", "血氧或感染线索"]),
            ("哮喘急性发作", ["喘息气促", "过敏或运动诱因", "峰流速或血氧线索"]),
            ("肺结核鉴别", ["慢性咳嗽", "盗汗或消瘦", "接触史或影像线索"]),
            ("少量咯血待查", ["咯血", "咳嗽或胸痛", "感染/结核/肿瘤风险"]),
            ("胸膜炎样胸痛", ["吸气相关胸痛", "发热或咳嗽", "胸部影像待查"]),
            ("老年肺炎非典型表现", ["精神食欲变差", "咳嗽不典型", "老年基础病"]),
            ("重症肺炎低氧风险", ["高热咳痰", "气促", "血氧下降"]),
            ("肺栓塞样气促胸痛", ["突发气促胸痛", "血栓风险", "血氧下降或心率快"]),
        ],
    },
    {
        "system": "cardiovascular",
        "prefix": "cardio",
        "risk": {"routine": 5, "urgent": 5, "emergent": 8},
        "difficulty": {"typical_single": 5, "common_confusable": 7, "atypical_presentation": 4, "noisy_or_incomplete": 2},
        "input_style": {"patient_narrative": 7, "semi_structured_case": 6, "exam_case": 5},
        "completeness": {"complete": 4, "mostly_complete": 8, "partially_missing": 5, "critically_missing": 1},
        "scenarios": [
            ("普通胸痛低风险", ["短暂刺痛", "与活动关系弱", "无出汗气促"]),
            ("稳定型心绞痛评估", ["活动诱发胸闷", "休息缓解", "心血管危险因素"]),
            ("心悸早搏疑似", ["阵发心悸", "无晕厥", "心电图待查"]),
            ("血压控制不佳", ["血压升高", "头晕或头痛", "用药依从性"]),
            ("轻度下肢水肿", ["双下肢水肿", "无明显气促", "心肾肝鉴别"]),
            ("心力衰竭加重", ["气短", "不能平卧", "下肢水肿"]),
            ("快速房颤相关症状", ["心悸胸闷", "脉搏不齐", "房颤病史"]),
            ("心绞痛近期加重", ["胸闷频率增加", "活动耐量下降", "危险因素"]),
            ("高血压亚急症", ["明显血压升高", "头痛胸闷", "暂无靶器官损害证据"]),
            ("晕厥后心源性风险评估", ["短暂意识丧失", "活动或心悸相关", "心电图待查"]),
            ("急性冠脉综合征", ["压榨样胸痛", "放射痛或出汗", "危险因素"]),
            ("主动脉夹层", ["突发撕裂样胸背痛", "高血压病史", "双侧血压差或神经症状"]),
            ("肺栓塞胸痛鉴别", ["突发胸痛气促", "制动或血栓风险", "心率快或低氧"]),
            ("心源性晕厥", ["晕厥", "心悸或胸痛先兆", "心脏病史"]),
            ("急性心衰肺水肿", ["严重气促", "不能平卧", "粉红泡沫痰或低氧"]),
            ("高血压急症", ["血压显著升高", "胸痛/神经症状/肾损害线索", "靶器官风险"]),
            ("恶性室性心律失常风险", ["心悸晕厥", "结构性心脏病", "心电图异常待查"]),
            ("心包填塞疑似", ["胸闷气促", "低血压", "颈静脉怒张或心音低钝线索"]),
        ],
    },
    {
        "system": "gastrointestinal",
        "prefix": "gi",
        "risk": {"routine": 6, "urgent": 6, "emergent": 3},
        "difficulty": {"typical_single": 6, "common_confusable": 5, "atypical_presentation": 3, "noisy_or_incomplete": 1},
        "input_style": {"patient_narrative": 6, "semi_structured_case": 5, "exam_case": 4},
        "completeness": {"complete": 4, "mostly_complete": 7, "partially_missing": 4},
        "scenarios": [
            ("功能性消化不良", ["餐后饱胀", "上腹不适", "无消瘦黑便"]),
            ("胃食管反流病", ["反酸烧心", "平卧加重", "胸痛鉴别"]),
            ("急性胃肠炎轻症", ["腹泻呕吐", "饮食诱因", "无脱水红旗"]),
            ("胆囊结石症状", ["右上腹痛", "油腻饮食相关", "发热不明显"]),
            ("便秘相关腹痛", ["排便减少", "腹胀", "无肠梗阻红旗"]),
            ("肠易激综合征", ["反复腹痛", "排便相关", "无报警症状"]),
            ("急性胆囊炎", ["右上腹痛", "发热", "Murphy征或影像待查"]),
            ("急性胰腺炎", ["上腹痛向背部放射", "饮酒或胆石风险", "淀粉酶待查"]),
            ("炎症性肠病活动", ["腹痛腹泻", "黏液血便", "体重下降或贫血"]),
            ("阑尾炎鉴别", ["转移性右下腹痛", "发热", "压痛反跳痛线索"]),
            ("消化性溃疡活动", ["上腹痛节律性", "NSAID或幽门螺杆菌风险", "黑便待排"]),
            ("肝炎黄疸", ["乏力纳差", "尿黄皮肤黄", "肝功能待查"]),
            ("上消化道出血", ["黑便或呕血", "头晕乏力", "血压心率异常"]),
            ("肠梗阻", ["腹痛腹胀", "停止排气排便", "呕吐"]),
            ("急腹症腹膜炎风险", ["持续剧烈腹痛", "腹膜刺激征", "发热或休克线索"]),
        ],
    },
    {
        "system": "neurology",
        "prefix": "neuro",
        "risk": {"routine": 4, "urgent": 4, "emergent": 4},
        "difficulty": {"typical_single": 3, "common_confusable": 4, "atypical_presentation": 4, "noisy_or_incomplete": 1},
        "input_style": {"patient_narrative": 5, "semi_structured_case": 4, "exam_case": 3},
        "completeness": {"complete": 3, "mostly_complete": 5, "partially_missing": 3, "critically_missing": 1},
        "scenarios": [
            ("偏头痛", ["反复头痛", "畏光恶心", "神经缺损缺如"]),
            ("紧张型头痛", ["双侧紧箍样痛", "压力相关", "无红旗头痛"]),
            ("良性位置性眩晕", ["体位诱发眩晕", "短暂发作", "无神经缺损"]),
            ("周围神经麻木", ["肢端麻木", "慢性进展", "糖尿病或颈腰椎线索"]),
            ("癫痫首次发作", ["抽搐或意识丧失", "发作后恢复", "诱因待查"]),
            ("亚急性肢体无力", ["肢体无力", "进展时间", "肌病/神经病鉴别"]),
            ("头痛伴发热鉴别", ["发热头痛", "颈项强直待查", "感染风险"]),
            ("短暂性脑缺血发作高风险", ["短暂偏瘫或失语", "症状已缓解", "血管危险因素"]),
            ("急性缺血性脑卒中", ["突发言语不清", "单侧无力", "发病时间窗"]),
            ("脑出血", ["突发头痛呕吐", "高血压", "意识或神经缺损"]),
            ("蛛网膜下腔出血", ["雷击样头痛", "呕吐或颈强直", "起病突然"]),
            ("脊髓压迫风险", ["背痛", "进行性无力", "大小便功能异常"]),
        ],
    },
    {
        "system": "infection",
        "prefix": "infect",
        "risk": {"routine": 4, "urgent": 5, "emergent": 3},
        "difficulty": {"typical_single": 4, "common_confusable": 4, "atypical_presentation": 3, "noisy_or_incomplete": 1},
        "input_style": {"patient_narrative": 5, "semi_structured_case": 4, "exam_case": 3},
        "completeness": {"complete": 3, "mostly_complete": 5, "partially_missing": 3, "critically_missing": 1},
        "scenarios": [
            ("普通发热上感", ["低热", "咽痛流涕", "精神尚可"]),
            ("轻症胃肠感染", ["腹泻呕吐", "饮食诱因", "无脱水"]),
            ("单纯皮肤软组织感染", ["局部红肿痛", "范围小", "无全身中毒"]),
            ("旅行后低热观察", ["旅行史", "低热乏力", "无红旗"]),
            ("肺炎发热", ["高热咳痰", "肺部体征", "影像待查"]),
            ("肾盂肾炎发热", ["发热腰痛", "尿路刺激征", "尿检待查"]),
            ("蜂窝织炎进展", ["红肿范围扩大", "发热", "糖尿病或外伤"]),
            ("发热伴皮疹", ["发热", "皮疹形态", "药物或感染暴露"]),
            ("免疫低下发热", ["化疗/激素/免疫抑制", "发热", "感染灶不清"]),
            ("脓毒症风险", ["高热寒战", "低血压或心率快", "尿量少或精神差"]),
            ("脑膜炎风险", ["发热头痛", "颈项强直", "意识改变"]),
            ("感染性休克风险", ["感染表现", "低血压", "灌注不足"]),
        ],
    },
    {
        "system": "endocrine_metabolic",
        "prefix": "endo",
        "risk": {"routine": 4, "urgent": 3, "emergent": 2},
        "difficulty": {"typical_single": 4, "common_confusable": 3, "atypical_presentation": 1, "noisy_or_incomplete": 1},
        "input_style": {"patient_narrative": 4, "semi_structured_case": 3, "exam_case": 2},
        "completeness": {"complete": 3, "mostly_complete": 4, "partially_missing": 2},
        "scenarios": [
            ("甲亢症状初筛", ["心悸怕热", "体重下降", "甲功待查"]),
            ("甲减乏力", ["乏力怕冷", "水肿或便秘", "甲功待查"]),
            ("2型糖尿病控制不佳", ["多饮多尿", "血糖升高", "用药依从性"]),
            ("肥胖代谢综合征", ["体重增加", "血脂血糖异常", "生活方式风险"]),
            ("低血糖反复", ["出汗心慌", "进食后缓解", "降糖药使用"]),
            ("糖尿病足感染早期", ["足部破溃", "糖尿病史", "局部感染线索"]),
            ("肾上腺功能不足疑似", ["乏力低血压", "低钠或色素沉着线索", "激素使用史"]),
            ("糖尿病酮症酸中毒", ["高血糖", "腹痛呕吐", "酮体或酸中毒待查"]),
            ("高渗高血糖状态", ["极高血糖", "脱水", "意识改变"]),
        ],
    },
    {
        "system": "renal_urologic",
        "prefix": "renal",
        "risk": {"routine": 4, "urgent": 4, "emergent": 1},
        "difficulty": {"typical_single": 4, "common_confusable": 3, "atypical_presentation": 1, "noisy_or_incomplete": 1},
        "input_style": {"patient_narrative": 4, "semi_structured_case": 3, "exam_case": 2},
        "completeness": {"complete": 3, "mostly_complete": 4, "partially_missing": 2},
        "scenarios": [
            ("单纯膀胱炎", ["尿频尿急尿痛", "无发热腰痛", "尿常规待查"]),
            ("蛋白尿血尿复查", ["体检异常", "无明显症状", "肾功能待查"]),
            ("肾结石低风险", ["阵发腰痛", "血尿线索", "无感染红旗"]),
            ("前列腺增生尿频", ["夜尿增多", "排尿困难", "老年男性"]),
            ("急性肾盂肾炎", ["发热腰痛", "尿路刺激征", "肾区叩痛"]),
            ("输尿管结石伴感染", ["绞痛", "发热", "尿检异常"]),
            ("急性肾损伤", ["尿量减少", "肾毒性药物或脱水", "肌酐待查"]),
            ("肉眼血尿待查", ["肉眼血尿", "疼痛或无痛", "肿瘤/结石/感染鉴别"]),
            ("高钾或少尿急性肾衰风险", ["少尿", "乏力心悸", "肾功能或高钾线索"]),
        ],
    },
    {
        "system": "hematology_rheumatology",
        "prefix": "heme",
        "risk": {"routine": 3, "urgent": 2, "emergent": 1},
        "difficulty": {"typical_single": 2, "common_confusable": 2, "atypical_presentation": 1, "noisy_or_incomplete": 1},
        "input_style": {"patient_narrative": 2, "semi_structured_case": 2, "exam_case": 2},
        "completeness": {"complete": 2, "mostly_complete": 2, "partially_missing": 1, "critically_missing": 1},
        "scenarios": [
            ("缺铁性贫血", ["乏力头晕", "月经过多或消化道风险", "血红蛋白待查"]),
            ("关节痛风湿鉴别", ["多关节痛", "晨僵或肿胀", "炎症指标待查"]),
            ("轻度血小板异常复查", ["体检血小板异常", "出血症状缺如", "复查需求"]),
            ("深静脉血栓疑似", ["单侧下肢肿痛", "制动或肿瘤风险", "D-二聚体/超声待查"]),
            ("系统性红斑狼疮活动", ["皮疹关节痛", "蛋白尿或发热", "自身抗体线索"]),
            ("中性粒细胞减少伴发热", ["发热", "化疗或血液病史", "白细胞/中性粒低"]),
        ],
    },
    {
        "system": "drug_related",
        "prefix": "drug",
        "risk": {"routine": 3, "urgent": 3},
        "difficulty": {"typical_single": 2, "common_confusable": 2, "atypical_presentation": 1, "noisy_or_incomplete": 1},
        "input_style": {"patient_narrative": 3, "semi_structured_case": 2, "exam_case": 1},
        "completeness": {"complete": 1, "mostly_complete": 3, "partially_missing": 2},
        "scenarios": [
            ("抗生素胃肠反应", ["新近抗生素", "腹泻恶心", "感染本身鉴别"]),
            ("降压药体位性头晕", ["起立头晕", "降压药调整", "血压变化"]),
            ("他汀相关肌痛", ["肌痛乏力", "他汀用药", "肌酶待查"]),
            ("抗凝相关出血倾向", ["抗凝药", "瘀斑或黑便", "凝血/血红蛋白待查"]),
            ("降糖药低血糖风险", ["心慌出汗", "进食少或用药过量", "血糖低"]),
            ("NSAID相关胃出血风险", ["止痛药使用", "黑便或上腹痛", "溃疡史"]),
        ],
    },
    {
        "system": "general_internal_medicine",
        "prefix": "mixed",
        "risk": {"routine": 12, "urgent": 3},
        "difficulty": {"typical_single": 5, "common_confusable": 5, "atypical_presentation": 3, "noisy_or_incomplete": 2},
        "input_style": {"patient_narrative": 5, "semi_structured_case": 6, "exam_case": 4},
        "completeness": {"complete": 2, "mostly_complete": 7, "partially_missing": 4, "critically_missing": 2},
        "scenarios": [
            ("老年乏力多病共存", ["乏力食欲差", "多种慢病", "用药和感染筛查"]),
            ("不明原因体重下降", ["体重下降", "食欲或消化症状", "肿瘤/内分泌/感染鉴别"]),
            ("慢性乏力贫血评估", ["乏力头晕", "贫血线索", "慢病或出血风险"]),
            ("头晕伴血压波动", ["头晕", "血压变化", "用药和心律鉴别"]),
            ("食欲差伴多药使用", ["纳差", "多药并用", "肝肾功能待查"]),
            ("慢性咳嗽伴反酸", ["咳嗽迁延", "反酸烧心", "肺部红旗排查"]),
            ("反复低热待查", ["低热", "乏力", "感染/风湿/肿瘤鉴别"]),
            ("多部位疼痛风湿鉴别", ["多部位痛", "晨僵或乏力", "炎症指标待查"]),
            ("轻度水肿多因素", ["水肿", "心肾肝药物因素", "气促或尿量线索"]),
            ("失眠乏力伴内科鉴别", ["失眠乏力", "甲状腺或贫血线索", "情绪因素"]),
            ("老年跌倒风险评估", ["跌倒", "头晕或肌力下降", "用药和神经系统筛查"]),
            ("轻度腹痛合并糖尿病", ["腹痛", "糖尿病史", "感染或代谢风险"]),
            ("老年气短合并心肺病", ["气短加重", "心肺基础病", "血氧或水肿线索"]),
            ("发热伴基础病恶化", ["发热", "慢病基础", "精神食欲变差"]),
            ("多病共存用药后意识波动", ["意识波动", "多药使用", "感染/代谢/药物鉴别"]),
        ],
    },
]


def expand_quota(quota: dict[str, int]) -> list[str]:
    labels: list[str] = []
    for label, count in quota.items():
        labels.extend([label] * count)
    return labels


def reasoning_types(system: str, risk: str, difficulty: str, completeness: str) -> list[str]:
    items = ["diagnosis"]
    if difficulty in {"common_confusable", "atypical_presentation", "noisy_or_incomplete"}:
        items.append("differential")
    if risk in {"urgent", "emergent"}:
        items.extend(["triage", "risk_detection"])
    if completeness in {"partially_missing", "critically_missing"}:
        items.append("test_recommendation")
    if system == "drug_related":
        items.append("medication_related_reasoning")
    if system == "general_internal_medicine":
        items.append("comorbidity_reasoning")
    return list(dict.fromkeys(items))


def build_slots() -> list[dict]:
    slots: list[dict] = []
    for plan in SYSTEM_PLANS:
        system = plan["system"]
        scenarios = plan["scenarios"]
        count = len(scenarios)
        risk_labels = expand_quota(plan["risk"])
        difficulty_labels = expand_quota(plan["difficulty"])
        input_labels = expand_quota(plan["input_style"])
        completeness_labels = expand_quota(plan["completeness"])

        assert len(risk_labels) == count, (system, "risk", len(risk_labels), count)
        assert len(difficulty_labels) == count, (system, "difficulty", len(difficulty_labels), count)
        assert len(input_labels) == count, (system, "input_style", len(input_labels), count)
        assert len(completeness_labels) == count, (system, "completeness", len(completeness_labels), count)

        for idx, ((scenario, clues), risk, difficulty, input_style, completeness) in enumerate(
            zip(scenarios, risk_labels, difficulty_labels, input_labels, completeness_labels),
            start=1,
        ):
            slot_id = f"slot_{plan['prefix']}_{idx:03d}"
            slots.append(
                {
                    "slot_id": slot_id,
                    "split": "target_dev",
                    "primary_system": system,
                    "disease_scenario": scenario,
                    "risk_level": risk,
                    "difficulty_level": difficulty,
                    "input_style": input_style,
                    "completeness_level": completeness,
                    "reasoning_type": reasoning_types(system, risk, difficulty, completeness),
                    "must_have_evidence": clues + ["病程、诱因或危险因素", "至少1项体征、检查结果或明确缺失信息"],
                    "must_have_differentials": SYSTEM_DIFFS[system],
                    "must_have_risk_flags": RISK_FLAGS[risk],
                    "case_writing_notes": [
                        "case_text 不直接写出最终诊断",
                        "schema_target 必须使用 schema v1 五字段",
                        "diagnostic_basis 至少2条且能从 case_text 找到依据",
                    ],
                    "review_status": "planned",
                    "version": "v1.0",
                }
            )
    return slots


def write_outputs(slots: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for slot in slots:
            f.write(json.dumps(slot, ensure_ascii=False, separators=(",", ":")) + "\n")

    sections = [
        "# target_dev_slot_plan_v1 分布统计",
        "",
        "这份文件记录 `MedReason-Align` 的 120 条 `target_dev` 槽位分布计划。",
        "",
        f"- total_slots: {len(slots)}",
        f"- jsonl_path: `{JSONL_PATH}`",
        "",
    ]
    for key in ["primary_system", "risk_level", "difficulty_level", "input_style", "completeness_level"]:
        counter = Counter(slot[key] for slot in slots)
        sections.append(f"## {key}")
        sections.append("")
        for label, count in sorted(counter.items()):
            sections.append(f"- {label}: {count}")
        sections.append("")

    SUMMARY_PATH.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    slots = build_slots()
    assert len(slots) == 120, len(slots)
    assert len({slot["slot_id"] for slot in slots}) == 120
    write_outputs(slots)
    print(f"wrote {len(slots)} slots to {JSONL_PATH}")
    print(f"wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
