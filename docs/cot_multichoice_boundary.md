# CoT/Mixed 多选边界诊断报告

## 1. 背景

CoT/Mixed 在 CMExam-test-multi 上低于 Optimized Direct SFT。已有诊断基本排除了答案抽取失败和输出截断，本报告进一步检查 CoT/Mixed 是否改变了多选题的 precision-recall tradeoff。

## 2. 当前主结果

| 模型 | CMExam-test-single | CMExam-test-multi |
|---|---:|---:|
| Base Qwen2.5-7B-Instruct | 0.8295 | 0.5245 |
| Optimized Direct SFT | 0.8521 | 0.6373 |
| CoT SFT / Mixed | 0.8408 | 0.6078 |

## 3. 核心结论摘要

- CoT/Mixed option recall = `0.918723`，Direct option recall = `0.892598`。
- CoT/Mixed option precision = `0.904286`，Direct option precision = `0.931818`。
- CoT/Mixed extra option rate = `0.254902`，Direct extra option rate = `0.176471`。
- CoT/Mixed avg_pred_answer_len = `3.431373`，Direct = `3.235294`，gold 平均长度 = `3.377451`。
- 错误重叠显示 Direct 正确但 CoT/Mixed 因额外选项出错的样本数为 `11`。

## 4. Option-level 指标

| 模型 | exact_match | option_precision | option_recall | option_f1 | extra_option_rate | missing_option_rate | avg_pred_answer_len | avg_gold_answer_len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct SFT | 0.632353 | 0.931818 | 0.892598 | 0.911786 | 0.176471 | 0.254902 | 3.235294 | 3.377451 |
| CoT/Mixed | 0.607843 | 0.904286 | 0.918723 | 0.911447 | 0.254902 | 0.205882 | 3.431373 | 3.377451 |

## 5. 按 gold answer 长度分组

### Direct SFT

| 模型 | exact_match | option_precision | option_recall | option_f1 | extra_option_rate | missing_option_rate | avg_pred_answer_len | avg_gold_answer_len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.65 | 0.823529 | 0.875 | 0.848485 | 0.25 | 0.2 | 2.125 | 2 |
| 3 | 0.633803 | 0.912037 | 0.924883 | 0.918415 | 0.211268 | 0.183099 | 3.042254 | 3 |
| >=4 | 0.623656 | 0.969359 | 0.878788 | 0.921854 | 0.11828 | 0.333333 | 3.860215 | 4.258065 |

### CoT/Mixed

| 模型 | exact_match | option_precision | option_recall | option_f1 | extra_option_rate | missing_option_rate | avg_pred_answer_len | avg_gold_answer_len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.65 | 0.789474 | 0.9375 | 0.857143 | 0.325 | 0.075 | 2.375 | 2 |
| 3 | 0.577465 | 0.868421 | 0.929577 | 0.897959 | 0.309859 | 0.197183 | 3.211268 | 3 |
| >=4 | 0.612903 | 0.954907 | 0.909091 | 0.931436 | 0.182796 | 0.268817 | 4.053763 | 4.258065 |

## 6. 预测答案长度分布

- Direct pred_answer_len_distribution: `{"0": 1, "1": 7, "2": 44, "3": 69, "4": 57, "5": 26}`
- CoT/Mixed pred_answer_len_distribution: `{"1": 4, "2": 37, "3": 71, "4": 51, "5": 41}`
- gold_answer_len_distribution: `{"2": 40, "3": 71, "4": 69, "5": 24}`
- Direct pred_minus_gold_len_distribution: `{"+1": 18, "+2": 5, "+3_or_more": 1, "-1": 30, "-2_or_less": 13, "0": 137}`
- CoT/Mixed pred_minus_gold_len_distribution: `{"+1": 26, "+2": 10, "+3_or_more": 2, "-1": 27, "-2_or_less": 6, "0": 133}`

## 7. CoT 训练数据分布分析

{
  "direct_train": {
    "path": "data/sft/cmb_optimized_direct/cmb_sft_optimized_direct.jsonl",
    "total": 5000,
    "by_format": {
      "direct_answer": {
        "total": 5000,
        "multi_count": 241,
        "multi_ratio": 0.0482,
        "avg_gold_answer_len": 1.1194,
        "gold_answer_len_distribution": {
          "1": 4760,
          "2": 51,
          "3": 66,
          "4": 78,
          "5": 45
        },
        "avg_question_length": 39.6704,
        "case_rich_ratio": 0.182,
        "option_confusing_ratio": 0.2364,
        "high_cot_worthiness_ratio": 0.3748
      }
    },
    "cot_multichoice_rationale_examples": [],
    "rationale_quality_summary": {
      "cot_multi_total": 0,
      "explains_all_correct_rate": 0.0,
      "mentions_wrong_options_rate": 0.0,
      "fan_select_phrase_rate": 0.0,
      "only_explains_correct_rate": 0.0,
      "avg_rationale_length": 0.0
    }
  },
  "cot_mixed_train": {
    "path": "data/sft/cmb_cot_mixed/cmb_sft_cot_mixed.jsonl",
    "total": 5000,
    "by_format": {
      "cot_teacher_deepseek": {
        "total": 986,
        "multi_count": 678,
        "multi_ratio": 0.687627,
        "avg_gold_answer_len": 2.598377,
        "gold_answer_len_distribution": {
          "1": 308,
          "2": 196,
          "3": 190,
          "4": 189,
          "5": 82,
          "6": 21
        },
        "avg_question_length": 78.711968,
        "case_rich_ratio": 0.72211,
        "option_confusing_ratio": 0.39858,
        "high_cot_worthiness_ratio": 0.876268
      },
      "direct_answer": {
        "total": 4014,
        "multi_count": 238,
        "multi_ratio": 0.059292,
        "avg_gold_answer_len": 1.14574,
        "gold_answer_len_distribution": {
          "1": 3776,
          "2": 44,
          "3": 76,
          "4": 83,
          "5": 35
        },
        "avg_question_length": 29.053313,
        "case_rich_ratio": 0.12581,
        "option_confusing_ratio": 0.169407,
        "high_cot_worthiness_ratio": 0.274041
      }
    },
    "cot_ratio": 0.1972,
    "cot_multichoice_rationale_examples": [
      {
        "sample_id": "2269",
        "question": "男性,19岁,左膝、右足跟腱周围肿痛2周,右足跟行走痛1个月来诊。需要追问哪些病史",
        "options": {
          "A": "尿频、尿急、尿痛史",
          "B": "家族史",
          "C": "眼疾史",
          "D": "皮损史",
          "E": "关节、腰背痛",
          "F": "心悸气促史"
        },
        "gold_answer": "ABCDEF",
        "teacher_analysis": "患者青年男性，多部位肌腱附着点肿痛，需考虑脊柱关节炎（如强直性脊柱炎、反应性关节炎等）。追问病史应涵盖：感染诱因（A.尿路感染史提示反应性关节炎）、家族史（B.脊柱关节炎有家族聚集性）、关节外表现（C.眼疾如前葡萄膜炎、D.皮损如银屑病）、中轴及外周关节受累（E.关节腰背痛）、以及心血管系统受累（F.心悸气促可能提示主动脉瓣关闭不全或传导阻滞）。所有选项均为脊柱关节炎相关病史，故全选。",
        "judgment": [
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项",
          "rationale较长"
        ]
      },
      {
        "sample_id": "756",
        "question": "女性,67岁,不慎摔倒,左髋部着地,当即左髋剧痛,不能站立,急诊来院,检查见左下肢缩短,外旋畸形患者最可能的诊断是",
        "options": {
          "A": "左髋关节前脱位",
          "B": "左髋关节后脱位",
          "C": "左髋关节中心脱位",
          "D": "左股骨颈骨折",
          "E": "左股骨干骨折",
          "F": "粗隆间骨折"
        },
        "gold_answer": "DF",
        "teacher_analysis": "老年女性摔倒后髋部剧痛、不能站立，左下肢缩短、外旋畸形，是髋部骨折的典型表现。股骨颈骨折和粗隆间骨折均可出现下肢缩短和外旋畸形，但粗隆间骨折外旋更明显（可达90°），股骨颈骨折外旋角度较小（45°-60°）。髋关节脱位多表现为弹性固定和特殊体位，前脱位呈外展外旋，后脱位呈内收内旋，中心脱位肢体缩短不明显。股骨干骨折多见于强大暴力，畸形以成角、缩短为主，外旋不典型。因此最可能的诊断为股骨颈骨折和粗隆间骨折。",
        "judgment": [
          "未解释所有正确项",
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项",
          "rationale较长"
        ]
      },
      {
        "sample_id": "130",
        "question": "患儿女性,9岁。因水肿、少尿并血尿收入院。起病无诱因,不伴发热,患儿曾对症治疗,颜面及四肢水肿消退,而腹胀日益加重,出现持续肉眼血尿和大量蛋白尿。既往健康。家族无相似病人。体检:营养欠佳,发育中等,精神差,颜面不肿,双下肢中凹陷性水肿。心肺无异常。腹部膨隆,移动性浊音阳性,肝脾未触及,未及异常包块,余无特殊。为了证实诊断,必须做下列哪些检査",
        "options": {
          "A": "血清补体C3",
          "B": "抗核抗体(ANA)",
          "C": "血小板及凝血酶原时间",
          "D": "下腔静脉造影",
          "E": "血浆纤维蛋白原定量"
        },
        "gold_answer": "ABCDE",
        "teacher_analysis": "患儿表现为水肿、少尿、血尿、大量蛋白尿，腹水明显，需考虑肾病综合征及继发性肾小球疾病。血清补体C3有助于鉴别链球菌感染后肾炎或狼疮性肾炎；抗核抗体用于排查系统性红斑狼疮；血小板及凝血酶原时间、血浆纤维蛋白原定量可评估肾病综合征高凝状态及血栓风险；下腔静脉造影可明确有无肾静脉血栓等血管并发症。以上检查均为明确诊断和评估病情所必需。",
        "judgment": [
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项",
          "rationale较长"
        ]
      },
      {
        "sample_id": "538",
        "question": "患者男性,38岁。腹部疼痛4小时入院,患者于4小时前因大量饮酒后突发腹部疼痛,为剑突下持续性疼痛,休息后无缓解,疼痛不随体位而减轻,并伴有呕吐、腹胀,无呕血、黑便,无发热、咳嗽、气促、心悸等不适,患者既往有胃、十二指肠溃疡病史,无外伤史、肝炎病史。应予以哪些检查",
        "options": {
          "A": "腹部B超",
          "B": "血常规",
          "C": "凝血功能",
          "D": "胸片",
          "E": "肾功能",
          "F": "大便常规+隐血试验"
        },
        "gold_answer": "ABCDEF",
        "teacher_analysis": "患者大量饮酒后突发剑突下持续性疼痛，伴呕吐、腹胀，有溃疡病史，需考虑急性胰腺炎、溃疡穿孔等急腹症。腹部B超可评估胰腺、胆囊及腹腔积液；血常规可判断感染或失血；凝血功能为术前准备及评估病情；胸片可排除膈下游离气体等穿孔征象；肾功能评估脏器灌注及全身状况；大便常规+隐血试验可排查消化道出血。以上检查均为急腹症常规评估项目，故全选。",
        "judgment": [
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项",
          "rationale较长"
        ]
      },
      {
        "sample_id": "2499",
        "question": "对截瘫患者应积极处理,防止并发症压疮发生,下列哪几项是正确的",
        "options": {
          "A": "床褥平软,及时料理,保持皮肤清洁干燥",
          "B": "每2小时翻身1次,夜间可适当减少次数,保证患者休息",
          "C": "对骨隆突部位用软垫或气枕垫好,每日用50%乙醇擦洗,滑石粉按摩",
          "D": "对已发生深的压疮要及时剪去坏死组织,勤换敷料",
          "E": "待炎症控制、肉芽新鲜时,做转移皮瓣缝合"
        },
        "gold_answer": "ABCDE",
        "teacher_analysis": "截瘫患者因感觉和运动障碍，压疮预防与处理需综合措施。A项保持床褥平软和皮肤清洁干燥可减少摩擦和潮湿；B项定时翻身（每2小时）是减压关键，夜间适当减少次数以兼顾休息；C项骨隆突处用软垫保护，50%乙醇擦洗和滑石粉按摩可促进局部循环；D项对深度压疮及时清创、勤换敷料可控制感染；E项在炎症控制、肉芽新鲜后行转移皮瓣缝合是修复创面的有效方法。各选项均为压疮防治的正确措施，故全选。",
        "judgment": [
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项",
          "rationale较长"
        ]
      },
      {
        "sample_id": "218",
        "question": "老年男性,65岁,既往有胆囊结石病史,因上腹痛4小时由家属送诊,查体:意识淡漠,血压90/50mmHg,右上腹部肌紧张可能的诊断为",
        "options": {
          "A": "急性胆囊炎",
          "B": "急性胆管炎",
          "C": "急性胰腺炎",
          "D": "感染性休克",
          "E": "急性胆原性胰腺炎",
          "F": "肠系膜静脉血栓"
        },
        "gold_answer": "ABCDEF",
        "teacher_analysis": "老年男性，胆囊结石病史，上腹痛，意识淡漠，血压低（90/50mmHg），右上腹肌紧张。胆囊结石可诱发急性胆囊炎（A），结石嵌顿或感染上行可致急性胆管炎（B），胆石移行可引发急性胰腺炎（C）或胆源性胰腺炎（E）。血压低、意识淡漠提示存在感染性休克（D）。长期胆道感染、炎症或高凝状态可能并发肠系膜静脉血栓（F），虽少见但属可能诊断。所有选项均符合临床表现及病理关联。",
        "judgment": [
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项",
          "rationale较长"
        ]
      },
      {
        "sample_id": "239",
        "question": "血友病患儿反复使用VIII因子替代治疗后出现抗VIII因子抗体,可采用的方法有",
        "options": {
          "A": "增加VIII因子剂量",
          "B": "活化VIIa因子",
          "C": "静脉大剂量丙种球蛋白",
          "D": "免疫抑制剂",
          "E": "链球菌蛋白A"
        },
        "gold_answer": "ABCDE",
        "teacher_analysis": "血友病患儿产生抗VIII因子抑制物后，治疗策略包括：增加VIII因子剂量用于低滴度抑制物；活化VIIa因子可绕过VIII因子途径止血；静脉大剂量丙种球蛋白和免疫抑制剂用于免疫调节清除抗体；链球菌蛋白A可吸附抗体。所有选项均为临床可选用方法。",
        "judgment": [
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "774",
        "question": "做出急性肺损伤(ALI)/急性呼吸窘迫综合征(ARDS)诊断前,宜排除",
        "options": {
          "A": "自发性气胸",
          "B": "大片肺不张",
          "C": "上气道阻塞",
          "D": "急性肺栓塞",
          "E": "心源性肺水肿"
        },
        "gold_answer": "ABCDE",
        "teacher_analysis": "ALI/ARDS诊断需排除其他原因导致的急性呼吸衰竭和双侧肺浸润影。自发性气胸、大片肺不张、上气道阻塞、急性肺栓塞均可引起类似表现，心源性肺水肿需通过临床和血流动力学评估排除，因此全部选项均需排除。",
        "judgment": [
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "137",
        "question": "患者男性,25岁。不慎从4米高处跌下,半小时后被送入急诊室。体检:血压30/15mmHg,神清,气促,面色苍白,四肢发凉,脉细弱,左侧胸压痛明显、胸廓塌陷、有骨擦感及反常呼吸征,左胸见一2cm×2.5cm创口,可听到气体出入创口响声,左侧呼吸音消失,右侧呼吸音减低。体格检查时,还应重点检査哪些项目",
        "options": {
          "A": "气管",
          "B": "胸壁",
          "C": "心、肺",
          "D": "肝、脾",
          "E": "脊柱",
          "F": "颈静脉"
        },
        "gold_answer": "ABCDEF",
        "teacher_analysis": "患者高处坠落伤，出现休克、左侧开放性气胸及连枷胸表现，伤情严重。除胸部损伤外，需全面评估：气管位置可判断纵隔移位；胸壁检查明确损伤范围；心肺检查评估呼吸循环功能；肝脾检查排除腹部闭合伤；脊柱检查排除脊柱骨折；颈静脉怒张提示张力性气胸或心包填塞。因此所有选项均为重点检查项目。",
        "judgment": [
          "未解释所有正确项",
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "235",
        "question": "26岁女性,闭经50天出现恶心呕吐,尿妊娠试验阳性,行人工流产术未吸出组织物,以下可能的情况",
        "options": {
          "A": "双子宫,一侧宫腔妊娠",
          "B": "异位妊娠",
          "C": "残角子宫妊娠",
          "D": "单纯闭经未妊娠",
          "E": "吸宫技术不熟练"
        },
        "gold_answer": "ABCDE",
        "teacher_analysis": "患者停经、尿妊娠试验阳性提示妊娠，但人工流产未吸出组织物，可能原因包括：双子宫一侧妊娠时吸管未进入妊娠侧宫腔；异位妊娠时宫腔内无妊娠物；残角子宫妊娠因与单角子宫不相通而无法吸出；单纯闭经未妊娠则尿妊娠试验假阳性；吸宫技术不熟练可导致操作失败。以上情况均可能导致未吸出组织物，故全选。",
        "judgment": [
          "未解释所有正确项",
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "17",
        "question": "患者男性,40岁,生活在北方农村,头痛、发 作性抽搐2个月,2个月内发生抽搐3次,表现为全 面性强直-阵挛发作,近1周头痛加重,为全头痛伴 恶心、呕吐。查体:36.7°C,神志清,反应较迟钝,颅神经未见异常,四肢肌力5级,肌张力略增高,右侧肢体肌反射活跃,左侧肢体肌反射亢进,双侧病理反射阳性,诊断为症状性癫痫,全面性强直-阵挛发作收入院。进一步的处理计划是",
        "options": {
          "A": "继续第二次吡喹酮治疗",
          "B": "甘露醇脱水治疗",
          "C": "继续抗癫痫治疗",
          "D": "密切观察颅压变化",
          "E": "脑室内病灶手术切除"
        },
        "gold_answer": "ABCDE",
        "teacher_analysis": "患者诊断为脑囊虫病继发癫痫，目前头痛加重伴恶心呕吐，提示颅高压。处理需综合：吡喹酮驱虫治疗需多疗程，故继续第二次治疗；甘露醇脱水降颅压；继续抗癫痫控制发作；密切观察颅压变化以防脑疝；若脑室内病灶引起梗阻性脑积水或药物无效，需手术切除。所有选项均为合理处理计划。",
        "judgment": [
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "507",
        "question": "患者,女,36岁。诊断为风湿热8年。门诊坚持每月肌注长效青霉素120万U。近日因受凉后感冒,出现发热、咽痛、流涕、心悸症状。来社区门诊看病。查体:T:38.2°C,咽红,扁桃体II大,心率109次/分,无杂音,双肺音清,未及啰音。门诊按照上呼吸道感染治疗2天后,患者心悸、胸闷加重,伴气短。查体:心率116次/分,律不齐,心前区可闻及收缩期杂音3/6级。进一步检查包括",
        "options": {
          "A": "ASO",
          "B": "血沉",
          "C": "心电图",
          "D": "血常规",
          "E": "C反应蛋白"
        },
        "gold_answer": "ABCDE",
        "teacher_analysis": "患者有风湿热病史，近期上感后出现心悸、胸闷、气短，查体心率快、律不齐、新出现收缩期杂音，提示风湿热复发或风湿性心脏炎可能。ASO可检测链球菌感染证据；血沉和C反应蛋白反映炎症活动；心电图评估心律失常和心脏受累；血常规辅助判断感染。以上检查均为风湿热诊断和活动性评估所需，故全选。",
        "judgment": [
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "910",
        "question": "病毒性心肌炎可能出现的超声心动图改变有",
        "options": {
          "A": "超声心动图未见异常",
          "B": "左心室整体收缩活动减弱",
          "C": "左心室室壁弥漫性增厚",
          "D": "左心室内径正常上限",
          "E": "左心室室壁活动异常"
        },
        "gold_answer": "ABCDE",
        "teacher_analysis": "病毒性心肌炎超声心动图表现多样，轻症可完全正常（A）。心肌受累时可出现左心室整体收缩活动减弱（B）、室壁活动异常（E）。心肌水肿可致室壁弥漫性增厚（C），早期心腔大小可处于正常上限（D）。因此所有选项均可能出现。",
        "judgment": [
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "2632",
        "question": "怀疑颅内感染的患儿,进行腰椎穿刺术的禁忌证是",
        "options": {
          "A": "颅内压异常增高",
          "B": "腰穿部位皮肤感染",
          "C": "严重心、肺功能受累和休克",
          "D": "惊厥持续状态",
          "E": "意识改变,瞳孔不等大"
        },
        "gold_answer": "ABCDE",
        "teacher_analysis": "腰椎穿刺术用于诊断颅内感染，但存在禁忌证。颅内压异常增高时穿刺可诱发脑疝；穿刺部位皮肤感染可导致病原体进入蛛网膜下腔；严重心肺功能受累或休克时难以耐受操作；惊厥持续状态时患儿无法配合且风险高；意识改变、瞳孔不等大提示可能已有脑疝或严重颅内高压，穿刺会加重病情。因此所有选项均为禁忌证。",
        "judgment": [
          "未解释所有正确项",
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "535",
        "question": "患儿,2岁男孩,发热12小时,体温39°C,在儿科门诊就诊过程中突然发生惊厥,表现为双眼凝视、四肢抖动、呼之不应,持续3~4分钟。査体:体温39°C,精神可,面色红润,咽充血,扁桃体I度肿大,心、肺、腹部检査无异常,脑膜刺激征阴性。此患儿处于惊厥持续状态,如何处理",
        "options": {
          "A": "吸氧",
          "B": "首选地西泮静脉推注止惊",
          "C": "尽快找出病因",
          "D": "积极降温",
          "E": "甘露醇降颅压",
          "F": "水合氯醛灌肠"
        },
        "gold_answer": "ABCDEF",
        "teacher_analysis": "该患儿为热性惊厥持续状态，处理需综合进行。吸氧可改善脑缺氧；地西泮静脉推注是止惊首选；尽快找病因以对因治疗；积极降温控制发热；甘露醇降颅压预防脑水肿；水合氯醛灌肠可作为止惊替代。所有选项均为必要措施，故全选。",
        "judgment": [
          "未解释所有正确项",
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "129",
        "question": "患者女性,65岁。因自觉食欲明显增加半年前来就诊。自觉“长胖很多”。经有关检查示:C-肽释放试验结果为空腹580pmol/L,1小时1120pmol/L,2小时1650pmol/L,3小时866pmol/L;糖化血红蛋白11%;24小时尿C-肽为18.4μg%,24小时尿蛋白118mg。体型较肥胖。需排除因其他原因引起的血糖升高、尿糖阳性或糖耐量降低的情况有",
        "options": {
          "A": "弥漫性胰腺病变",
          "B": "肝脏疾病",
          "C": "肢端肥大症",
          "D": "库欣综合征",
          "E": "甲状腺机能亢进症",
          "F": "生长抑素瘤"
        },
        "gold_answer": "ABCDEF",
        "teacher_analysis": "患者C肽水平升高、糖化血红蛋白11%提示高胰岛素血症和糖尿病，需鉴别继发性高血糖原因。弥漫性胰腺病变可破坏胰岛导致血糖异常；肝脏疾病影响糖代谢；肢端肥大症、库欣综合征、甲亢均可通过拮抗胰岛素或促进糖异生引起血糖升高；生长抑素瘤抑制胰岛素分泌但也可导致糖耐量异常。因此所有选项均需排除。",
        "judgment": [
          "未解释所有正确项",
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "2610",
        "question": "孕28~35周出现胎膜早破,可给予的处理包括",
        "options": {
          "A": "预防性使用抗生素",
          "B": "查血常规、C-反应蛋白除外感染",
          "C": "应用子宫收缩抑制药",
          "D": "促胎肺成熟",
          "E": "B超监测残余羊水量"
        },
        "gold_answer": "ABCDE",
        "teacher_analysis": "孕28~35周胎膜早破属未足月胎膜早破，处理原则包括：预防感染（使用抗生素）、监测感染指标（血常规、CRP）、抑制宫缩以延长孕周、促胎肺成熟（糖皮质激素）、超声监测羊水量评估胎儿安危。所有选项均为标准处理措施。",
        "judgment": [
          "未解释所有正确项",
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "148",
        "question": "患者女,46岁。白带增多1年,偶而呈粉红色,近3个月有性交出血,月经尚规则,妇科检查见宫颈中度糜烂,子宫正常大小,质软、双侧附件无异常所见。阴道镜下活检有癌细胞已突破基底膜5mm,有淋巴管被侵袭及病状融合现象。该患者进行放疗后,下列属于放射治疗的远期并发症的是",
        "options": {
          "A": "膀胱挛缩",
          "B": "放射性直肠炎",
          "C": "下肢淋巴水肿",
          "D": "输尿管狭窄",
          "E": "股骨颈骨折",
          "F": "骨髓造血功能障碍"
        },
        "gold_answer": "ABCDEF",
        "teacher_analysis": "该患者为宫颈癌放疗后，放射治疗的远期并发症包括：膀胱挛缩（A）因膀胱受照射后纤维化；放射性直肠炎（B）因直肠黏膜损伤；下肢淋巴水肿（C）因淋巴管阻塞；输尿管狭窄（D）因输尿管周围纤维化；股骨颈骨折（E）因骨盆照射后骨质疏松；骨髓造血功能障碍（F）因骨盆骨髓受照射。所有选项均为已知的放疗远期并发症，故全选。",
        "judgment": [
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "209",
        "question": "老年男性,65岁,既往有胆囊结石病史,因上腹痛4小时由家属送诊,查体:意识淡漠,血压90/50mmHg,右上腹部肌紧张,此时最恰当的治疗措施是",
        "options": {
          "A": "紧急完善术前准备,急诊手术",
          "B": "积极补液,抗休克治疗",
          "C": "同时行床旁B超检查",
          "D": "紧急化验血常规、肝功、胰酶",
          "E": "行腹部CT检查",
          "F": "行腹部平片检查"
        },
        "gold_answer": "ABCD",
        "teacher_analysis": "患者为老年男性，有胆囊结石病史，出现上腹痛、意识淡漠、低血压、右上腹肌紧张，提示急性胆道感染合并感染性休克。此时应积极抗休克（补液）同时紧急术前准备，并完善床旁B超、血常规、肝功、胰酶等检查以明确诊断和评估病情。CT和平片并非最优先，且可能延误抢救。",
        "judgment": [
          "未解释所有正确项",
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      },
      {
        "sample_id": "2655",
        "question": "可能诱发伤寒患者出现肠穿孔或肠出血的因素有",
        "options": {
          "A": "进食富含纤维素的食物",
          "B": "应用新斯的明缓解腹胀",
          "C": "进食无渣或少渣的半流质饮食",
          "D": "腹胀时应用肛管排气",
          "E": "便秘应用缓泻药"
        },
        "gold_answer": "ABE",
        "teacher_analysis": "伤寒患者肠道淋巴组织病变，肠壁变薄，在病程第2-3周易发生肠穿孔或肠出血。进食富含纤维素食物可机械刺激肠道；新斯的明增强肠蠕动；缓泻药促进肠蠕动和排便，均可能诱发穿孔或出血。无渣半流质饮食和肛管排气相对安全，不易诱发。",
        "judgment": [
          "未解释所有正确项",
          "未排除干扰项",
          "存在可能诱导泛选表达",
          "只解释正确项"
        ]
      }
    ],
    "rationale_quality_summary": {
      "cot_multi_total": 678,
      "explains_all_correct_rate": 0.753687,
      "mentions_wrong_options_rate": 0.862832,
      "fan_select_phrase_rate": 0.359882,
      "only_explains_correct_rate": 0.128319,
      "avg_rationale_length": 127.40413
    }
  }
}

## 8. Teacher rationale 质量分析

- CoT 多选样本数：`678`
- 解释所有正确项比例：`0.753687`
- 提及/排除错误选项比例：`0.862832`
- 可能诱导泛选表达比例：`0.359882`
- 只解释正确项比例：`0.128319`

代表性 CoT 多选样本见 JSON 的 `train_data_distribution.cot_mixed_train.cot_multichoice_rationale_examples` 字段。

## 9. CoT-worthiness / option-confusion 分组分析

### high option-confusion vs low option-confusion

{
  "Direct SFT": {
    "high": {
      "total": 13,
      "exact_match": 0.692308,
      "option_precision": 0.945946,
      "option_recall": 0.875,
      "option_f1": 0.909091,
      "extra_option_rate": 0.153846,
      "missing_option_rate": 0.307692,
      "avg_pred_answer_len": 2.846154,
      "avg_gold_answer_len": 3.076923,
      "pred_answer_len_distribution": {
        "1": 1,
        "2": 3,
        "3": 6,
        "4": 3
      },
      "gold_answer_len_distribution": {
        "2": 4,
        "3": 5,
        "4": 3,
        "5": 1
      },
      "pred_minus_gold_len_distribution": {
        "-1": 3,
        "0": 10
      }
    },
    "low": {
      "total": 191,
      "exact_match": 0.628272,
      "option_precision": 0.930979,
      "option_recall": 0.893683,
      "option_f1": 0.91195,
      "extra_option_rate": 0.17801,
      "missing_option_rate": 0.251309,
      "avg_pred_answer_len": 3.26178,
      "avg_gold_answer_len": 3.397906,
      "pred_answer_len_distribution": {
        "0": 1,
        "1": 6,
        "2": 41,
        "3": 63,
        "4": 54,
        "5": 26
      },
      "gold_answer_len_distribution": {
        "2": 36,
        "3": 66,
        "4": 66,
        "5": 23
      },
      "pred_minus_gold_len_distribution": {
        "+1": 18,
        "+2": 5,
        "+3_or_more": 1,
        "-1": 27,
        "-2_or_less": 13,
        "0": 127
      }
    }
  },
  "CoT/Mixed": {
    "high": {
      "total": 13,
      "exact_match": 0.538462,
      "option_precision": 0.916667,
      "option_recall": 0.825,
      "option_f1": 0.868421,
      "extra_option_rate": 0.230769,
      "missing_option_rate": 0.384615,
      "avg_pred_answer_len": 2.769231,
      "avg_gold_answer_len": 3.076923,
      "pred_answer_len_distribution": {
        "1": 1,
        "2": 4,
        "3": 6,
        "4": 1,
        "5": 1
      },
      "gold_answer_len_distribution": {
        "2": 4,
        "3": 5,
        "4": 3,
        "5": 1
      },
      "pred_minus_gold_len_distribution": {
        "+1": 1,
        "-1": 5,
        "0": 7
      }
    },
    "low": {
      "total": 191,
      "exact_match": 0.612565,
      "option_precision": 0.903614,
      "option_recall": 0.924499,
      "option_f1": 0.913937,
      "extra_option_rate": 0.256545,
      "missing_option_rate": 0.193717,
      "avg_pred_answer_len": 3.47644,
      "avg_gold_answer_len": 3.397906,
      "pred_answer_len_distribution": {
        "1": 3,
        "2": 33,
        "3": 65,
        "4": 50,
        "5": 40
      },
      "gold_answer_len_distribution": {
        "2": 36,
        "3": 66,
        "4": 66,
        "5": 23
      },
      "pred_minus_gold_len_distribution": {
        "+1": 25,
        "+2": 10,
        "+3_or_more": 2,
        "-1": 22,
        "-2_or_less": 6,
        "0": 126
      }
    }
  }
}

### case-rich vs non-case

{
  "Direct SFT": {
    "non_case": {
      "total": 204,
      "exact_match": 0.632353,
      "option_precision": 0.931818,
      "option_recall": 0.892598,
      "option_f1": 0.911786,
      "extra_option_rate": 0.176471,
      "missing_option_rate": 0.254902,
      "avg_pred_answer_len": 3.235294,
      "avg_gold_answer_len": 3.377451,
      "pred_answer_len_distribution": {
        "0": 1,
        "1": 7,
        "2": 44,
        "3": 69,
        "4": 57,
        "5": 26
      },
      "gold_answer_len_distribution": {
        "2": 40,
        "3": 71,
        "4": 69,
        "5": 24
      },
      "pred_minus_gold_len_distribution": {
        "+1": 18,
        "+2": 5,
        "+3_or_more": 1,
        "-1": 30,
        "-2_or_less": 13,
        "0": 137
      }
    }
  },
  "CoT/Mixed": {
    "non_case": {
      "total": 204,
      "exact_match": 0.607843,
      "option_precision": 0.904286,
      "option_recall": 0.918723,
      "option_f1": 0.911447,
      "extra_option_rate": 0.254902,
      "missing_option_rate": 0.205882,
      "avg_pred_answer_len": 3.431373,
      "avg_gold_answer_len": 3.377451,
      "pred_answer_len_distribution": {
        "1": 4,
        "2": 37,
        "3": 71,
        "4": 51,
        "5": 41
      },
      "gold_answer_len_distribution": {
        "2": 40,
        "3": 71,
        "4": 69,
        "5": 24
      },
      "pred_minus_gold_len_distribution": {
        "+1": 26,
        "+2": 10,
        "+3_or_more": 2,
        "-1": 27,
        "-2_or_less": 6,
        "0": 133
      }
    }
  }
}

### high CoT-worthiness vs low CoT-worthiness

{
  "Direct SFT": {
    "high": {
      "total": 13,
      "exact_match": 0.692308,
      "option_precision": 0.944444,
      "option_recall": 0.829268,
      "option_f1": 0.883117,
      "extra_option_rate": 0.153846,
      "missing_option_rate": 0.307692,
      "avg_pred_answer_len": 2.769231,
      "avg_gold_answer_len": 3.153846,
      "pred_answer_len_distribution": {
        "1": 1,
        "2": 3,
        "3": 7,
        "4": 2
      },
      "gold_answer_len_distribution": {
        "2": 3,
        "3": 6,
        "4": 3,
        "5": 1
      },
      "pred_minus_gold_len_distribution": {
        "-1": 2,
        "-2_or_less": 1,
        "0": 10
      }
    },
    "low": {
      "total": 191,
      "exact_match": 0.628272,
      "option_precision": 0.93109,
      "option_recall": 0.896605,
      "option_f1": 0.913522,
      "extra_option_rate": 0.17801,
      "missing_option_rate": 0.251309,
      "avg_pred_answer_len": 3.267016,
      "avg_gold_answer_len": 3.39267,
      "pred_answer_len_distribution": {
        "0": 1,
        "1": 6,
        "2": 41,
        "3": 62,
        "4": 55,
        "5": 26
      },
      "gold_answer_len_distribution": {
        "2": 37,
        "3": 65,
        "4": 66,
        "5": 23
      },
      "pred_minus_gold_len_distribution": {
        "+1": 18,
        "+2": 5,
        "+3_or_more": 1,
        "-1": 28,
        "-2_or_less": 12,
        "0": 127
      }
    }
  },
  "CoT/Mixed": {
    "high": {
      "total": 13,
      "exact_match": 0.461538,
      "option_precision": 0.888889,
      "option_recall": 0.780488,
      "option_f1": 0.831169,
      "extra_option_rate": 0.307692,
      "missing_option_rate": 0.384615,
      "avg_pred_answer_len": 2.769231,
      "avg_gold_answer_len": 3.153846,
      "pred_answer_len_distribution": {
        "1": 1,
        "2": 4,
        "3": 6,
        "4": 1,
        "5": 1
      },
      "gold_answer_len_distribution": {
        "2": 3,
        "3": 6,
        "4": 3,
        "5": 1
      },
      "pred_minus_gold_len_distribution": {
        "+1": 2,
        "-1": 4,
        "-2_or_less": 1,
        "0": 6
      }
    },
    "low": {
      "total": 191,
      "exact_match": 0.617801,
      "option_precision": 0.90512,
      "option_recall": 0.927469,
      "option_f1": 0.916158,
      "extra_option_rate": 0.251309,
      "missing_option_rate": 0.193717,
      "avg_pred_answer_len": 3.47644,
      "avg_gold_answer_len": 3.39267,
      "pred_answer_len_distribution": {
        "1": 3,
        "2": 33,
        "3": 65,
        "4": 50,
        "5": 40
      },
      "gold_answer_len_distribution": {
        "2": 37,
        "3": 65,
        "4": 66,
        "5": 23
      },
      "pred_minus_gold_len_distribution": {
        "+1": 24,
        "+2": 10,
        "+3_or_more": 2,
        "-1": 23,
        "-2_or_less": 5,
        "0": 127
      }
    }
  }
}

### recall-like vs reasoning-like

{
  "Direct SFT": {
    "reasoning_like": {
      "total": 67,
      "exact_match": 0.61194,
      "option_precision": 0.908213,
      "option_recall": 0.87037,
      "option_f1": 0.888889,
      "extra_option_rate": 0.223881,
      "missing_option_rate": 0.268657,
      "avg_pred_answer_len": 3.089552,
      "avg_gold_answer_len": 3.223881,
      "pred_answer_len_distribution": {
        "0": 1,
        "1": 2,
        "2": 18,
        "3": 23,
        "4": 15,
        "5": 8
      },
      "gold_answer_len_distribution": {
        "2": 16,
        "3": 25,
        "4": 21,
        "5": 5
      },
      "pred_minus_gold_len_distribution": {
        "+1": 6,
        "+2": 1,
        "+3_or_more": 1,
        "-1": 8,
        "-2_or_less": 5,
        "0": 46
      }
    },
    "recall_like": {
      "total": 137,
      "exact_match": 0.642336,
      "option_precision": 0.942605,
      "option_recall": 0.902748,
      "option_f1": 0.922246,
      "extra_option_rate": 0.153285,
      "missing_option_rate": 0.248175,
      "avg_pred_answer_len": 3.306569,
      "avg_gold_answer_len": 3.452555,
      "pred_answer_len_distribution": {
        "1": 5,
        "2": 26,
        "3": 46,
        "4": 42,
        "5": 18
      },
      "gold_answer_len_distribution": {
        "2": 24,
        "3": 46,
        "4": 48,
        "5": 19
      },
      "pred_minus_gold_len_distribution": {
        "+1": 12,
        "+2": 4,
        "-1": 22,
        "-2_or_less": 8,
        "0": 91
      }
    }
  },
  "CoT/Mixed": {
    "reasoning_like": {
      "total": 67,
      "exact_match": 0.522388,
      "option_precision": 0.880734,
      "option_recall": 0.888889,
      "option_f1": 0.884793,
      "extra_option_rate": 0.328358,
      "missing_option_rate": 0.283582,
      "avg_pred_answer_len": 3.253731,
      "avg_gold_answer_len": 3.223881,
      "pred_answer_len_distribution": {
        "1": 3,
        "2": 13,
        "3": 25,
        "4": 16,
        "5": 10
      },
      "gold_answer_len_distribution": {
        "2": 16,
        "3": 25,
        "4": 21,
        "5": 5
      },
      "pred_minus_gold_len_distribution": {
        "+1": 10,
        "+2": 2,
        "+3_or_more": 1,
        "-1": 12,
        "-2_or_less": 1,
        "0": 41
      }
    },
    "recall_like": {
      "total": 137,
      "exact_match": 0.649635,
      "option_precision": 0.914938,
      "option_recall": 0.932347,
      "option_f1": 0.92356,
      "extra_option_rate": 0.218978,
      "missing_option_rate": 0.167883,
      "avg_pred_answer_len": 3.518248,
      "avg_gold_answer_len": 3.452555,
      "pred_answer_len_distribution": {
        "1": 1,
        "2": 24,
        "3": 46,
        "4": 35,
        "5": 31
      },
      "gold_answer_len_distribution": {
        "2": 24,
        "3": 46,
        "4": 48,
        "5": 19
      },
      "pred_minus_gold_len_distribution": {
        "+1": 16,
        "+2": 8,
        "+3_or_more": 1,
        "-1": 15,
        "-2_or_less": 5,
        "0": 92
      }
    }
  }
}

### 否定题 vs 非否定题

{
  "Direct SFT": {
    "False": {
      "total": 203,
      "exact_match": 0.635468,
      "option_precision": 0.931715,
      "option_recall": 0.893741,
      "option_f1": 0.912333,
      "extra_option_rate": 0.17734,
      "missing_option_rate": 0.251232,
      "avg_pred_answer_len": 3.246305,
      "avg_gold_answer_len": 3.384236,
      "pred_answer_len_distribution": {
        "0": 1,
        "1": 6,
        "2": 44,
        "3": 69,
        "4": 57,
        "5": 26
      },
      "gold_answer_len_distribution": {
        "2": 39,
        "3": 71,
        "4": 69,
        "5": 24
      },
      "pred_minus_gold_len_distribution": {
        "+1": 18,
        "+2": 5,
        "+3_or_more": 1,
        "-1": 29,
        "-2_or_less": 13,
        "0": 137
      }
    },
    "True": {
      "total": 1,
      "exact_match": 0.0,
      "option_precision": 1.0,
      "option_recall": 0.5,
      "option_f1": 0.666667,
      "extra_option_rate": 0.0,
      "missing_option_rate": 1.0,
      "avg_pred_answer_len": 1,
      "avg_gold_answer_len": 2,
      "pred_answer_len_distribution": {
        "1": 1
      },
      "gold_answer_len_distribution": {
        "2": 1
      },
      "pred_minus_gold_len_distribution": {
        "-1": 1
      }
    }
  },
  "CoT/Mixed": {
    "False": {
      "total": 203,
      "exact_match": 0.610837,
      "option_precision": 0.906609,
      "option_recall": 0.918486,
      "option_f1": 0.912509,
      "extra_option_rate": 0.251232,
      "missing_option_rate": 0.206897,
      "avg_pred_answer_len": 3.428571,
      "avg_gold_answer_len": 3.384236,
      "pred_answer_len_distribution": {
        "1": 4,
        "2": 37,
        "3": 71,
        "4": 50,
        "5": 41
      },
      "gold_answer_len_distribution": {
        "2": 39,
        "3": 71,
        "4": 69,
        "5": 24
      },
      "pred_minus_gold_len_distribution": {
        "+1": 26,
        "+2": 9,
        "+3_or_more": 2,
        "-1": 27,
        "-2_or_less": 6,
        "0": 133
      }
    },
    "True": {
      "total": 1,
      "exact_match": 0.0,
      "option_precision": 0.5,
      "option_recall": 1.0,
      "option_f1": 0.666667,
      "extra_option_rate": 1.0,
      "missing_option_rate": 0.0,
      "avg_pred_answer_len": 4,
      "avg_gold_answer_len": 2,
      "pred_answer_len_distribution": {
        "4": 1
      },
      "gold_answer_len_distribution": {
        "2": 1
      },
      "pred_minus_gold_len_distribution": {
        "+2": 1
      }
    }
  }
}

## 10. 错误重叠案例

{
  "both_correct_count": 112,
  "both_wrong_count": 63,
  "direct_correct_cot_wrong_count": 17,
  "direct_wrong_cot_correct_count": 12,
  "direct_wrong_cot_wrong_but_cot_higher_recall_count": 10,
  "direct_correct_cot_wrong_due_to_extra_count": 11,
  "direct_wrong_cot_correct_due_to_less_missing_count": 11,
  "representative_error_cases": {
    "direct_correct_cot_wrong_due_to_extra": [
      {
        "index": 2,
        "sample_id": "cmexam_test_178",
        "question": "按照《医疗器械监督管理条例》，人体使用医疗器械旨在达到的预期目的有",
        "options": {
          "A": "对疾病的预防、诊断、治疗、监护、缓解",
          "B": "对损伤或者残疾的诊断、治疗、监护、缓解、补偿",
          "C": "对解剖或者生理过程的研究、替代、调节",
          "D": "妊娠控制",
          "E": "用药理学、免疫学或者代谢的手段作用于人体体表及体内"
        },
        "gold": "ABCD",
        "direct_raw_output": "ABCD",
        "direct_extracted_answer": "ABCD",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.0,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      },
      {
        "index": 25,
        "sample_id": "cmexam_test_1061",
        "question": "多食肥甘厚味容易",
        "options": {
          "A": "生痰",
          "B": "助湿",
          "C": "化热",
          "D": "发生痈疡",
          "E": "发生瘿瘤"
        },
        "gold": "ACD",
        "direct_raw_output": "ACD",
        "direct_extracted_answer": "ACD",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 2,
        "negative_question": false,
        "option_confusion_score": 0.0,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      },
      {
        "index": 29,
        "sample_id": "cmexam_test_1128",
        "question": "主产于青藏高原地区的药材有",
        "options": {
          "A": "冬虫夏草",
          "B": "雪莲花",
          "C": "红景天",
          "D": "西红花",
          "E": "炉贝母"
        },
        "gold": "ABCE",
        "direct_raw_output": "ABCE",
        "direct_extracted_answer": "ABCE",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.8,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      },
      {
        "index": 53,
        "sample_id": "cmexam_test_1896",
        "question": "牛膝与川牛膝均有的功效有",
        "options": {
          "A": "逐瘀",
          "B": "通经",
          "C": "利尿通淋",
          "D": "引血下行",
          "E": "补肝肾强筋骨"
        },
        "gold": "ABCD",
        "direct_raw_output": "ABCD",
        "direct_extracted_answer": "ABCD",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.0,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      },
      {
        "index": 72,
        "sample_id": "cmexam_test_2626",
        "question": "根据《药品注册管理办法》，关于药品再注册的说法，正确的有",
        "options": {
          "A": "未在规定时限内完成药品批准证明文件和药品监督管理部门要求的研究工作，且无合理理由的，不予再注册",
          "B": "药品注册证书有效期届满后，但未超过6个月提出再注册申请的，予以再注册",
          "C": "药品经上市后评价，属于疗效不确切、不良反应大或者因其他原因危害人体健康的，不予再注册",
          "D": "药品上市许可持有人应当在药品注册证书有效期届满前3个月申请再注册"
        },
        "gold": "AC",
        "direct_raw_output": "AC",
        "direct_extracted_answer": "AC",
        "cot_raw_output": "ACD",
        "cot_extracted_answer": "ACD",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.4,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      },
      {
        "index": 77,
        "sample_id": "cmexam_test_2758",
        "question": "能够增加蛋白多肽药物经皮给药吸收的物理化学方法有",
        "options": {
          "A": "气体导入法",
          "B": "超声波导入法",
          "C": "离子导入法",
          "D": "电穿孔导入法",
          "E": "紫外光导入法"
        },
        "gold": "BCD",
        "direct_raw_output": "BCD",
        "direct_extracted_answer": "BCD",
        "cot_raw_output": "ABCD",
        "cot_extracted_answer": "ABCD",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 1.2,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      },
      {
        "index": 80,
        "sample_id": "cmexam_test_2860",
        "question": "含汞的药物服用过量或长期服用会出现腹痛腹泻，口中有金属味，流涎，牙龈肿胀溃烂，少尿，蛋白尿，严重者可发生急性肾功能衰竭。含汞的中药有",
        "options": {
          "A": "朱砂",
          "B": "雄黄",
          "C": "轻粉",
          "D": "银朱",
          "E": "红粉"
        },
        "gold": "ACDE",
        "direct_raw_output": "ACDE",
        "direct_extracted_answer": "ACDE",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.8,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      },
      {
        "index": 114,
        "sample_id": "cmexam_test_3713",
        "question": "煅淬法炮制常用的淬液有",
        "options": {
          "A": "酒",
          "B": "醋",
          "C": "麻油",
          "D": "药汁",
          "E": "蜜水"
        },
        "gold": "ABD",
        "direct_raw_output": "ABD",
        "direct_extracted_answer": "ABD",
        "cot_raw_output": "ABCD",
        "cot_extracted_answer": "ABCD",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.0,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      }
    ],
    "direct_wrong_cot_correct_due_to_less_missing": [
      {
        "index": 23,
        "sample_id": "cmexam_test_987",
        "question": "能软化药材的方法",
        "options": {
          "A": "淋法",
          "B": "淘洗法",
          "C": "泡法",
          "D": "漂法",
          "E": "润法"
        },
        "gold": "ABCDE",
        "direct_raw_output": "CE",
        "direct_extracted_answer": "CE",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 3,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 0,
        "negative_question": false,
        "option_confusion_score": 0.4,
        "brief_analysis": "CoT/Mixed 正确，Direct 主要因为漏选失败。"
      },
      {
        "index": 28,
        "sample_id": "cmexam_test_1114",
        "question": "阴阳学说的基本内容包括",
        "options": {
          "A": "阴阳的对立制约",
          "B": "阴阳的互根互用",
          "C": "阴阳的消长平衡",
          "D": "阴阳的相互转化",
          "E": "阴阳的统一属性"
        },
        "gold": "ABCDE",
        "direct_raw_output": "ABCD",
        "direct_extracted_answer": "ABCD",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 1,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 0,
        "negative_question": false,
        "option_confusion_score": 1.2,
        "brief_analysis": "CoT/Mixed 正确，Direct 主要因为漏选失败。"
      },
      {
        "index": 94,
        "sample_id": "cmexam_test_3336",
        "question": "荔枝核的功效有",
        "options": {
          "A": "理气",
          "B": "祛寒",
          "C": "散滞",
          "D": "止痛",
          "E": "助阳"
        },
        "gold": "ABCD",
        "direct_raw_output": "ACD",
        "direct_extracted_answer": "ACD",
        "cot_raw_output": "ABCD",
        "cot_extracted_answer": "ABCD",
        "direct_missing": 1,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 0,
        "negative_question": false,
        "option_confusion_score": 0.8,
        "brief_analysis": "CoT/Mixed 正确，Direct 主要因为漏选失败。"
      },
      {
        "index": 99,
        "sample_id": "cmexam_test_3429",
        "question": "根据《中药品种保护条例》2013年有6家企业生产的“复方大青叶合剂”获批为国家中药保护品种，保护期限为7年，关于复方大青叶合剂的中药品种保护的说法正确的是",
        "options": {
          "A": "复方大青叶合剂为中药一级保护品种",
          "B": "中药保护品种在保护期满后可以申请延长保护期限",
          "C": "擅自仿制和生产复方大青叶合剂的，以生产假药论处",
          "D": "这6家企业必须是中国境内的生产企业"
        },
        "gold": "BCD",
        "direct_raw_output": "BC",
        "direct_extracted_answer": "BC",
        "cot_raw_output": "BCD",
        "cot_extracted_answer": "BCD",
        "direct_missing": 1,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 0,
        "negative_question": false,
        "option_confusion_score": 0.4,
        "brief_analysis": "CoT/Mixed 正确，Direct 主要因为漏选失败。"
      },
      {
        "index": 115,
        "sample_id": "cmexam_test_3715",
        "question": "对照品系指",
        "options": {
          "A": "自行制备、精制、标定后使用的标准物质",
          "B": "由卫生部指定的单位制备、标定和供应的标准物质",
          "C": "按效价单位（或㎍）计",
          "D": "均按干燥品（或无水物）进行计算后使用",
          "E": "均应附有使用说明书、质量要求、使用有效期和装量等"
        },
        "gold": "BDE",
        "direct_raw_output": "B",
        "direct_extracted_answer": "B",
        "cot_raw_output": "BDE",
        "cot_extracted_answer": "BDE",
        "direct_missing": 2,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 0,
        "negative_question": false,
        "option_confusion_score": 0.4,
        "brief_analysis": "CoT/Mixed 正确，Direct 主要因为漏选失败。"
      },
      {
        "index": 131,
        "sample_id": "cmexam_test_4279",
        "question": "目前己经实施药品电子监管的品种包括",
        "options": {
          "A": "麻醉药品",
          "B": "血液制品",
          "C": "中药注射剂",
          "D": "含地芬诺酯复方制剂",
          "E": "国家基本药物"
        },
        "gold": "ABCDE",
        "direct_raw_output": "ABC",
        "direct_extracted_answer": "ABC",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 2,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 0,
        "negative_question": false,
        "option_confusion_score": 0.0,
        "brief_analysis": "CoT/Mixed 正确，Direct 主要因为漏选失败。"
      },
      {
        "index": 132,
        "sample_id": "cmexam_test_4297",
        "question": "车前子的功效有",
        "options": {
          "A": "健脾",
          "B": "渗湿止泻",
          "C": "利水通淋",
          "D": "清肺化痰",
          "E": "清肝明目"
        },
        "gold": "BCDE",
        "direct_raw_output": "CDE",
        "direct_extracted_answer": "CDE",
        "cot_raw_output": "BCDE",
        "cot_extracted_answer": "BCDE",
        "direct_missing": 1,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 0,
        "negative_question": false,
        "option_confusion_score": 0.4,
        "brief_analysis": "CoT/Mixed 正确，Direct 主要因为漏选失败。"
      },
      {
        "index": 167,
        "sample_id": "cmexam_test_5361",
        "question": "合理用药基本原则中“经济”的含义，是指在用药安全、有效的前提下，必须做到",
        "options": {
          "A": "用药不滥",
          "B": "经济实用",
          "C": "利于环保",
          "D": "降低发病率",
          "E": "用法简便"
        },
        "gold": "ABCE",
        "direct_raw_output": "ABC",
        "direct_extracted_answer": "ABC",
        "cot_raw_output": "ABCE",
        "cot_extracted_answer": "ABCE",
        "direct_missing": 1,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 0,
        "negative_question": false,
        "option_confusion_score": 0.8,
        "brief_analysis": "CoT/Mixed 正确，Direct 主要因为漏选失败。"
      }
    ],
    "both_wrong_cot_higher_recall": [
      {
        "index": 3,
        "sample_id": "cmexam_test_200",
        "question": "结构中含有硫原子连接的H₂受体拮抗剂的抗溃疡药物有",
        "options": {
          "A": "西咪替丁",
          "B": "法莫替丁",
          "C": "罗沙替丁乙酸酯盐酸盐",
          "D": "尼扎替丁",
          "E": "盐酸雷尼替丁"
        },
        "gold": "ABDE",
        "direct_raw_output": "BE",
        "direct_extracted_answer": "BE",
        "cot_raw_output": "BCDE",
        "cot_extracted_answer": "BCDE",
        "direct_missing": 2,
        "direct_extra": 0,
        "cot_missing": 1,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.0,
        "brief_analysis": "两者都错，但 CoT/Mixed 覆盖了更多 gold 正确项。"
      },
      {
        "index": 39,
        "sample_id": "cmexam_test_1483",
        "question": "促胃肠动力药在应用中可致的典型不良反应有",
        "options": {
          "A": "高泌乳素血症",
          "B": "类磺胺药过敏反应",
          "C": "锥体外系反应",
          "D": "5-羟色胺综合征",
          "E": "灰婴综合征"
        },
        "gold": "AC",
        "direct_raw_output": "AD",
        "direct_extracted_answer": "AD",
        "cot_raw_output": "ACD",
        "cot_extracted_answer": "ACD",
        "direct_missing": 1,
        "direct_extra": 1,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.4,
        "brief_analysis": "两者都错，但 CoT/Mixed 覆盖了更多 gold 正确项。"
      },
      {
        "index": 56,
        "sample_id": "cmexam_test_2051",
        "question": "用于除去药液中的热原的方法不包括",
        "options": {
          "A": "高温法",
          "B": "酸碱法",
          "C": "吸附法",
          "D": "超滤法",
          "E": "反渗透法"
        },
        "gold": "AB",
        "direct_raw_output": "A",
        "direct_extracted_answer": "A",
        "cot_raw_output": "ABDE",
        "cot_extracted_answer": "ABDE",
        "direct_missing": 1,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 2,
        "negative_question": true,
        "option_confusion_score": 0.8,
        "brief_analysis": "两者都错，但 CoT/Mixed 覆盖了更多 gold 正确项。"
      },
      {
        "index": 65,
        "sample_id": "cmexam_test_2299",
        "question": "蚤休的功效有",
        "options": {
          "A": "清热解毒",
          "B": "利水消肿",
          "C": "消肿止痛",
          "D": "燥湿止痒",
          "E": "息风定惊"
        },
        "gold": "ACE",
        "direct_raw_output": "AE",
        "direct_extracted_answer": "AE",
        "cot_raw_output": "ACDE",
        "cot_extracted_answer": "ACDE",
        "direct_missing": 1,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.8,
        "brief_analysis": "两者都错，但 CoT/Mixed 覆盖了更多 gold 正确项。"
      },
      {
        "index": 103,
        "sample_id": "cmexam_test_3501",
        "question": "与磺胺类药物合用，会使其在体内溶解性降低而引起结晶尿，导致肾毒性增加的中药有",
        "options": {
          "A": "地榆",
          "B": "乌梅",
          "C": "陈皮",
          "D": "木瓜",
          "E": "女贞子"
        },
        "gold": "BCDE",
        "direct_raw_output": "AB",
        "direct_extracted_answer": "AB",
        "cot_raw_output": "ABDE",
        "cot_extracted_answer": "ABDE",
        "direct_missing": 3,
        "direct_extra": 1,
        "cot_missing": 1,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.4,
        "brief_analysis": "两者都错，但 CoT/Mixed 覆盖了更多 gold 正确项。"
      },
      {
        "index": 107,
        "sample_id": "cmexam_test_3627",
        "question": "能养血安神的中成药有",
        "options": {
          "A": "二至丸",
          "B": "百令胶囊",
          "C": "归脾浓缩丸",
          "D": "健脾生血颗粒",
          "E": "参竹精颗粒"
        },
        "gold": "CD",
        "direct_raw_output": "C",
        "direct_extracted_answer": "C",
        "cot_raw_output": "CDE",
        "cot_extracted_answer": "CDE",
        "direct_missing": 1,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.4,
        "brief_analysis": "两者都错，但 CoT/Mixed 覆盖了更多 gold 正确项。"
      },
      {
        "index": 143,
        "sample_id": "cmexam_test_4649",
        "question": "在治疗剂量下，地高辛的作用有",
        "options": {
          "A": "加强心肌收缩力",
          "B": "减慢心率",
          "C": "降低自律性",
          "D": "减慢传导",
          "E": "缩短心房肌和心室肌的有效不应期"
        },
        "gold": "ABCD",
        "direct_raw_output": "ABDE",
        "direct_extracted_answer": "ABDE",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 1,
        "direct_extra": 1,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.0,
        "brief_analysis": "两者都错，但 CoT/Mixed 覆盖了更多 gold 正确项。"
      },
      {
        "index": 165,
        "sample_id": "cmexam_test_5331",
        "question": "在体内经代谢后，其代谢产物具有活性的药物是",
        "options": {
          "B": "保泰松",
          "C": "卡马西平",
          "D": "苯妥英钠",
          "E": "阿苯达唑"
        },
        "gold": "BCE",
        "direct_raw_output": "ACE",
        "direct_extracted_answer": "",
        "cot_raw_output": "BCDE",
        "cot_extracted_answer": "BCDE",
        "direct_missing": 3,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.8,
        "brief_analysis": "两者都错，但 CoT/Mixed 覆盖了更多 gold 正确项。"
      }
    ],
    "cot_extra_in_negative_or_high_confusion": [
      {
        "index": 56,
        "sample_id": "cmexam_test_2051",
        "question": "用于除去药液中的热原的方法不包括",
        "options": {
          "A": "高温法",
          "B": "酸碱法",
          "C": "吸附法",
          "D": "超滤法",
          "E": "反渗透法"
        },
        "gold": "AB",
        "direct_raw_output": "A",
        "direct_extracted_answer": "A",
        "cot_raw_output": "ABDE",
        "cot_extracted_answer": "ABDE",
        "direct_missing": 1,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 2,
        "negative_question": true,
        "option_confusion_score": 0.8,
        "brief_analysis": "两者都错，但 CoT/Mixed 覆盖了更多 gold 正确项。"
      },
      {
        "index": 68,
        "sample_id": "cmexam_test_2394",
        "question": "根据《药品经营许可证管理办法》，省级药品监督管理部门负责",
        "options": {
          "A": "药品批发企业经营范围的变更",
          "B": "拟开办药品批发企业的企业名称审核",
          "C": "药品批发企业《药品经营许可证》的发证",
          "D": "药品批发企业《药品经营许可证》的换证",
          "E": "药品批发企业《药品经营许可证》的日常监督管理"
        },
        "gold": "ACDE",
        "direct_raw_output": "BCDE",
        "direct_extracted_answer": "BCDE",
        "cot_raw_output": "BCD",
        "cot_extracted_answer": "BCD",
        "direct_missing": 1,
        "direct_extra": 1,
        "cot_missing": 2,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 1.2,
        "brief_analysis": "CoT/Mixed 预测答案长度大于 gold，出现泛选。"
      },
      {
        "index": 77,
        "sample_id": "cmexam_test_2758",
        "question": "能够增加蛋白多肽药物经皮给药吸收的物理化学方法有",
        "options": {
          "A": "气体导入法",
          "B": "超声波导入法",
          "C": "离子导入法",
          "D": "电穿孔导入法",
          "E": "紫外光导入法"
        },
        "gold": "BCD",
        "direct_raw_output": "BCD",
        "direct_extracted_answer": "BCD",
        "cot_raw_output": "ABCD",
        "cot_extracted_answer": "ABCD",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 1.2,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      },
      {
        "index": 155,
        "sample_id": "cmexam_test_5051",
        "question": "医药品的B/A值（B=销售金额排序，A=药品DDDS排序）对5种药品进行药频度分析，得出下列数值中，可反映出药价偏高，患者接受程度低的有",
        "options": {
          "A": "B/A=1.0",
          "B": "B/A=0.6",
          "C": "B/A=3.0",
          "D": "B/A=1.5",
          "E": "B/A=0.8"
        },
        "gold": "BE",
        "direct_raw_output": "C",
        "direct_extracted_answer": "C",
        "cot_raw_output": "C",
        "cot_extracted_answer": "C",
        "direct_missing": 2,
        "direct_extra": 1,
        "cot_missing": 2,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 1.6,
        "brief_analysis": "CoT/Mixed 预测答案长度大于 gold，出现泛选。"
      }
    ],
    "cot_pred_len_greater_than_gold": [
      {
        "index": 2,
        "sample_id": "cmexam_test_178",
        "question": "按照《医疗器械监督管理条例》，人体使用医疗器械旨在达到的预期目的有",
        "options": {
          "A": "对疾病的预防、诊断、治疗、监护、缓解",
          "B": "对损伤或者残疾的诊断、治疗、监护、缓解、补偿",
          "C": "对解剖或者生理过程的研究、替代、调节",
          "D": "妊娠控制",
          "E": "用药理学、免疫学或者代谢的手段作用于人体体表及体内"
        },
        "gold": "ABCD",
        "direct_raw_output": "ABCD",
        "direct_extracted_answer": "ABCD",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.0,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      },
      {
        "index": 5,
        "sample_id": "cmexam_test_274",
        "question": "菊花治",
        "options": {
          "A": "肺热燥咳",
          "B": "肝热目赤",
          "C": "肝阳上亢",
          "D": "疔疮肿毒",
          "E": "风热表证"
        },
        "gold": "BDE",
        "direct_raw_output": "BCDE",
        "direct_extracted_answer": "BCDE",
        "cot_raw_output": "BCDE",
        "cot_extracted_answer": "BCDE",
        "direct_missing": 0,
        "direct_extra": 1,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.8,
        "brief_analysis": "CoT/Mixed 预测答案长度大于 gold，出现泛选。"
      },
      {
        "index": 18,
        "sample_id": "cmexam_test_819",
        "question": "处方书写基本要求",
        "options": {
          "A": "每张处方仅限于一名患者用药",
          "B": "药品名称书写应当使用药品的商品名",
          "C": "中药饮片应当单独开具处方",
          "D": "每张西药处方不得超过5种药品",
          "E": "非特殊情况无须注明临床诊断"
        },
        "gold": "ACD",
        "direct_raw_output": "ACDE",
        "direct_extracted_answer": "ACDE",
        "cot_raw_output": "ACDE",
        "cot_extracted_answer": "ACDE",
        "direct_missing": 0,
        "direct_extra": 1,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.8,
        "brief_analysis": "CoT/Mixed 预测答案长度大于 gold，出现泛选。"
      },
      {
        "index": 25,
        "sample_id": "cmexam_test_1061",
        "question": "多食肥甘厚味容易",
        "options": {
          "A": "生痰",
          "B": "助湿",
          "C": "化热",
          "D": "发生痈疡",
          "E": "发生瘿瘤"
        },
        "gold": "ACD",
        "direct_raw_output": "ACD",
        "direct_extracted_answer": "ACD",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 2,
        "negative_question": false,
        "option_confusion_score": 0.0,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      },
      {
        "index": 29,
        "sample_id": "cmexam_test_1128",
        "question": "主产于青藏高原地区的药材有",
        "options": {
          "A": "冬虫夏草",
          "B": "雪莲花",
          "C": "红景天",
          "D": "西红花",
          "E": "炉贝母"
        },
        "gold": "ABCE",
        "direct_raw_output": "ABCE",
        "direct_extracted_answer": "ABCE",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 0,
        "direct_extra": 0,
        "cot_missing": 0,
        "cot_extra": 1,
        "negative_question": false,
        "option_confusion_score": 0.8,
        "brief_analysis": "Direct 正确，CoT/Mixed 因额外选项导致 exact match 失败。"
      },
      {
        "index": 31,
        "sample_id": "cmexam_test_1145",
        "question": "妊娠禁用的中成药有",
        "options": {
          "A": "木香槟榔丸",
          "B": "牛黄清心丸",
          "C": "妇科分清丸",
          "D": "牛黄解毒丸",
          "E": "麝香保心丸"
        },
        "gold": "ADE",
        "direct_raw_output": "ABDE",
        "direct_extracted_answer": "ABDE",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 0,
        "direct_extra": 1,
        "cot_missing": 0,
        "cot_extra": 2,
        "negative_question": false,
        "option_confusion_score": 0.8,
        "brief_analysis": "CoT/Mixed 预测答案长度大于 gold，出现泛选。"
      },
      {
        "index": 32,
        "sample_id": "cmexam_test_1182",
        "question": "冠心苏合丸注意事项事项有",
        "options": {
          "A": "孕妇禁用",
          "B": "阴虚火旺者禁用",
          "C": "胃溃疡患者禁用",
          "D": "出血性疾病慎用",
          "E": "过敏体质者慎用"
        },
        "gold": "ACE",
        "direct_raw_output": "ABCDE",
        "direct_extracted_answer": "ABCDE",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 0,
        "direct_extra": 2,
        "cot_missing": 0,
        "cot_extra": 2,
        "negative_question": false,
        "option_confusion_score": 0.4,
        "brief_analysis": "CoT/Mixed 预测答案长度大于 gold，出现泛选。"
      },
      {
        "index": 38,
        "sample_id": "cmexam_test_1393",
        "question": "以种子药用的药材有",
        "options": {
          "A": "女贞子",
          "B": "枸杞子",
          "C": "蛇床子",
          "D": "菟丝子",
          "E": "牵牛子"
        },
        "gold": "DE",
        "direct_raw_output": "BCDE",
        "direct_extracted_answer": "BCDE",
        "cot_raw_output": "ABCDE",
        "cot_extracted_answer": "ABCDE",
        "direct_missing": 0,
        "direct_extra": 2,
        "cot_missing": 0,
        "cot_extra": 3,
        "negative_question": false,
        "option_confusion_score": 0.8,
        "brief_analysis": "CoT/Mixed 预测答案长度大于 gold，出现泛选。"
      }
    ]
  }
}

## 11. 评测设置检查

- Direct 与 CoT/Mixed 使用相同 CMExam multi task yaml 和相同 prompt。
- multi task 为 `generate_until`，`max_gen_toks: 16`，`do_sample: false`，属于 greedy 短答案生成。
- prompt 要求“直接给出所有正确选项字母”，但没有明确写“所有且仅有正确选项；多选、漏选、错选都判错”。
- 建议后续测试更严格 prompt：多选题必须选择所有且仅有正确选项；多选、漏选、错选都判错。请不要选择仅与题干相关但不属于标准答案的选项。

## 12. 结论

CoT/Mixed 没有超过 Direct，并不说明 CoT 完全无效。当前更像是 CoT/Mixed 改变了多选题的 precision-recall tradeoff：降低漏选、提高或维持 recall，但增加额外错误选项，导致 exact match 被拉低。

## 13. 下一步建议

- 报告 option-level precision / recall / F1，而不只看 exact match。
- 对 high CoT-worthiness 子集单独评估。
- 调整 multi eval prompt，强调“所有且仅有正确选项”。
- 优化 teacher rationale，要求必须解释为什么错误项不选。
- 降低 CoT 混合比例，例如 10% / 15% / 20% 对比。
- 做 selective CoT：只对 case-rich / high-confusion / multi-step 题使用 CoT。
- 在 GRPO reward 中加入 extra option penalty。
- 对多选题设计 set-level reward：正确项 recall + 错误项 precision 双目标。
- 对 CoT 多选样本增加“干扰项排除”监督。
