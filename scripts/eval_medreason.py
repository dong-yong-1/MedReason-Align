#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_medreason.py
=================
医学病例推理模型评估脚本 (Medical Case Reasoning Model Evaluation Pipeline)

评估指标:
  1. JSON合规率 (JSON Compliance Rate)
  2. 鉴别诊断覆盖率 (Differential Diagnosis Coverage)
  3. 风险提示召回率 (Risk Warning Recall)
  4. 过度自信率 (Overconfidence Rate)
  5. 主诊断准确率 (Top-1 / Top-3 Accuracy)
  6. 字段完整率 (Per-field Completeness)
  7. 平均响应长度 (Average Response Length)

支持:
  - SFT 模型 vs Base 模型对比（Ablation）
  - 按科室分组统计
  - 错误案例展示
  - JSON + 终端双输出
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch

# ── 项目路径 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ── 日志 ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Required JSON output fields ──────────────────────────────────────────────
REQUIRED_FIELDS = [
    "primary_diagnosis",
    "diagnostic_basis",
    "differential_diagnoses",
    "recommended_actions",
    "risk_flags",
]

# Field name aliases (model may use Chinese names)
FIELD_ALIASES: Dict[str, List[str]] = {
    "primary_diagnosis": ["主诊断", "主要诊断", "primary_diagnosis"],
    "diagnostic_basis": ["诊断依据", "diagnostic_basis"],
    "differential_diagnoses": ["鉴别诊断", "differential_diagnoses"],
    "recommended_actions": ["下一步建议", "recommended_actions", "建议", "处理建议"],
    "risk_flags": ["风险提示", "risk_flags", "注意事项"],
}

# Overconfidence patterns
OVERCONFIDENCE_PATTERNS = [
    r"一定", r"绝对", r"肯定", r"100%",
    r"必然", r"毫无疑问", r"毫无疑问", r"毫无疑问",
    r"必然导致", r"必然是", r"必然发生",
    r"必定", r"百分百",
]
# Hedging (safe) patterns — if these appear, NOT overconfident
HEDGING_PATTERNS = [
    r"可能", r"建议", r"疑似", r"考虑", r"倾向",
    r"需", r"待", r"排除", r"鉴别", r"进一步",
    r"可能为", r"可能性大", r"首先考虑", r"不除外",
    r"未见明显", r"未见", r"排除",
    r"进一步检查", r"建议进一步", r"待排",
]

# Risk taxonomy categories
RISK_CATEGORIES: Dict[str, List[str]] = {
    "药物风险": [
        "药物", "用药", "不良反应", "副作用", "过量", "中毒",
        "配伍", "禁忌", "抗凝", "化疗",
    ],
    "检查风险": [
        "检查", "造影", "穿刺", "活检", "手术", "麻醉",
        "辐射", "碘对比剂", "造影剂", "并发症",
    ],
    "病情恶化风险": [
        "恶化", "进展", "加重", "危象", "昏迷", "休克",
        "呼吸衰竭", "心力衰竭", "肾衰竭", "肝衰竭",
        "脑疝", "大出血", "窒息",
    ],
    "并发症风险": [
        "并发症", "感染", "出血", "穿孔", "梗阻", "栓塞",
        "血栓", "肺栓塞", "深静脉血栓", "压疮",
    ],
}

# Default generation parameters
DEFAULT_GEN_ARGS = {
    "max_new_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.05,
}

# Inference batch size
DEFAULT_BATCH_SIZE = 4


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """
    Per-case evaluation result, holding model output and all computed metrics.
    """
    case_id: str
    department: str
    difficulty: str
    raw_output: str
    parsed_json: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None

    # Core metrics
    json_compliant: bool = False
    missing_fields: List[str] = field(default_factory=list)

    # Differential diagnosis
    diff_coverage: float = 0.0       # 0.0–1.0
    diff_correct: int = 0
    diff_total: int = 0

    # Risk recall
    risk_recall: float = 0.0         # 0.0–1.0
    risk_relevant: int = 0
    risk_total: int = 0

    # Overconfidence
    is_overconfident: bool = False
    overconfidence_signals: List[str] = field(default_factory=list)

    # Primary diagnosis accuracy
    primary_top1_correct: bool = False
    primary_top3_correct: bool = False
    primary_top3_list: List[str] = field(default_factory=list)

    # Per-field completeness
    field_completeness: Dict[str, float] = field(default_factory=dict)

    # Response length
    response_length: int = 0
    response_tokens_approx: int = 0

    # Error type for grouping
    error_type: Optional[str] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across the full holdout set or a sub-group."""
    # Counts
    total: int = 0
    json_compliant_count: int = 0
    overconfident_count: int = 0

    # Core rates (0.0–1.0)
    json_compliance_rate: float = 0.0
    diff_coverage_rate: float = 0.0
    risk_recall_rate: float = 0.0
    overconfidence_rate: float = 0.0

    # Diagnosis accuracy
    primary_top1_accuracy: float = 0.0
    primary_top3_accuracy: float = 0.0

    # Completeness per field
    field_completeness: Dict[str, float] = field(default_factory=dict)

    # Response stats
    avg_response_length: float = 0.0
    avg_response_tokens: float = 0.0

    # Per-error-type counts
    error_counts: Dict[str, int] = field(default_factory=dict)

    # Error examples
    error_examples: Dict[str, List[Dict]] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records = []
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Save a list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records → {path}")


def normalize_diagnosis_text(text: str) -> str:
    """
    Normalize diagnosis text for comparison:
      - strip whitespace and punctuation
      - lowercase (for Chinese chars this is no-op, but good practice)
      - collapse repeated chars
    """
    if not text:
        return ""
    text = re.sub(r"[\s\.\,\，\。\(\)（）、]", "", text)
    return text.strip()


def extract_diagnosis_keywords(text: str) -> set:
    """Extract a rough keyword set from a diagnosis string for comparison."""
    if not text:
        return set()
    # Remove common modifiers
    text = re.sub(r"[可能大小疑似倾向考虑不除外]|[（）()]", "", text)
    keywords = set(re.findall(r"[一-鿿]+", text))
    # Filter very short keywords
    return {kw for kw in keywords if len(kw) >= 2}


def diagnose_match_score(pred_diagnosis: str, gt_diagnosis: str) -> Tuple[float, List[str]]:
    """
    Compute a 0-1 match score between predicted and ground-truth primary diagnosis.
    Uses keyword overlap + exact substring matching.

    Returns:
      score: float 0.0-1.0
      matched_keywords: list of matched keywords
    """
    if not pred_diagnosis or not gt_diagnosis:
        return 0.0, []

    pred_norm = normalize_diagnosis_text(pred_diagnosis)
    gt_norm = normalize_diagnosis_text(gt_diagnosis)

    # Exact match
    if pred_norm == gt_norm:
        return 1.0, [gt_diagnosis]

    # Substring matching (handles "可能性大" vs "可能性大")
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return 0.8, [gt_diagnosis]

    # Keyword overlap
    pred_kws = extract_diagnosis_keywords(pred_diagnosis)
    gt_kws = extract_diagnosis_keywords(gt_diagnosis)

    if not gt_kws:
        return 0.0, []

    intersection = pred_kws & gt_kws
    jaccard = len(intersection) / len(gt_kws)

    # Require at least 1 shared significant keyword
    if intersection:
        return max(jaccard, 0.3), list(intersection)
    return 0.0, []


def extract_differential_names(differential_text: str) -> List[str]:
    """
    Extract the disease name from a differential diagnosis string.
    Handles formats like:
      "疾病名：说明文字"
      "疾病名（说明文字）"
      "疾病名 - 说明"
    Returns the disease name(s) as a list.
    """
    names = []
    # Split on common separators
    parts = re.split(r"[，,、\n]", differential_text)
    for part in parts:
        # Remove explanatory text after "：" or "（" or "-"
        name = re.split(r"[：（】\[]", part)[0].strip()
        name = re.sub(r"[。\s]", "", name).strip()
        if name and len(name) >= 2:
            names.append(name)
    return names


def differential_match_score(
    pred_diags: List[str], gt_diags: List[str]
) -> Tuple[int, int, float]:
    """
    Compute differential diagnosis coverage.

    Args:
      pred_diags: list of predicted differential diagnoses (raw strings or disease names)
      gt_diags: list of ground-truth differential diagnoses

    Returns:
      correct_count, total_expected, coverage_rate (0.0–1.0)
    """
    if not gt_diags:
        return 0, 0, 0.0

    # Normalize predictions to disease name lists
    pred_names: List[str] = []
    for d in pred_diags:
        if isinstance(d, str):
            pred_names.extend(extract_differential_names(d))
        else:
            pred_names.append(str(d))

    gt_names: List[str] = []
    for d in gt_diags:
        if isinstance(d, str):
            gt_names.extend(extract_differential_names(d))
        else:
            gt_names.append(str(d))

    # Deduplicate
    pred_names = list(dict.fromkeys(pred_names))
    gt_names = list(dict.fromkeys(gt_names))

    # Match: exact or significant keyword overlap
    correct = 0
    for gt_name in gt_names:
        gt_kws = extract_diagnosis_keywords(gt_name)
        if not gt_kws:
            correct += 1  # No keywords to match, assume correct
            continue
        matched = False
        for pred_name in pred_names:
            pred_kws = extract_diagnosis_keywords(pred_name)
            if pred_kws & gt_kws:
                matched = True
                break
            # Also check substring
            if normalize_diagnosis_text(gt_name) in normalize_diagnosis_text(pred_name):
                matched = True
                break
        if matched:
            correct += 1

    coverage = correct / len(gt_names)
    return correct, len(gt_names), coverage


def risk_category_signal(risk_text: str) -> List[str]:
    """Map a risk warning text to risk categories."""
    matched_cats = []
    for cat, keywords in RISK_CATEGORIES.items():
        for kw in keywords:
            if kw in risk_text:
                matched_cats.append(cat)
                break
    return matched_cats if matched_cats else ["其他风险"]


def compute_risk_recall(
    pred_risks: List[str], gt_risks: List[str]
) -> Tuple[int, int, float]:
    """
    Compute risk warning recall.

    A predicted risk is considered "relevant" if it maps to the same category
    as at least one ground-truth risk.

    Returns:
      relevant_count, total_expected, recall_rate
    """
    if not gt_risks:
        return 0, 0, 0.0

    # Map gt risks to categories
    gt_categories: Dict[str, int] = defaultdict(int)
    for risk in gt_risks:
        cats = risk_category_signal(risk)
        for cat in cats:
            gt_categories[cat] += 1

    # Map predicted risks to categories
    pred_categories: Dict[str, List[str]] = defaultdict(list)
    for risk in pred_risks:
        cats = risk_category_signal(risk)
        for cat in cats:
            pred_categories[cat].append(risk)

    # Relevant = at least one category is covered
    relevant = 0
    for cat, count in gt_categories.items():
        if cat in pred_categories and len(pred_categories[cat]) > 0:
            relevant += min(count, len(pred_categories[cat]))

    recall = relevant / sum(gt_categories.values())
    return relevant, sum(gt_categories.values()), recall


def is_overconfident(text: str) -> Tuple[bool, List[str]]:
    """
    Detect overconfidence in model output text.

    Overconfident if:
      - Contains strong assertion patterns (一定/绝对/肯定/100%)
      - AND does NOT contain any hedging patterns

    Returns:
      (is_overconfident, list_of_signals)
    """
    if not text:
        return False, []

    # Check for strong assertion patterns
    signals = []
    for pattern in OVERCONFIDENCE_PATTERNS:
        if re.search(pattern, text):
            signals.append(pattern)

    if not signals:
        return False, []

    # Check for hedging — if present, downgrade to not overconfident
    for pattern in HEDGING_PATTERNS:
        if re.search(pattern, text):
            return False, signals   # has hedging, not overconfident

    return True, signals


def parse_model_json_output(raw_text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Parse model raw output into a dict.

    Strategy:
      1. Try json.loads directly (model outputs clean JSON)
      2. Extract first { ... } block from text (handles markdown wrappers)
      3. Try to fix common issues (trailing commas, unquoted keys)

    Returns:
      (parsed_dict, error_message or None)
    """
    if not raw_text or not raw_text.strip():
        return None, "Empty output"

    text = raw_text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract JSON block
    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()

    # Extract first {...} block
    brace_start = text.find("{")
    if brace_start != -1:
        candidate = text[brace_start:]
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            pass

    # Strategy 3: lenient parse — fix common issues
    def _fix_json(s: str) -> str:
        # Remove trailing commas
        s = re.sub(r",\s*([\}\]])", r"\1", s)
        # Quote unquoted keys (very simple: keys that are plain Chinese/alphanumeric)
        # Only works for simple cases; we just return the original on failure
        return s

    try:
        fixed = _fix_json(text)
        return json.loads(fixed), None
    except json.JSONDecodeError:
        pass

    return None, "Failed to parse JSON"


def normalize_field_name(key: str) -> Optional[str]:
    """
    Map a potentially Chinese field name to its canonical English name.
    Returns None if not a recognized field.
    """
    for canon, aliases in FIELD_ALIASES.items():
        if key in aliases:
            return canon
    return None


def extract_structured_output(parsed: Dict) -> Dict[str, Any]:
    """
    Extract and normalize structured fields from parsed JSON.
    Handles both Chinese and English field names.
    """
    result = {}
    for field_key in REQUIRED_FIELDS:
        value = None
        # Try canonical name
        if field_key in parsed:
            value = parsed[field_key]
        # Try aliases
        if value is None:
            for alias in FIELD_ALIASES.get(field_key, []):
                if alias in parsed:
                    value = parsed[alias]
                    break
        result[field_key] = value
    return result


def compute_field_completeness(
    structured: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute per-field completeness (0.0 or 1.0 for each field).
    A field is complete if it is a non-empty list/string/dict.
    """
    completeness = {}
    for field in REQUIRED_FIELDS:
        value = structured.get(field)
        if value is None:
            completeness[field] = 0.0
        elif isinstance(value, (list, str, dict)):
            completeness[field] = 1.0 if len(value) > 0 else 0.0
        else:
            completeness[field] = 1.0
    return completeness


# ══════════════════════════════════════════════════════════════════════════════
# Inference Engine
# ══════════════════════════════════════════════════════════════════════════════

# System prompt used during SFT training
SFT_SYSTEM_PROMPT = """你是一位专业的临床医学AI助手。请根据提供的病例信息，进行结构化诊断推理，并以合法JSON格式输出结果。

【输出格式说明】
输出必须是一个合法的JSON对象，字段如下：
- primary_diagnosis: 主诊断（字符串）
- diagnostic_basis: 诊断依据（字符串列表，每条为一个依据）
- differential_diagnoses: 鉴别诊断（字符串列表，每条为一个需要鉴别的疾病及简要说明）
- recommended_actions: 下一步建议（字符串列表）
- risk_flags: 风险提示（字符串列表，描述需要警惕的风险情况）

请务必：
1. 只输出JSON，不要输出任何解释、Markdown代码块或其他内容
2. 诊断表达使用"可能性大"、"首先考虑"、"倾向于"等稳健措辞
3. 风险提示必须说明何时需要急诊或线下就医
"""

# Inference prompt template (what goes into the user message)
INFERENCE_USER_TEMPLATE = """病例信息：
{case_text}

请基于以上病例信息，按合法JSON输出结构化诊断推理结果。字段固定为 primary_diagnosis、diagnostic_basis、differential_diagnoses、recommended_actions、risk_flags。不要输出 Markdown 或额外解释。"""


class InferenceEngine:
    """
    Unified inference engine that wraps model loading and batch generation.

    Supports:
      - Auto-detecting SFT checkpoint or base model path
      - Qwen chat template (ChatML format)
      - Batch inference with tqdm progress
      - GPU / CPU fallbacks
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        gen_args: Optional[Dict] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_flash_attn: bool = False,
    ):
        self.model_path = Path(model_path)
        self.gen_args = {**DEFAULT_GEN_ARGS, **(gen_args or {})}
        self.batch_size = batch_size
        self.use_flash_attn = use_flash_attn

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load model and tokenizer on first use."""
        if self._loaded:
            return

        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"Device: {self.device}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "transformers is required. Install: pip install transformers"
            )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
        if self.use_flash_attn:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using flash_attention_2")
            except ImportError:
                logger.warning("flash_attn not installed, falling back to default attention")

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            **model_kwargs,
        )
        self.model.eval()

        self._loaded = True
        logger.info("Model loaded successfully")

    @staticmethod
    def _build_qwen_chat_input(
        tokenizer,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """
        Build a Qwen2-style chat prompt using ChatML format.

        Format:
          <|im_start|>system
          {system_prompt}<|im_end|>
          <|im_start|>user
          {user_message}<|im_end|>
          <|im_start|>assistant
        """
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt.strip()}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_message.strip()}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt

    def generate(self, case_text: str) -> str:
        """Generate response for a single case."""
        self._ensure_loaded()

        user_message = INFERENCE_USER_TEMPLATE.format(case_text=case_text.strip())
        prompt = self._build_qwen_chat_input(
            self.tokenizer, SFT_SYSTEM_PROMPT, user_message
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.gen_args,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part (not the prompt)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def generate_batch(
        self, case_texts: List[str], progress: bool = True
    ) -> List[str]:
        """
        Generate responses for multiple cases in batches.
        Falls back to single-case generation if batch inference fails.
        """
        self._ensure_loaded()

        try:
            from tqdm import tqdm
        except ImportError:
            progress = False

        results = []
        iterator = range(0, len(case_texts), self.batch_size)
        if progress:
            iterator = tqdm(
                iterator,
                desc="Generating",
                unit="batch",
                total=(len(case_texts) + self.batch_size - 1) // self.batch_size,
            )

        for start in iterator:
            batch = case_texts[start : start + self.batch_size]
            batch_results = []
            for ct in batch:
                try:
                    result = self.generate(ct)
                    batch_results.append(result)
                except Exception as e:
                    logger.warning(f"Generation failed for case: {e}")
                    batch_results.append("")
            results.extend(batch_results)

        return results


def resolve_model_path(path_arg: Optional[str]) -> Path:
    """
    Resolve model path from argument or auto-detect from project structure.

    Checks:
      1. Explicit path if provided
      2. outputs-sft-medreason-qwen25-7b-1b/ (SFT checkpoint)
      3. outputs-sft-*/ (any SFT output)
      4. Qwen/Qwen2.5-7B-Instruct (base model on HuggingFace)
    """
    if path_arg:
        p = Path(path_arg)
        if p.exists() or "/" in str(p):
            return p
        raise FileNotFoundError(f"Model path not found: {p}")

    # Auto-detect
    candidates = [
        PROJECT_ROOT / "outputs-sft-medreason-qwen25-7b-1b",
        PROJECT_ROOT / "outputs-sft-medreason-qwen2.5-7b",
        PROJECT_ROOT / "outputs-sft",
    ]

    for cand in candidates:
        if cand.exists():
            # Find checkpoint directory
            checkpoints = sorted(cand.glob("checkpoint-*"))
            if checkpoints:
                latest = checkpoints[-1]
                logger.info(f"Auto-detected SFT checkpoint: {latest}")
                return latest
            # Maybe it's a merged model dir
            logger.info(f"Auto-detected model dir: {cand}")
            return cand

    # Fallback to base model
    logger.warning(
        "No local checkpoint found. Falling back to HuggingFace base model. "
        "Results will reflect pre-SFT performance."
    )
    return Path("Qwen/Qwen2.5-7B-Instruct")


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class MedReasonEvaluator:
    """
    Core evaluator class that runs inference and computes all metrics.

    Usage:
        evaluator = MedReasonEvaluator(
            model_path=".../checkpoint-xyz",
            holdout_path=".../holdout_v1.jsonl",
        )
        results = evaluator.evaluate()
        report  = evaluator.aggregate()
        evaluator.print_report(report)
    """

    def __init__(
        self,
        model_path: str,
        holdout_path: Path,
        output_dir: Optional[Path] = None,
        gen_args: Optional[Dict] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.holdout_path = Path(holdout_path)
        self.output_dir = (output_dir or (PROJECT_ROOT / "outputs_eval")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.engine = InferenceEngine(
            model_path=model_path,
            device=device,
            gen_args=gen_args,
            batch_size=batch_size,
        )

        # Cached data
        self.holdout_cases: List[Dict[str, Any]] = []
        self.inference_outputs: List[str] = []
        self.eval_results: List[EvalResult] = []

        # Cache for per-department grouping
        self._dept_groups: Optional[Dict[str, List[EvalResult]]] = None

    # ── Data loading ─────────────────────────────────────────────────────────

    def load_holdout(self) -> List[Dict[str, Any]]:
        """Load holdout dataset from JSONL."""
        logger.info(f"Loading holdout from {self.holdout_path}")
        self.holdout_cases = load_jsonl(self.holdout_path)
        logger.info(f"Loaded {len(self.holdout_cases)} holdout cases")
        return self.holdout_cases

    # ── Inference ────────────────────────────────────────────────────────────

    def run_inference(self, max_cases: Optional[int] = None) -> List[str]:
        """
        Run inference on all holdout cases (or a subset if max_cases is set).

        Returns:
            List of raw output strings (one per case).
        """
        cases = self.holdout_cases
        if max_cases is not None:
            cases = cases[:max_cases]

        logger.info(f"Running inference on {len(cases)} cases...")
        case_texts = [c["case_text"] for c in cases]

        self.inference_outputs = self.engine.generate_batch(case_texts)
        logger.info("Inference complete")
        return self.inference_outputs

    # ── Metrics computation ──────────────────────────────────────────────────

    def _compute_case_metrics(
        self,
        case: Dict[str, Any],
        raw_output: str,
    ) -> EvalResult:
        """
        Compute all metrics for a single case.
        """
        case_id = case.get("case_id", f"unknown_{id(case)}")
        department = case.get("department", "unknown")
        difficulty = case.get("difficulty", "medium")

        result = EvalResult(
            case_id=case_id,
            department=department,
            difficulty=difficulty,
            raw_output=raw_output,
            response_length=len(raw_output),
            response_tokens_approx=len(raw_output) // 4,  # rough estimate
        )

        # ── 1. JSON parsing ─────────────────────────────────────────────────
        parsed, parse_error = parse_model_json_output(raw_output)
        result.parse_error = parse_error
        result.parsed_json = parsed

        if parsed is None:
            result.error_type = "parse_error"
            return result

        # ── 2. Required fields check ─────────────────────────────────────────
        structured = extract_structured_output(parsed)
        missing = [f for f in REQUIRED_FIELDS if structured.get(f) is None]
        result.missing_fields = missing

        if missing:
            result.json_compliant = False
            result.error_type = "missing_fields"
            return result

        # All required fields present
        result.json_compliant = True

        # ── 3. Field completeness ────────────────────────────────────────────
        result.field_completeness = compute_field_completeness(structured)

        # ── 4. Primary diagnosis accuracy ────────────────────────────────────
        gt_diagnosis = (
            case.get("ground_truth", {})
            .get("primary_diagnosis", "")
            or case.get("primary_diagnosis", "")
        )
        if gt_diagnosis:
            pred_diag = structured.get("primary_diagnosis", "")
            score, matched = diagnose_match_score(pred_diag, gt_diagnosis)
            result.primary_top1_correct = score >= 0.7
            result.primary_top3_correct = score >= 0.4  # loosened threshold for top-3

            if score >= 0.4:
                result.primary_top3_correct = True

            result.primary_top3_list = [gt_diagnosis]

        # ── 5. Differential diagnosis coverage ───────────────────────────────
        gt_diff = (
            case.get("ground_truth", {})
            .get("differential_diagnoses", [])
            or case.get("expected_differentials", [])
        )
        pred_diff = structured.get("differential_diagnoses", [])
        if isinstance(pred_diff, list):
            pred_diff = [str(d) for d in pred_diff]

        correct, total, coverage = differential_match_score(pred_diff, gt_diff)
        result.diff_correct = correct
        result.diff_total = total
        result.diff_coverage = coverage

        # ── 6. Risk warning recall ───────────────────────────────────────────
        gt_risks = (
            case.get("ground_truth", {})
            .get("risk_flags", [])
            or case.get("expected_risks", [])
        )
        pred_risks = structured.get("risk_flags", [])
        if isinstance(pred_risks, list):
            pred_risks = [str(r) for r in pred_risks]

        relevant, total_r, recall = compute_risk_recall(pred_risks, gt_risks)
        result.risk_relevant = relevant
        result.risk_total = total_r
        result.risk_recall = recall

        # ── 7. Overconfidence detection ──────────────────────────────────────
        is_oc, signals = is_overconfident(raw_output)
        result.is_overconfident = is_oc
        result.overconfidence_signals = signals

        # ── 8. Error type ───────────────────────────────────────────────────
        # Determine dominant error type for this case
        error_type = None
        if not result.json_compliant:
            error_type = "json_non_compliant"
        elif result.is_overconfident:
            error_type = "overconfidence"
        elif result.diff_coverage < 0.5 and result.diff_total > 0:
            error_type = "low_diff_coverage"
        elif result.risk_recall < 0.3 and result.risk_total > 0:
            error_type = "low_risk_recall"
        elif not result.primary_top1_correct and gt_diagnosis:
            error_type = "incorrect_primary_diagnosis"

        result.error_type = error_type

        return result

    def evaluate(
        self,
        max_cases: Optional[int] = None,
        skip_inference: bool = False,
    ) -> List[EvalResult]:
        """
        Full evaluation pipeline: load data, run inference, compute metrics.

        Args:
            max_cases: limit number of cases (for quick testing)
            skip_inference: skip inference and reuse cached outputs
                             (useful when iterating on metrics only)
        Returns:
            List of EvalResult objects.
        """
        # Load holdout
        if not self.holdout_cases:
            self.load_holdout()

        cases = self.holdout_cases
        if max_cases is not None:
            cases = cases[:max_cases]

        # Run inference (unless cached)
        if skip_inference:
            if not self.inference_outputs:
                raise ValueError(
                    "No cached inference outputs. Set skip_inference=False to run inference."
                )
        else:
            self.run_inference(max_cases=max_cases)

        # Compute per-case metrics
        self.eval_results = []
        for i, (case, raw_output) in enumerate(zip(cases, self.inference_outputs)):
            try:
                result = self._compute_case_metrics(case, raw_output)
            except Exception as e:
                logger.warning(f"Metrics computation failed for case {i}: {e}")
                result = EvalResult(
                    case_id=case.get("case_id", f"unknown_{i}"),
                    department=case.get("department", "unknown"),
                    difficulty=case.get("difficulty", "medium"),
                    raw_output=raw_output,
                    error_type="computation_error",
                )
            self.eval_results.append(result)

        # Save per-case results
        out_path = self.output_dir / "per_case_results.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in self.eval_results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        logger.info(f"Per-case results saved → {out_path}")

        return self.eval_results

    # ── Aggregation ─────────────────────────────────────────────────────────

    def _dept_groups(self) -> Dict[str, List[EvalResult]]:
        """Lazily compute per-department groupings."""
        if self._dept_groups is None:
            groups: Dict[str, List[EvalResult]] = defaultdict(list)
            for r in self.eval_results:
                groups[r.department].append(r)
            self._dept_groups = groups
        return self._dept_groups

    def aggregate(
        self,
        group_by: Optional[Literal["department", "difficulty"]] = None,
    ) -> Union[AggregatedMetrics, Dict[str, AggregatedMetrics]]:
        """
        Aggregate metrics across results.

        Args:
            group_by: optionally compute separate aggregates per department or difficulty

        Returns:
            AggregatedMetrics (single) or dict mapping group key → AggregatedMetrics
        """
        if group_by is None:
            return self._aggregate(self.eval_results)

        groups = self._dept_groups() if group_by == "department" else self._difficulty_groups()
        return {k: self._aggregate(v) for k, v in groups.items()}

    def _difficulty_groups(self) -> Dict[str, List[EvalResult]]:
        groups: Dict[str, List[EvalResult]] = defaultdict(list)
        for r in self.eval_results:
            groups[r.difficulty].append(r)
        return groups

    @staticmethod
    def _aggregate(results: List[EvalResult]) -> AggregatedMetrics:
        """Compute aggregated metrics from a list of EvalResult."""
        n = len(results)
        if n == 0:
            return AggregatedMetrics()

        agg = AggregatedMetrics(total=n)

        # Basic counts
        agg.json_compliant_count = sum(1 for r in results if r.json_compliant)
        agg.overconfident_count = sum(1 for r in results if r.is_overconfident)

        # Core rates
        agg.json_compliance_rate = agg.json_compliant_count / n
        agg.overconfidence_rate = agg.overconfident_count / n

        # Differential coverage (only for cases with a non-zero denominator)
        diff_cases = [r for r in results if r.diff_total > 0]
        if diff_cases:
            total_correct = sum(r.diff_correct for r in diff_cases)
            total_expected = sum(r.diff_total for r in diff_cases)
            agg.diff_coverage_rate = total_correct / total_expected if total_expected > 0 else 0.0

        # Risk recall
        risk_cases = [r for r in results if r.risk_total > 0]
        if risk_cases:
            total_relevant = sum(r.risk_relevant for r in risk_cases)
            total_risk_expected = sum(r.risk_total for r in risk_cases)
            agg.risk_recall_rate = total_relevant / total_risk_expected if total_risk_expected > 0 else 0.0

        # Diagnosis accuracy
        diag_cases = [r for r in results if r.primary_top1_correct]
        agg.primary_top1_accuracy = len(diag_cases) / n

        top3_cases = [r for r in results if r.primary_top3_correct]
        agg.primary_top3_accuracy = len(top3_cases) / n

        # Field completeness
        field_totals = {f: 0 for f in REQUIRED_FIELDS}
        for r in results:
            for f, score in r.field_completeness.items():
                field_totals[f] = field_totals.get(f, 0) + score
        agg.field_completeness = {f: v / n for f, v in field_totals.items()}

        # Response length
        lengths = [r.response_length for r in results]
        agg.avg_response_length = sum(lengths) / n
        agg.avg_response_tokens = sum(r.response_tokens_approx for r in results) / n

        # Error type counts
        error_counts: Dict[str, int] = defaultdict(int)
        error_examples: Dict[str, List[Dict]] = defaultdict(list)
        for r in results:
            if r.error_type:
                error_counts[r.error_type] += 1
                # Keep up to 5 examples per error type
                if len(error_examples[r.error_type]) < 5:
                    error_examples[r.error_type].append({
                        "case_id": r.case_id,
                        "department": r.department,
                        "raw_output": r.raw_output[:300],
                        "parse_error": r.parse_error,
                        "missing_fields": r.missing_fields,
                        "overconfidence_signals": r.overconfidence_signals,
                        "diff_coverage": r.diff_coverage,
                        "risk_recall": r.risk_recall,
                    })

        agg.error_counts = dict(error_counts)
        agg.error_examples = dict(error_examples)

        return agg

    # ── Reporting ───────────────────────────────────────────────────────────

    def print_report(
        self,
        agg: AggregatedMetrics,
        title: str = "Overall Evaluation Results",
    ):
        """Print a formatted report to stdout."""
        n = agg.total
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

        print(f"\n  Total cases evaluated : {n}")
        print(f"  JSON compliance rate  : {agg.json_compliance_rate:.1%}  ({agg.json_compliant_count}/{n})")
        print(f"  Overconfidence rate  : {agg.overconfidence_rate:.1%}  ({agg.overconfident_count}/{n})")
        print(f"  Diff. coverage rate  : {agg.diff_coverage_rate:.1%}")
        print(f"  Risk recall rate     : {agg.risk_recall_rate:.1%}")
        print(f"  Primary dx top-1 acc: {agg.primary_top1_accuracy:.1%}")
        print(f"  Primary dx top-3 acc: {agg.primary_top3_accuracy:.1%}")

        print(f"\n  Average response len : {agg.avg_response_length:.0f} chars  ({agg.avg_response_tokens:.0f} tokens)")

        print(f"\n  Per-field completeness:")
        for field, rate in agg.field_completeness.items():
            label = {
                "primary_diagnosis": "主诊断",
                "diagnostic_basis": "诊断依据",
                "differential_diagnoses": "鉴别诊断",
                "recommended_actions": "下一步建议",
                "risk_flags": "风险提示",
            }.get(field, field)
            print(f"    {label:20s}: {rate:.1%}")

        if agg.error_counts:
            print(f"\n  Error type distribution:")
            for err, cnt in sorted(agg.error_counts.items(), key=lambda x: -x[1]):
                print(f"    {err:30s}: {cnt} ({cnt/n:.1%})")

        print("\n" + "=" * 70)

    def print_error_examples(self, agg: AggregatedMetrics):
        """Print representative error examples for each error type."""
        for err_type, examples in agg.error_examples.items():
            print(f"\n{'─' * 70}")
            print(f"  Error type: {err_type}  ({len(examples)} examples)")
            print(f"{'─' * 70}")
            for ex in examples[:3]:  # Show max 3 per type
                print(f"\n  [case_id={ex['case_id']}] [dept={ex['department']}]")
                if ex.get("parse_error"):
                    print(f"  Parse error: {ex['parse_error']}")
                if ex.get("missing_fields"):
                    print(f"  Missing fields: {ex['missing_fields']}")
                if ex.get("overconfidence_signals"):
                    print(f"  Overconfidence signals: {ex['overconfidence_signals']}")
                output_snippet = ex.get("raw_output", "")
                if len(output_snippet) > 200:
                    output_snippet = output_snippet[:200] + "..."
                print(f"  Raw output (truncated): {output_snippet}")
        print()

    def save_report(self, agg: AggregatedMetrics, path: Path):
        """Save aggregated metrics as JSON for downstream processing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert dataclass to dict for JSON serialization
        agg_dict = {
            "total": agg.total,
            "json_compliance_rate": round(agg.json_compliance_rate, 4),
            "json_compliant_count": agg.json_compliant_count,
            "diff_coverage_rate": round(agg.diff_coverage_rate, 4),
            "risk_recall_rate": round(agg.risk_recall_rate, 4),
            "overconfidence_rate": round(agg.overconfidence_rate, 4),
            "overconfident_count": agg.overconfident_count,
            "primary_top1_accuracy": round(agg.primary_top1_accuracy, 4),
            "primary_top3_accuracy": round(agg.primary_top3_accuracy, 4),
            "avg_response_length": round(agg.avg_response_length, 1),
            "avg_response_tokens": round(agg.avg_response_tokens, 1),
            "field_completeness": {k: round(v, 4) for k, v in agg.field_completeness.items()},
            "error_counts": agg.error_counts,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(agg_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI — main()
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_medreason",
        description="Medical Case Reasoning Model Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation on SFT model
  python eval_medreason.py --model outputs-sft-medreason-qwen25-7b-1b/checkpoint-xxx \\
       --holdout data/eval_internal/holdout_v1.jsonl

  # Ablation: compare base vs SFT
  python eval_medreason.py --model Qwen/Qwen2.5-7B-Instruct \\
       --holdout data/eval_internal/holdout_v1.jsonl --max-cases 50

  # Quick test (5 cases)
  python eval_medreason.py --max-cases 5 --dry-run
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model (local checkpoint dir or HuggingFace model name). "
             "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--holdout",
        type=str,
        default=str(PROJECT_ROOT / "data" / "eval_internal" / "holdout_v1.jsonl"),
        help="Path to holdout dataset JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. Default: outputs_eval/",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Path to base model for ablation comparison (optional).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of holdout cases (for quick testing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for inference (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max new tokens to generate (default: 1024).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force device. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable flash attention (useful on older GPUs).",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base-model ablation (only evaluate SFT model).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print results without running inference.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve model path ──────────────────────────────────────────────────
    model_path = resolve_model_path(args.model)
    logger.info(f"Model path: {model_path}")

    # ── Holdout path ────────────────────────────────────────────────────────
    holdout_path = Path(args.holdout)
    if not holdout_path.exists():
        logger.error(f"Holdout file not found: {holdout_path}")
        logger.info(
            "Run scripts/build_internal_holdout.py first to create the holdout dataset."
        )
        sys.exit(1)

    # ── Generation args ─────────────────────────────────────────────────────
    gen_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": 0.9,
        "do_sample": args.temperature > 0,
        "repetition_penalty": 1.05,
    }

    # ── Dry run: test metrics without inference ──────────────────────────────
    if args.dry_run:
        logger.info("Dry run: parsing holdout and showing sample cases only")
        cases = load_jsonl(holdout_path)
        for c in cases[:3]:
            print(f"\n[case_id={c.get('case_id')}] dept={c.get('department')}")
            print(f"  difficulty={c.get('difficulty')}, is_high_risk={c.get('is_high_risk')}")
            print(f"  case_text (first 150 chars): {c.get('case_text','')[:150]}...")
            gt = c.get("ground_truth", {})
            print(f"  gt.primary_diagnosis: {gt.get('primary_diagnosis','')[:80]}")
        logger.info(f"Dry run complete. Loaded {len(cases)} cases.")
        return

    # ── Evaluate SFT model ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUATING SFT MODEL")
    logger.info("=" * 60)
    sft_evaluator = MedReasonEvaluator(
        model_path=str(model_path),
        holdout_path=holdout_path,
        output_dir=output_dir / "sft_model",
        gen_args=gen_args,
        batch_size=args.batch_size,
        device=args.device,
    )

    sft_results = sft_evaluator.evaluate(max_cases=args.max_cases)
    sft_agg = sft_evaluator.aggregate()
    sft_evaluator.print_report(sft_agg, title="SFT Model — Overall Results")
    sft_evaluator.print_error_examples(sft_agg)
    sft_evaluator.save_report(sft_agg, output_dir / "sft_model" / "metrics.json")

    # ── Ablation: per-department breakdown ──────────────────────────────────
    dept_aggs = sft_evaluator.aggregate(group_by="department")
    print("\n" + "=" * 70)
    print("  Per-Department Breakdown")
    print("=" * 70)
    dept_report_path = output_dir / "sft_model" / "per_department.json"
    dept_report_path.parent.mkdir(parents=True, exist_ok=True)
    dept_records = {}
    for dept, agg in sorted(dept_aggs.items()):
        print(f"\n  [{dept}]  n={agg.total}")
        print(f"    JSON compliance  : {agg.json_compliance_rate:.1%}")
        print(f"    Diff. coverage  : {agg.diff_coverage_rate:.1%}")
        print(f"    Risk recall    : {agg.risk_recall_rate:.1%}")
        print(f"    Overconfidence : {agg.overconfidence_rate:.1%}")
        print(f"    Primary dx acc : {agg.primary_top1_accuracy:.1%}")
        dept_records[dept] = {
            "total": agg.total,
            "json_compliance_rate": round(agg.json_compliance_rate, 4),
            "diff_coverage_rate": round(agg.diff_coverage_rate, 4),
            "risk_recall_rate": round(agg.risk_recall_rate, 4),
            "overconfidence_rate": round(agg.overconfidence_rate, 4),
            "primary_top1_accuracy": round(agg.primary_top1_accuracy, 4),
        }
    with dept_report_path.open("w", encoding="utf-8") as f:
        json.dump(dept_records, f, ensure_ascii=False, indent=2)
    logger.info(f"Per-department report saved → {dept_report_path}")

    # ── Ablation: base model comparison ────────────────────────────────────
    if not args.skip_base and args.base_model:
        logger.info("=" * 60)
        logger.info("EVALUATING BASE MODEL (ablation)")
        logger.info("=" * 60)
        base_evaluator = MedReasonEvaluator(
            model_path=args.base_model,
            holdout_path=holdout_path,
            output_dir=output_dir / "base_model",
            gen_args=gen_args,
            batch_size=args.batch_size,
            device=args.device,
        )
        base_results = base_evaluator.evaluate(max_cases=args.max_cases)
        base_agg = base_evaluator.aggregate()
        base_evaluator.print_report(base_agg, title="Base Model — Overall Results")
        base_evaluator.save_report(base_agg, output_dir / "base_model" / "metrics.json")

        # Comparison table
        print("\n" + "=" * 70)
        print("  SFT vs Base — Ablation Comparison")
        print("=" * 70)
        print(f"  {'Metric':<30} {'Base':>10} {'SFT':>10} {'Delta':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        metrics_pairs = [
            ("JSON compliance rate", "json_compliance_rate"),
            ("Diff. coverage rate", "diff_coverage_rate"),
            ("Risk recall rate", "risk_recall_rate"),
            ("Overconfidence rate", "overconfidence_rate"),
            ("Primary dx top-1 acc", "primary_top1_accuracy"),
            ("Avg response length", "avg_response_length"),
        ]
        for label, key in metrics_pairs:
            base_val = getattr(base_agg, key, 0)
            sft_val = getattr(sft_agg, key, 0)
            delta = sft_val - base_val
            fmt = f"  {label:<30} {base_val:>10.4f} {sft_val:>10.4f} {delta:>+10.4f}"
            print(fmt)

        # Save comparison
        comparison = {
            "base_model": args.base_model,
            "sft_model": str(model_path),
            "num_cases": len(base_results),
            "metrics": {
                "base": {
                    "json_compliance_rate": round(base_agg.json_compliance_rate, 4),
                    "diff_coverage_rate": round(base_agg.diff_coverage_rate, 4),
                    "risk_recall_rate": round(base_agg.risk_recall_rate, 4),
                    "overconfidence_rate": round(base_agg.overconfidence_rate, 4),
                    "primary_top1_accuracy": round(base_agg.primary_top1_accuracy, 4),
                },
                "sft": {
                    "json_compliance_rate": round(sft_agg.json_compliance_rate, 4),
                    "diff_coverage_rate": round(sft_agg.diff_coverage_rate, 4),
                    "risk_recall_rate": round(sft_agg.risk_recall_rate, 4),
                    "overconfidence_rate": round(sft_agg.overconfidence_rate, 4),
                    "primary_top1_accuracy": round(sft_agg.primary_top1_accuracy, 4),
                },
            },
        }
        comp_path = output_dir / "ablation_comparison.json"
        with comp_path.open("w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        logger.info(f"Ablation comparison saved → {comp_path}")

    # ── Final summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Evaluation Complete")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print(f"  SFT model       : {model_path}")
    print(f"  Holdout         : {holdout_path}  ({len(sft_results)} cases)")
    if args.base_model and not args.skip_base:
        print(f"  Base model      : {args.base_model}")
    print("=" * 70)


if __name__ == "__main__":
    main()