#!/usr/bin/env python3
"""Sample CoT/Mixed SFT completions before answer-boundary GRPO v2 training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_med_choice_rewards_v2 import score_choice_completion_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="/root/autodl-tmp/models/Qwen/Qwen/Qwen2___5-7B-Instruct")
    parser.add_argument("--peft-path", default="outputs/sft/cmb_cot_mixed")
    parser.add_argument("--input", default="data/grpo/cmb_grpo_pilot_v2_train.jsonl")
    parser.add_argument("--output-jsonl", default="data/grpo/cot_mixed_before_grpo_v2_samples.jsonl")
    parser.add_argument("--output-md", default="data/grpo/cot_mixed_before_grpo_v2_samples.md")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow very slow CPU generation if CUDA is unavailable.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_input_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def completion_from_generated(input_text: str, decoded: str) -> str:
    if decoded.startswith(input_text):
        return decoded[len(input_text) :].strip()
    marker = "请作答："
    pos = decoded.rfind(marker)
    if pos >= 0:
        return decoded[pos + len(marker) :].strip()
    return decoded.strip()


def load_model(args: argparse.Namespace) -> tuple[Any, Any, torch.device]:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable. Refusing to load a 7B model on CPU for debug sampling. "
            "Switch to GPU mode or pass --allow-cpu for a tiny manual run."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.peft_path, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    return tokenizer, model, device


def write_outputs(records: list[dict[str, Any]], output_jsonl: Path, output_md: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [
        "# CoT/Mixed Before GRPO v2 Completion Debug",
        "",
        f"- total_completions: {len(records)}",
        "- purpose: inspect answer-only prompt behavior before training.",
        "",
    ]
    for idx, record in enumerate(records, 1):
        breakdown = record["reward_breakdown"]
        lines.extend(
            [
                f"## {idx}. {record['sample_id']} / gen={record['generation_index']}",
                "",
                f"- gold: `{record['answer']}`",
                f"- pred: `{record['pred']}`",
                f"- answer_len: {record['answer_len']}",
                f"- reward: {breakdown['final_reward']:.4f}",
                f"- precision/recall/f1: {breakdown['precision']:.4f} / {breakdown['recall']:.4f} / {breakdown['f1']:.4f}",
                f"- extra/missing: {breakdown['extra_count']} / {breakdown['missing_count']}",
                f"- labels: {breakdown['answer_label_count']}, valid_extract: {breakdown['valid_extract']}, extra_text: {breakdown['extra_text_after_answer']}",
                "",
                "**Question**",
                "",
                record["question"],
                "",
                "**Completion**",
                "",
                "```text",
                record["completion"],
                "```",
                "",
            ]
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(Path(args.input))
    if not rows:
        raise ValueError(f"No rows found in {args.input}")
    sample_count = min(args.num_samples, len(rows))
    sampled = random.sample(rows, sample_count)

    tokenizer, model, device = load_model(args)
    records: list[dict[str, Any]] = []
    for row in sampled:
        input_text = build_input_text(tokenizer, row["prompt"])
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for gen_idx, decoded in enumerate(decoded_outputs):
            completion = completion_from_generated(input_text, decoded)
            score = score_choice_completion_v2(completion, row["answer"], row["valid_options"])
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "generation_index": gen_idx,
                    "question": row["question"],
                    "options": row["options"],
                    "answer": row["answer"],
                    "answer_len": row["answer_len"],
                    "prompt": row["prompt"],
                    "completion": completion,
                    "pred": score.pred,
                    "reward_breakdown": score.to_dict(),
                }
            )

    write_outputs(records, Path(args.output_jsonl), Path(args.output_md))
    print(f"wrote {len(records)} completions to {args.output_jsonl}")
    print(f"wrote markdown to {args.output_md}")


if __name__ == "__main__":
    main()
