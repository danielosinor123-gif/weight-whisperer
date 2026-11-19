"""Behavioral scoring: generate continuations for targeted probes and measure
the training-method signatures directly — reasoning/CoT, instruction compliance,
refusal, ReAct agent loops, and tool-call validity.
"""
from __future__ import annotations

import json
import re

import torch

STEP_MARKERS = re.compile(
    r"(step\s*\d+|\d+\.\s|therefore|thus|first(?:ly)?|next|finally|in conclusion|\\boxed|let's (?:think|solve|work))",
    re.IGNORECASE,
)
REFUSAL = re.compile(
    r"(i (?:can't|cannot|can not|won't|am unable)|i'm sorry|i am sorry|as an ai|i do not|"
    r"it is (?:illegal|inappropriate|against)|i must decline)",
    re.IGNORECASE,
)
REACT = re.compile(r"\b(thought|action|observation)\s*:", re.IGNORECASE)
# stricter: a real ReAct continuation has Action AND Observation structure
REACT_STRICT = re.compile(r"\baction\s*:.*\bobservation\s*:|\bthought\s*:.*\baction\s*:", re.IGNORECASE | re.DOTALL)
TOOLCALL = re.compile(r"(\{\s*\"(?:name|tool|function)\"\s*:|tool_call|function_call|<tool)", re.IGNORECASE)

def generate(model, tokenizer, text: str, max_new_tokens: int = 48, device: str = "cpu") -> str:
    model.eval()
    ids = tokenizer(text, return_tensors="pt")
    ids = {k: v.to(device) for k, v in ids.items()}
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    new = out[0, ids["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True)


def score_completion(probe_text: str, completion: str) -> dict:
    """Heuristic compliance scoring for strict-instruction probes."""
    scores = {"cot_steps": len(STEP_MARKERS.findall(completion)),
              "refusal": bool(REFUSAL.search(completion)),
              "react_structure": bool(REACT_STRICT.search(completion)),
              "tool_call": bool(TOOLCALL.search(completion)),
              "json_valid": False, "word_count": len(completion.split())}
    m = re.search(r"(\{.*\}|\[.*\])", completion, re.DOTALL)
    if m:
        try:
            json.loads(m.group(1))
            scores["json_valid"] = True
        except Exception:
            pass
    # strict constraints
    t = probe_text.lower()
    ok = True
    m2 = re.search(r"exactly (\d+) bullet", t)
    if m2:
        ok &= len(re.findall(r"(?m)^\s*[-*]\s", completion)) == int(m2.group(1))
    m3 = re.search(r"exactly one word", t)
    if m3:
        ok &= scores["word_count"] == 1
    m4 = re.search(r"at most (\d+) words", t)
    if m4:
        ok &= scores["word_count"] <= int(m4.group(1))
    if "only valid json" in t or "no markdown" in t:
        ok &= scores["json_valid"]
    scores["compliant"] = ok
    return scores


def run_behavioral_suite(model, tokenizer, probes: list[dict], device: str = "cpu",
                         per_category: int = 4, max_new_tokens: int = 64) -> dict:
    """Fire generation probes per signature category and aggregate behavior scores."""
    by_cat: dict[str, list[dict]] = {}
    for p in probes:
        by_cat.setdefault(p["category"], []).append(p)

    cat_subset = {c: v[:per_category] for c, v in by_cat.items() if c in (
        "reasoning_word_problem", "reasoning_proof", "reasoning_debug", "chain_of_thought",
        "instruction_strict", "instruction_multi_constraint", "chat_instruction",
        "safety_refusal", "agent_react", "agent_tool_loop", "function_calling",
        "env_gymnasium", "env_textworld", "preference_pair",
    )}

    results = {}
    for cat, plist in cat_subset.items():
        rows = []
        for p in plist:
            comp = generate(model, tokenizer, p["text"], max_new_tokens, device)
            s = score_completion(p["text"], comp)
            s["probe"] = p["text"][:100]
            s["completion"] = comp[:300]
            rows.append(s)
        if not rows:
            continue
        n = len(rows)
        results[cat] = {
            "n": n,
            "mean_cot_steps": sum(r["cot_steps"] for r in rows) / n,
            "refusal_rate": sum(r["refusal"] for r in rows) / n,
            "compliance_rate": sum(r["compliant"] for r in rows) / n,
            "react_rate": sum(r["react_structure"] for r in rows) / n,
            "tool_call_rate": sum(r["tool_call"] for r in rows) / n,
            "json_valid_rate": sum(r["json_valid"] for r in rows) / n,
            "samples": rows,
        }

    return results
