"""Synthesize a dataset-recipe hypothesis from probe firing patterns.

Combines:
  - per-category activation selectivity per layer (which layers care about what)
  - layer-type profiles (early/mid/late specialization)
  - vocabulary / chat-template signals
  - top predicted tokens per category (behavioral confirmation)

Outputs recipe.json + a human-readable recipe_report.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .capture import role_of
from .targeting import build_activation_matrix, category_selectivity, layer_coverage


def _layer_key(module_name: str) -> int:
    """Extract numeric layer index from module name for early/mid/late grouping."""
    parts = module_name.split(".")
    for p in parts:
        if p.isdigit():
            return int(p)
    return -1


# training-method signature -> (probe categories, behavior metric, how to read it)
SIGNATURE_DEFS = {
    "reasoning_step_by_step": {
        "categories": ["chain_of_thought", "reasoning_word_problem", "reasoning_proof", "reasoning_debug"],
        "behavior": ("mean_cot_steps", "mean across reasoning-category generations"),
        "meaning": "Model trained on worked, step-by-step reasoning traces (CoT/SFT on solutions).",
    },
    "instruction_following": {
        "categories": ["chat_instruction", "system_prompt_style", "instruction_strict",
                       "instruction_multi_constraint"],
        "behavior": ("compliance_rate", "fraction of strict-constraint probes followed"),
        "meaning": "Model trained on instruction-tuned (SFT) data with format constraints.",
    },
    "safety_alignment": {
        "categories": ["safety_refusal"],
        "behavior": ("refusal_rate", "fraction of harmful-request probes refused"),
        "meaning": "Model has RLHF/safety-alignment refusal training.",
    },
    "agent_environment_interaction": {
        "categories": ["agent_react", "agent_tool_loop", "function_calling",
                       "env_gymnasium", "env_textworld", "reward_feedback"],
        "behavior": ("react_rate", "fraction of ReAct probes continued in Thought/Action/Observation format"),
        "meaning": "Model trained on agentic trajectories / tool use / RL environment interactions.",
    },
    "preference_optimization": {
        "categories": ["preference_pair"],
        "behavior": None,
        "meaning": "Model saw pairwise preference comparisons (RLHF/DPO-style data).",
    },
}


def _confidence(selectivity: float, behavior_value: float | None, behavior_high: float,
                behavior_low: float = 0.0, n_probes: int = 0) -> str:
    """Combine activation selectivity + behavioral metric into low/medium/high.
    Requires minimum evidence (n_probes) before allowing 'high'."""
    if n_probes < 4:
        return "low"  # insufficient probe evidence
    score = 0
    if selectivity >= 2.0:
        score += 2
    elif selectivity >= 1.2:
        score += 1
    if behavior_value is not None:
        if behavior_value >= behavior_high:
            score += 2
        elif behavior_value >= (behavior_high + behavior_low) / 2:
            score += 1
    return {2: "high", 1: "medium", 0: "low"}.get(score, "high" if score >= 3 else "medium")


def build_signatures(category_summary: dict, behavior: dict) -> dict:
    sigs = {}
    for name, spec in SIGNATURE_DEFS.items():
        sels = [category_summary[c]["selectivity"] for c in spec["categories"] if c in category_summary]
        mean_sel = float(np.mean(sels)) if sels else 0.0
        beh = None
        if spec["behavior"]:
            metric, _ = spec["behavior"]
            vals = []
            for c in spec["categories"]:
                if c in behavior and metric in behavior[c]:
                    vals.append(behavior[c][metric])
            beh = float(np.mean(vals)) if vals else None
        high = {"mean_cot_steps": 2.0, "compliance_rate": 0.5, "refusal_rate": 0.5,
                "react_rate": 0.3}.get(spec["behavior"][0], 0.5) if spec["behavior"] else 0.5
        n_probes = max((category_summary[c]["n_probes"] for c in spec["categories"] if c in category_summary), default=0)
        sigs[name] = {
            "mean_category_selectivity": round(mean_sel, 3),
            "behavior_metric": spec["behavior"][0] if spec["behavior"] else None,
            "behavior_value": round(beh, 3) if beh is not None else None,
            "confidence": _confidence(mean_sel, beh, high, n_probes=n_probes),
            "n_evidence_probes": n_probes,
            "evidence_categories": spec["categories"],
            "interpretation": spec["meaning"],
        }
    return sigs


def synthesize(records, hooked_modules: dict, vocab_signals: dict,
               logit_salience: dict, model_name: str, out_dir: Path,
               behavior: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    modules = sorted(hooked_modules.keys())
    layer_indices = sorted({_layer_key(m) for m in modules if _layer_key(m) >= 0})
    n_layers = max(layer_indices) + 1 if layer_indices else 1

    # ---- per-category selectivity, aggregated over MLP modules (most neuron-like)
    NEURON_ROLES = ("mlp", "mlp.gate", "mlp.up", "mlp.fc1", "mlp.fc2")
    neuron_modules = [m for m in modules if role_of(m) in NEURON_ROLES]
    per_cat = {}
    for m in neuron_modules:
        matrix, cats = build_activation_matrix(records, m)
        if matrix is None:
            continue
        sel = category_selectivity(matrix, cats)
        for c, stats in sel.items():
            agg = per_cat.setdefault(c, {"selectivity": [], "frac_selective_units": [],
                                         "mean_abs_activation": [], "n_probes": stats["n_probes"]})
            agg["selectivity"].append(stats["selectivity"])
            agg["frac_selective_units"].append(stats["frac_selective_units"])
            agg["mean_abs_activation"].append(stats["mean_abs_activation"])

    category_summary = {}
    for c, agg in per_cat.items():
        category_summary[c] = {
            "selectivity": float(np.mean(agg["selectivity"])),
            "frac_selective_units": float(np.mean(agg["frac_selective_units"])),
            "mean_abs_activation": float(np.mean(agg["mean_abs_activation"])),
            "n_probes": agg["n_probes"],
        }

    # ---- layer specialization: which third of the network responds to each category
    layer_specialization = {}
    for m in neuron_modules:
        li = _layer_key(m)
        if li < 0:
            continue
        third = "early" if li < n_layers / 3 else ("mid" if li < 2 * n_layers / 3 else "late")
        matrix, cats = build_activation_matrix(records, m)
        if matrix is None:
            continue
        cats_arr = np.array(cats)
        for c in set(cats):
            sub = matrix[cats_arr == c]
            if len(sub) == 0:
                continue
            layer_specialization.setdefault(c, {"early": [], "mid": [], "late": []})
            layer_specialization[c][third].append(float(np.abs(sub).mean()))
    for c, d in layer_specialization.items():
        for k in d:
            d[k] = float(np.mean(d[k])) if d[k] else 0.0

    # ---- coverage
    coverage = layer_coverage(records, modules)
    variable_fracs = [v["frac_units_variable"] for v in coverage.values()]
    coverage_summary = {
        "n_modules_hooked": len(modules),
        "n_probes": len(records),
        "mean_frac_units_variable": float(np.mean(variable_fracs)) if variable_fracs else 0.0,
    }

    # ---- chat template detection
    chat_hits = [h for h in vocab_signals.get("chat_template", []) if h["in_vocab"]]
    format_hypothesis = "unknown"
    token_strs = " ".join(h["token"] for h in chat_hits)
    if "<|im_start|>" in token_strs:
        format_hypothesis = "ChatML (<|im_start|>/<|im_end|>)"
    elif "[INST]" in token_strs:
        format_hypothesis = "Llama-style [INST]"
    elif "<start_of_turn>" in token_strs:
        format_hypothesis = "Gemma-style turns"
    elif chat_hits:
        format_hypothesis = "Custom special tokens: " + ", ".join(h["token"] for h in chat_hits[:6])

    recipe = {
        "model": model_name,
        "n_layers": n_layers,
        "coverage": coverage_summary,
        "detected_format": {
            "chat_template_hypothesis": format_hypothesis,
            "special_tokens_found": [h["token"] for h in chat_hits],
            "vocab_signals": vocab_signals,
        },
        "training_method_signatures": build_signatures(category_summary, behavior or {}),
        "task_mixture_evidence": dict(sorted(category_summary.items(),
                                             key=lambda kv: -kv[1]["selectivity"])),
        "layer_specialization": layer_specialization,
        "behavioral_samples": logit_salience,
        "behavioral_scores": behavior or {},
    }

    (out_dir / "recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    (out_dir / "recipe_report.md").write_text(_render_markdown(recipe), encoding="utf-8")
    return recipe


def _render_markdown(recipe: dict) -> str:
    lines = []
    a = lines.append
    a(f"# Dataset Recipe Hypothesis — {recipe['model']}\n")
    a("## What this is\n")
    a("Reverse-engineered from targeted activation probes: every probe's per-unit firing "
      "pattern was recorded across all hooked modules, then aggregated into per-category "
      "selectivity and layer specialization. This is a *hypothesis* about the training data, "
      "not ground truth.\n")
    a("## Coverage\n")
    cov = recipe["coverage"]
    a(f"- Probes fired: **{cov['n_probes']}**")
    a(f"- Modules hooked: **{cov['n_modules_hooked']}**")
    a(f"- Mean fraction of units showing variable responses: **{cov['mean_frac_units_variable']:.1%}**\n")
    a("## Detected format\n")
    a(f"- Chat template hypothesis: **{recipe['detected_format']['chat_template_hypothesis']}**")
    st = recipe["detected_format"]["special_tokens_found"]
    a(f"- Special tokens present in vocab: {', '.join('`' + t + '`' for t in st) if st else 'none detected'}\n")
    a("## Training method signatures\n")
    a("| Signature | Confidence | Neuron selectivity | Behavior metric | value | Interpretation |")
    a("|---|---|---|---|---|---|")
    for name, s in recipe["training_method_signatures"].items():
        bm = s["behavior_metric"] or "-"
        bv = s["behavior_value"] if s["behavior_value"] is not None else "-"
        a(f"| **{name}** | {s['confidence']} | {s['mean_category_selectivity']} | {bm} | {bv} | {s['interpretation']} |")
    a("")
    if recipe.get("behavioral_scores"):
        a("### Behavioral details\n")
        for cat, d in recipe["behavioral_scores"].items():
            a(f"- **{cat}** (n={d['n']}): CoT steps {d['mean_cot_steps']:.1f}, "
              f"compliance {d['compliance_rate']:.0%}, refusal {d['refusal_rate']:.0%}, "
              f"ReAct {d['react_rate']:.0%}, tool-call {d['tool_call_rate']:.0%}, "
              f"valid-JSON {d['json_valid_rate']:.0%}")
        a("")
    a("## Task mixture evidence (ranked by neuron selectivity)\n")
    a("| Category | Selectivity | % selective units | Mean |act| | Probes |")
    a("|---|---|---|---|---|")
    for cat, s in recipe["task_mixture_evidence"].items():
        a(f"| {cat} | {s['selectivity']:.3f} | {s['frac_selective_units']:.1%} | "
          f"{s['mean_abs_activation']:.3f} | {s['n_probes']} |")
    a("\n## Layer specialization (where each category fires)\n")
    a("| Category | Early | Mid | Late |")
    a("|---|---|---|---|")
    for cat, d in recipe["layer_specialization"].items():
        a(f"| {cat} | {d['early']:.3f} | {d['mid']:.3f} | {d['late']:.3f} |")
    a("\n## Interpretation guide\n")
    a("- **High selectivity + many selective units** → the model likely saw a lot of this data type.")
    a("- **Early-layer firing** → surface/lexical features (language, syntax, format tokens).")
    a("- **Late-layer firing** → task/format semantics (instruction following, output structure).")
    a("- **Chat template tokens in vocab** → strong direct evidence of the SFT/RLHF data format.")
    a("- Compare against `behavioral_samples` (top predicted tokens) to confirm behaviorally.\n")
    return "\n".join(lines)
