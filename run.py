"""CLI: fire agent-designed probes at a model, capture all per-unit activations,
and synthesize a dataset-recipe hypothesis.

Usage:
  python -m dataset_recipe_inferrer.run --model Qwen/Qwen2.5-0.5B-Instruct \
      --rounds 2 --probes-per-round 40 --out out_qwen
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .agent import TemplateAgent, LLMAgent
from .behavior import run_behavioral_suite
from .capture import ActivationCapture
from .logit_analysis import analyze_vocab, logit_salience
from .probes import build_probes, PROBE_SPECS
from .synthesizer import synthesize
from .targeting import build_activation_matrix, category_selectivity


def fire_probes(model, tokenizer, capture: ActivationCapture, probes, start_id: int,
                max_length: int = 256, device: str = "cpu"):
    records = []
    model.eval()
    with torch.no_grad():
        for i, p in enumerate(probes):
            rec = capture.start_probe(start_id + i, p["category"], p["text"])
            ids = tokenizer(p["text"], return_tensors="pt",
                            truncation=True, max_length=max_length)
            ids = {k: v.to(device) for k, v in ids.items()}
            out = model(**ids)
            capture.finish_probe(rec, tokenizer, out.logits, ids["input_ids"].shape[1])
            records.append(rec)
    return records


def quick_category_scores(records, hooked_modules) -> dict[str, float]:
    """Cheap per-category response score for the agent's next-round targeting."""
    scores = {}
    mlp_mods = [m for m in hooked_modules if "mlp" in m][:8]
    for m in mlp_mods:
        matrix, cats = build_activation_matrix(records, m)
        if matrix is None:
            continue
        sel = category_selectivity(matrix, cats)
        for c, s in sel.items():
            scores.setdefault(c, []).append(s["selectivity"])
    return {c: sum(v) / len(v) for c, v in scores.items()}


def main():
    ap = argparse.ArgumentParser(description="Agent-driven weight probing -> dataset recipe")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--probes-per-category", type=int, default=3)
    ap.add_argument("--probes-per-round", type=int, default=30)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--out", default="recipe_out")
    ap.add_argument("--llm-endpoint", default=None, help="Optional OpenAI-compatible endpoint")
    ap.add_argument("--llm-key", default=None)
    ap.add_argument("--seed", type=int, default=42)
    # scaling options for bigger models
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--layer-sample", type=int, default=1,
                    help="Hook only every k-th transformer layer (big models)")
    ap.add_argument("--roles", default=None,
                    help="Comma-separated role filter, e.g. mlp.down,attn.o (default: all)")
    ap.add_argument("--save-max-units", type=int, default=4096,
                    help="Cap units stored per module in activations.npz")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[load] {args.model} on {args.device} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] {n_params/1e6:.1f}M params")

    agent = LLMAgent(endpoint=args.llm_endpoint, api_key=args.llm_key, seed=args.seed) \
        if args.llm_endpoint else TemplateAgent(seed=args.seed)

    role_filter = set(args.roles.split(",")) if args.roles else None
    all_records = []
    with ActivationCapture(model, layer_sample=args.layer_sample,
                           role_filter=role_filter) as capture:
        print(f"[hook] {len(capture.hooked_modules)} modules hooked")
        next_id = 0

        # round 0: balanced seed bank
        probes = build_probes(per_category=args.probes_per_category, seed=args.seed)
        print(f"[round 0] firing {len(probes)} seed probes across {len(PROBE_SPECS)} categories")
        recs = fire_probes(model, tokenizer, capture, probes, next_id, args.max_length, args.device)
        next_id += len(probes)
        all_records.extend(recs)

        # adaptive rounds
        for r in range(1, args.rounds + 1):
            scores = quick_category_scores(all_records, capture.hooked_modules)
            probes = agent.propose(scores, n=args.probes_per_round)
            if not probes:
                break
            print(f"[round {r}] agent proposed {len(probes)} targeted probes "
                  f"(weakest: {sorted(scores.items(), key=lambda kv: kv[1])[:3]})")
            recs = fire_probes(model, tokenizer, capture, probes, next_id, args.max_length, args.device)
            next_id += len(probes)
            all_records.extend(recs)

        print("[analysis] vocabulary + logit signals")
        vocab_signals = analyze_vocab(tokenizer)
        sal_prompts = {c: [p["text"] for p in build_probes(2, args.seed) if p["category"] == c]
                       for c in list(PROBE_SPECS)[:8]}
        salience = logit_salience(model, tokenizer, sal_prompts)

        print("[behavior] generation-based signature scoring (reasoning/instruction/agent)")
        all_probes = [{"text": r.text, "category": r.category} for r in all_records]
        behavior = run_behavioral_suite(model, tokenizer, all_probes, device=args.device)

        print("[synthesis] building recipe")
        recipe = synthesize(all_records, capture.hooked_modules, vocab_signals,
                            salience, args.model, out_dir, behavior=behavior)

    # dump raw per-unit activations (compressed) for downstream analysis
    import numpy as np
    raw = {}
    for rec in all_records:
        for m, act in rec.activations.items():
            if act.get("last") is not None:
                raw.setdefault(m, {"vectors": [], "cats": [], "ids": []})
                v = act["last"].numpy()
                if len(v) > args.save_max_units:  # subsample units for storage
                    idx = np.linspace(0, len(v) - 1, args.save_max_units).astype(int)
                    v = v[idx]
                raw[m]["vectors"].append(v)
                raw[m]["cats"].append(rec.category)
                raw[m]["ids"].append(rec.probe_id)
    npz = {f"{m}::vectors": np.stack(v["vectors"]) for m, v in raw.items()}
    np.savez_compressed(out_dir / "activations.npz", **npz)
    (out_dir / "activation_meta.json").write_text(json.dumps(
        {m: {"cats": v["cats"], "ids": v["ids"]} for m, v in raw.items()}), encoding="utf-8")

    print(f"[done] {len(all_records)} probes in {time.time()-t0:.0f}s")
    print(f"[done] report: {out_dir/'recipe_report.md'}")
    print(f"[done] raw activations: {out_dir/'activations.npz'}")


if __name__ == "__main__":
    main()
