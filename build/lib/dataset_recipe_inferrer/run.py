"""CLI + library: fire agent-designed probes at a model, capture all per-unit
activations, and synthesize a dataset-recipe hypothesis.

CLI (after `pip install -e .`):
    weight-whisperer --model Qwen/Qwen2.5-0.5B-Instruct --rounds 2 --out out_qwen
    # or without installing:
    python -m dataset_recipe_inferrer.run --model <hf-id-or-local-path> --out out_x

Library (bring your own probe agent):
    from dataset_recipe_inferrer.run import probe_model
    probe_model("Qwen/Qwen2.5-0.5B-Instruct", agent=MyAgent(), out="out_qwen")

Any object with `propose(category_scores: dict[str, float], n: int) -> list[dict]`
works as an agent; each probe dict needs {"text", "category", "hypothesis"}.
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


def probe_model(model: str, agent=None, *, rounds: int = 2, probes_per_category: int = 3,
                probes_per_round: int = 30, max_length: int = 256, out: str = "recipe_out",
                device: str = "cpu", dtype: str = "float32", layer_sample: int = 1,
                roles: str | None = None, save_max_units: int = 4096, seed: int = 42,
                llm_endpoint: str | None = None, llm_key: str | None = None,
                llm_model: str = "gpt-4o-mini") -> Path:
    """Probe `model` (HF hub id or local path) and write the recipe to `out/`.

    `agent` may be any object with propose(category_scores, n); if None, an
    LLMAgent is used when llm_endpoint is given, else the offline TemplateAgent.
    Returns the output directory.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[load] {model} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model)
    dt = getattr(torch, dtype)
    net = AutoModelForCausalLM.from_pretrained(model, dtype=dt).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[load] {n_params/1e6:.1f}M params")

    if agent is None:
        agent = LLMAgent(endpoint=llm_endpoint, api_key=llm_key, model=llm_model, seed=seed) \
            if llm_endpoint else TemplateAgent(seed=seed)
    print(f"[agent] {type(agent).__name__}")

    role_filter = set(roles.split(",")) if roles else None
    all_records = []
    with ActivationCapture(net, layer_sample=layer_sample,
                           role_filter=role_filter) as capture:
        print(f"[hook] {len(capture.hooked_modules)} modules hooked")
        next_id = 0

        # round 0: balanced seed bank
        probes = build_probes(per_category=probes_per_category, seed=seed)
        print(f"[round 0] firing {len(probes)} seed probes across {len(PROBE_SPECS)} categories")
        recs = fire_probes(net, tokenizer, capture, probes, next_id, max_length, device)
        next_id += len(probes)
        all_records.extend(recs)

        # adaptive rounds
        for r in range(1, rounds + 1):
            scores = quick_category_scores(all_records, capture.hooked_modules)
            probes = agent.propose(scores, n=probes_per_round)
            if not probes:
                break
            print(f"[round {r}] agent proposed {len(probes)} targeted probes "
                  f"(weakest: {sorted(scores.items(), key=lambda kv: kv[1])[:3]})")
            recs = fire_probes(net, tokenizer, capture, probes, next_id, max_length, device)
            next_id += len(probes)
            all_records.extend(recs)

        print("[analysis] vocabulary + logit signals")
        vocab_signals = analyze_vocab(tokenizer)
        sal_prompts = {c: [p["text"] for p in build_probes(2, seed) if p["category"] == c]
                       for c in list(PROBE_SPECS)[:8]}
        salience = logit_salience(net, tokenizer, sal_prompts)

        print("[behavior] generation-based signature scoring (reasoning/instruction/agent)")
        all_probes = [{"text": r.text, "category": r.category} for r in all_records]
        behavior = run_behavioral_suite(net, tokenizer, all_probes, device=device)

        print("[synthesis] building recipe")
        recipe = synthesize(all_records, capture.hooked_modules, vocab_signals,
                            salience, model, out_dir, behavior=behavior)

    # dump raw per-unit activations (compressed) for downstream analysis
    import numpy as np
    raw = {}
    for rec in all_records:
        for m, act in rec.activations.items():
            if act.get("last") is not None:
                raw.setdefault(m, {"vectors": [], "cats": [], "ids": []})
                v = act["last"].numpy()
                if len(v) > save_max_units:  # subsample units for storage
                    idx = np.linspace(0, len(v) - 1, save_max_units).astype(int)
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
    return out_dir


def main():
    ap = argparse.ArgumentParser(
        prog="weight-whisperer",
        description="Read a model's hidden training-data recipe off its weights: "
                    "an agent fires targeted probes while hooks record per-unit "
                    "activations across every module, then synthesizes recipe.json/report.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="What to probe: any HuggingFace hub id (e.g. Qwen/Qwen2.5-0.5B-Instruct, "
                         "gpt2), a local folder with saved weights, or anything "
                         "AutoModelForCausalLM.from_pretrained accepts. Default: %(default)s")
    ap.add_argument("--out", default="recipe_out",
                    help="Output directory (gets recipe_report.md, recipe.json, activations.npz). "
                         "Default: %(default)s")
    ap.add_argument("--rounds", type=int, default=2,
                    help="Adaptive probing rounds after the seed bank. Default: %(default)s")
    ap.add_argument("--probes-per-category", type=int, default=3,
                    help="Seed probes fired per category in round 0. Default: %(default)s")
    ap.add_argument("--probes-per-round", type=int, default=30,
                    help="New agent-designed probes per adaptive round. Default: %(default)s")
    ap.add_argument("--max-length", type=int, default=256,
                    help="Max tokenized probe length. Default: %(default)s")
    ap.add_argument("--seed", type=int, default=42, help="Random seed. Default: %(default)s")
    # --- connect an LLM-backed probe agent (optional) ---
    ap.add_argument("--llm-endpoint", default=None,
                    help="OpenAI-compatible chat endpoint for the probe-design agent, e.g. "
                         "https://api.openai.com/v1/chat/completions, a NVIDIA NIM endpoint, or a "
                         "local server (vLLM/Ollama: http://localhost:8000/v1/chat/completions). "
                         "Omit to use the built-in offline TemplateAgent (no key, no network).")
    ap.add_argument("--llm-key", default=None, help="API key for --llm-endpoint (Bearer token).")
    ap.add_argument("--llm-model", default="gpt-4o-mini",
                    help="Chat model name sent to --llm-endpoint. Default: %(default)s")
    # --- scaling options for bigger models ---
    ap.add_argument("--device", default="cpu", help="cpu or cuda. Default: %(default)s")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="Compute dtype. Default: %(default)s")
    ap.add_argument("--layer-sample", type=int, default=1,
                    help="Hook only every k-th transformer layer (1 = all). Use 4-8 for 7B+ models. "
                         "Default: %(default)s")
    ap.add_argument("--roles", default=None,
                    help="Comma-separated module-role filter, e.g. 'mlp.down,attn.o,lm_head' "
                         "(default: hook everything).")
    ap.add_argument("--save-max-units", type=int, default=4096,
                    help="Cap units stored per module in activations.npz. Default: %(default)s")
    args = ap.parse_args()

    probe_model(args.model, rounds=args.rounds, probes_per_category=args.probes_per_category,
                probes_per_round=args.probes_per_round, max_length=args.max_length,
                out=args.out, device=args.device, dtype=args.dtype,
                layer_sample=args.layer_sample, roles=args.roles,
                save_max_units=args.save_max_units, seed=args.seed,
                llm_endpoint=args.llm_endpoint, llm_key=args.llm_key, llm_model=args.llm_model)


if __name__ == "__main__":
    main()
