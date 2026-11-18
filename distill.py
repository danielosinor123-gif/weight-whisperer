"""distill.py — complete the distillation loop:

  1. gen    : read a probed recipe.json, use the TEACHER model to generate a synthetic
              chat dataset that matches the recipe's task mixture + detected format.
  2. train  : fine-tune a small STUDENT model on that synthetic data (causal LM loss).
  3. verify : run the same prober on the student and compare training-method signatures
              against the teacher -> distillation_report.md

Usage:
  python -m dataset_recipe_inferrer.distill all \
      --teacher Qwen/Qwen2.5-0.5B-Instruct \
      --student Qwen/Qwen2.5-0.5B \
      --recipe out_qwen05b/recipe.json \
      --out distill_out --samples 256 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .agent import TemplateAgent
from .behavior import run_behavioral_suite
from .capture import ActivationCapture
from .logit_analysis import analyze_vocab, logit_salience
from .probes import PROBE_SPECS, build_probes
from .run import fire_probes, quick_category_scores
from .synthesizer import synthesize


# ---------------------------------------------------------------------------
# Stage 1: recipe -> synthetic data (teacher-generated, recipe-weighted)
# ---------------------------------------------------------------------------

def gen_synthetic_data(teacher, tokenizer, recipe: dict, n_samples: int,
                       device: str, max_new_tokens: int = 96, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    mixture = recipe.get("task_mixture_evidence", {})
    # weight categories by selectivity; always include signature categories
    cats = list(mixture.keys()) or list(PROBE_SPECS.keys())
    weights = [max(mixture.get(c, {}).get("selectivity", 0.1), 0.02) for c in cats]

    bank = build_probes(per_category=4, seed=seed)
    by_cat = {}
    for p in bank:
        by_cat.setdefault(p["category"], []).append(p["text"])

    samples = []
    teacher.eval()
    for i in range(n_samples):
        cat = rng.choices(cats, weights=weights, k=1)[0]
        prompt = rng.choice(by_cat.get(cat, ["Explain a topic."]))
        ids = tokenizer(prompt, return_tensors="pt")
        ids = {k: v.to(device) for k, v in ids.items()}
        with torch.no_grad():
            out = teacher.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                                   temperature=0.8, top_p=0.95,
                                   pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        if len(completion.strip()) < 8:
            continue
        samples.append({"category": cat, "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion.strip()},
        ]})
    return samples


def render_chat(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=add_generation_prompt)
    except Exception:
        # no chat template (e.g. base GPT-2): plain prompt/response format
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        return text + ("assistant: " if add_generation_prompt else "")


# ---------------------------------------------------------------------------
# Stage 2: train student on synthetic data
# ---------------------------------------------------------------------------

class ChatDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length: int):
        self.enc = []
        for t in texts:
            ids = tokenizer(t, truncation=True, max_length=max_length).input_ids
            if len(ids) > 8:
                self.enc.append(ids)

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        ids = self.enc[i]
        return {"input_ids": ids, "labels": ids.copy()}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def train_student(student_name: str, data_path: Path, out_dir: Path, device: str,
                  dtype, epochs: int, batch_size: int, lr: float, max_length: int,
                  seed: int = 0) -> str:
    tokenizer = AutoTokenizer.from_pretrained(student_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(student_name, dtype=dtype).to(device)
    model.train()

    samples = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # synthetic_data.jsonl already contains rendered "text" (chat template applied at gen time)
    texts = [s["text"] if "text" in s else render_chat(tokenizer, s["messages"]) for s in samples]
    ds = ChatDataset(texts, tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    torch.manual_seed(seed)
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"[train] epoch {ep} step {step} loss {loss.item():.3f}", flush=True)
    model.save_pretrained(out_dir / "student")
    tokenizer.save_pretrained(out_dir / "student")
    print(f"[train] done: {step} steps, final loss {loss.item():.3f}, {time.time()-t0:.0f}s", flush=True)
    return str(out_dir / "student")


# ---------------------------------------------------------------------------
# Stage 3: probe the student, compare signatures with the teacher recipe
# ---------------------------------------------------------------------------

def probe_model(model_name: str, out_dir: Path, device: str, dtype, rounds: int,
                probes_per_category: int, probes_per_round: int, seed: int,
                layer_sample: int = 1, roles: str | None = None) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    agent = TemplateAgent(seed=seed)
    role_filter = set(roles.split(",")) if roles else None
    with ActivationCapture(model, layer_sample=layer_sample, role_filter=role_filter) as cap:
        all_records, next_id = [], 0
        probes = build_probes(per_category=probes_per_category, seed=seed)
        recs = fire_probes(model, tokenizer, cap, probes, next_id, device=device)
        next_id += len(probes)
        all_records.extend(recs)
        for r in range(1, rounds + 1):
            scores = quick_category_scores(all_records, cap.hooked_modules)
            probes = agent.propose(scores, n=probes_per_round)
            if not probes:
                break
            recs = fire_probes(model, tokenizer, cap, probes, next_id, device=device)
            next_id += len(probes)
            all_records.extend(recs)
        vocab_signals = analyze_vocab(tokenizer)
        sal_prompts = {c: [p["text"] for p in build_probes(2, seed) if p["category"] == c]
                       for c in list(PROBE_SPECS)[:8]}
        salience = logit_salience(model, tokenizer, sal_prompts)
        all_probes = [{"text": r.text, "category": r.category} for r in all_records]
        behavior = run_behavioral_suite(model, tokenizer, all_probes, device=device)
        recipe = synthesize(all_records, cap.hooked_modules, vocab_signals, salience,
                            model_name, out_dir, behavior=behavior)
    return recipe


def compare_signatures(teacher_recipe: dict, student_recipe: dict) -> dict:
    t = teacher_recipe.get("training_method_signatures", {})
    s = student_recipe.get("training_method_signatures", {})
    rows = []
    for name in sorted(set(t) | set(s)):
        tv, sv = t.get(name), s.get(name)
        rows.append({
            "signature": name,
            "teacher_confidence": tv["confidence"] if tv else "-",
            "student_confidence": sv["confidence"] if sv else "-",
            "teacher_selectivity": tv["mean_category_selectivity"] if tv else None,
            "student_selectivity": sv["mean_category_selectivity"] if sv else None,
            "teacher_behavior": tv["behavior_value"] if tv else None,
            "student_behavior": sv["behavior_value"] if sv else None,
        })
    conf_rank = {"low": 0, "medium": 1, "high": 2, "-": -1}
    matches = sum(1 for r in rows
                  if r["teacher_confidence"] != "-" and r["student_confidence"] == r["teacher_confidence"])
    return {"rows": rows, "signature_match_rate": matches / max(1, len(rows)),
            "n_signatures": len(rows)}


def render_distill_report(teacher_name, student_name, comparison, out_path: Path):
    lines = [f"# Distillation report — teacher vs student",
             "", f"- Teacher: `{teacher_name}` (probed recipe)",
             f"- Student: `{student_name}` (trained on synthetic recipe data, re-probed)",
             "",
             f"**Signature match rate: {comparison['signature_match_rate']:.0%}** "
             f"({comparison['n_signatures']} training-method signatures compared)",
             "",
             "| Signature | Teacher conf | Student conf | Teacher sel. | Student sel. | Teacher beh. | Student beh. |",
             "|---|---|---|---|---|---|---|"]
    for r in comparison["rows"]:
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else (v or "-")
        lines.append(f"| {r['signature']} | {r['teacher_confidence']} | {r['student_confidence']} | "
                     f"{fmt(r['teacher_selectivity'])} | {fmt(r['student_selectivity'])} | "
                     f"{fmt(r['teacher_behavior'])} | {fmt(r['student_behavior'])} |")
    lines += ["", "Higher match rate = the student absorbed the teacher's training-method profile."]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="recipe -> synthetic data -> student -> verify")
    ap.add_argument("stage", choices=["gen", "train", "verify", "all"])
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--recipe", default="out_qwen05b/recipe.json")
    ap.add_argument("--out", default="distill_out")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--probes-per-category", type=int, default=2)
    ap.add_argument("--probes-per-round", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    data_path = out_dir / "synthetic_data.jsonl"

    if args.stage in ("gen", "all"):
        print(f"[gen] teacher={args.teacher} recipe={args.recipe} samples={args.samples}")
        recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
        tok = AutoTokenizer.from_pretrained(args.teacher)
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype).to(args.device)
        samples = gen_synthetic_data(teacher, tok, recipe, args.samples, args.device,
                                     args.max_new_tokens, args.seed)
        # render with the teacher's own chat template -> format matches the recipe
        with data_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps({"text": render_chat(tok, s["messages"]), "category": s["category"]}) + "\n")
        (out_dir / "synthetic_raw.json").write_text(json.dumps(samples, indent=1), encoding="utf-8")
        print(f"[gen] wrote {len(samples)} samples -> {data_path}")
        del teacher

    if args.stage in ("train", "all"):
        print(f"[train] student={args.student} data={data_path}")
        student_path = train_student(args.student, data_path, out_dir, args.device, dtype,
                                     args.epochs, args.batch_size, args.lr, args.max_length, args.seed)
        print(f"[train] student saved -> {student_path}")

    if args.stage in ("verify", "all"):
        student_name = str(out_dir / "student") if args.stage == "all" else args.student
        print(f"[verify] probing student {student_name}")
        student_recipe = probe_model(student_name, out_dir / "student_probe", args.device, dtype,
                                     args.rounds, args.probes_per_category, args.probes_per_round,
                                     args.seed)
        teacher_recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
        comparison = compare_signatures(teacher_recipe, student_recipe)
        render_distill_report(args.teacher, student_name, comparison, out_dir / "distillation_report.md")
        (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"[verify] signature match rate: {comparison['signature_match_rate']:.0%}")
        print(f"[verify] report: {out_dir/'distillation_report.md'}")


if __name__ == "__main__":
    main()
