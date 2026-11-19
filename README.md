# dataset_recipe_inferrer

> ⭐ **If this was helpful, please leave a star on the repo** — it helps others discover it: https://github.com/danielosinor123-gif/weight-whisperer

Agent-driven weight probing: fire targeted inputs at a model, capture **per-unit
activations across every hooked module**, and synthesize a hypothesis about what
the model's **training dataset recipe** looked like.

## How it works

1. **Hook everything** (`capture.py`) — forward hooks on every attention
   projection (q/k/v/o), every MLP projection (gate/up/down or fc1/fc2), layer
   norms, embeddings, and lm_head. Each probe produces a per-unit firing vector
   for every module — this is "touching all the individual weights".
2. **Probe bank** (`probes.py`) — ~24 categories × templated inputs, each tagged
   with a dataset hypothesis: chat/instruction format, system prompts, code
   (Python/JS/SQL), math + chain-of-thought, JSON/structured output, Markdown,
   function calling, multilingual, safety refusals, summarization, QA, creative
   writing, roleplay, email, classification, few-shot, long documents, debate,
   dialogue, news, Wikipedia style.
3. **Agent loop** (`agent.py`) — after each round, scores categories by neuron
   selectivity, then generates new probes biased toward weakly-covered categories
   (template mutation offline; optional LLM backend if an endpoint is configured).
4. **Neuron targeting** (`targeting.py`) — per-unit top-activating probes,
   per-category selectivity (between/within variance + selective-unit fraction),
   per-module coverage.
5. **Token signals** (`logit_analysis.py`) — detects chat-template special tokens
   in the vocab (ChatML, [INST], Gemma turns…), code/markdown/reasoning tokens,
   and top predicted tokens per category.
6. **Synthesis** (`synthesizer.py`) — combines everything into `recipe.json` +
   `recipe_report.md`: task-mixture evidence ranked by selectivity, early/mid/late
   layer specialization per category, detected data format, behavioral samples.

## Run

```powershell
$env:PYTHONUTF8='1'
# small instruct model (downloads ~1GB on first run)
python -m dataset_recipe_inferrer.run --model Qwen/Qwen2.5-0.5B-Instruct --rounds 2 --out out_qwen

# quick smoke test on a tiny random model
python -m dataset_recipe_inferrer.run --model hf-internal-testing/tiny-random-gpt2 --rounds 1 --probes-per-category 1 --probes-per-round 5 --out out_smoke

# bigger models (example: 7B on GPU; on CPU use layer sampling + bf16)
python -m dataset_recipe_inferrer.run --model Qwen/Qwen2.5-7B-Instruct --device cuda --dtype bfloat16 \
    --layer-sample 4 --roles mlp.down,attn.o,lm_head --rounds 2 --out out_qwen7b
```

Outputs in `--out`:
- `recipe_report.md` — human-readable dataset-recipe hypothesis
- `recipe.json` — full structured results
- `activations.npz` + `activation_meta.json` — raw per-unit activation matrices
  (probes × units per module) for further analysis (clustering, SAE training, etc.)

## Reading the results

The report has a **Training method signatures** table that directly answers
"what training methods are inside this model":

| Signature | Evidence used |
|---|---|
| `reasoning_step_by_step` | CoT/proof/debug probe selectivity + count of step markers in generated solutions |
| `instruction_following` | strict/multi-constraint instruction probes + behavioral compliance rate |
| `safety_alignment` | harmful-request probes + measured refusal rate |
| `agent_environment_interaction` | ReAct/tool-loop/Gymnasium/text-game probes + ReAct continuation rate |
| `preference_optimization` | pairwise preference (A vs B) probe selectivity |

- **High selectivity + many selective units** for a category → strong evidence the
  model saw lots of that data type.
- **Early-layer firing** → surface/lexical features (language, syntax, format tokens).
- **Late-layer firing** → task/format semantics (instruction following, structure).
- **Chat template tokens in vocab** → direct evidence of the SFT/RLHF data format.
- Random-weight models (smoke test) show no meaningful selectivity — a built-in
  negative control.

## Distillation loop (`distill.py`)

Complete the loop — teacher recipe becomes a student's actual training data:

```powershell
python -m dataset_recipe_inferrer.distill all `
    --teacher Qwen/Qwen2.5-0.5B-Instruct --student Qwen/Qwen2.5-0.5B `
    --recipe out_qwen05b/recipe.json --out distill_out --samples 256 --epochs 1
```

- **gen**: teacher generates synthetic chat samples, category-weighted by the
  recipe's selectivity, rendered with the teacher's own chat template
- **train**: fine-tunes the student on the synthetic JSONL (causal LM loss)
- **verify**: re-probes the student and compares training-method signatures vs the
  teacher → `distillation_report.md` with a signature match rate

## Limitations

- Activation selectivity is *correlational evidence*, not proof of dataset contents.
- Results depend on probe coverage; increase `--rounds` / `--probes-per-category`
  for stronger signal. Real models need more probes than the defaults.
- CPU-friendly but slow for large models; use ≤1B models locally.
