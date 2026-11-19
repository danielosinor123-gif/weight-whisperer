# weight-whisperer

> ⭐ **If this was helpful, please leave a star on the repo** — it helps others discover it: https://github.com/danielosinor123-gif/weight-whisperer

Read a model's **hidden training-data recipe off its weights**: an agent fires targeted
inputs ("probes") at any HuggingFace model while forward hooks record **per-unit
activations across every module** (attention q/k/v/o, MLP, layer norms, embeddings,
LM head), then synthesizes a structured recipe hypothesis — task mixture, chat format,
training-method signatures — and you can **distill that recipe into a student model**
that reproduces the teacher's profile (100% signature match, 5/5, in our runs).

## Install

```bash
git clone https://github.com/danielosinor123-gif/weight-whisperer.git
cd weight-whisperer
pip install -e .          # gives you the `weight-whisperer` command
# or without installing: pip install -r requirements.txt, then run `python -m ...` from the folder ABOVE the repo
```

## Quickstart

```bash
# 1) smoke test on a tiny random model (seconds, proves the pipeline works; expect no signal — it's the negative control)
weight-whisperer --model hf-internal-testing/tiny-random-gpt2 --rounds 1 --probes-per-category 1 --probes-per-round 5 --out out_smoke

# 2) probe a real model (downloads ~1 GB on first run)
weight-whisperer --model Qwen/Qwen2.5-0.5B-Instruct --rounds 2 --out out_qwen

# 3) read the result
cat out_qwen/recipe_report.md
```

## Point it at any model

`--model` is passed straight to `AutoModelForCausalLM.from_pretrained`, so all of these work:

| You have | `--model` value |
|---|---|
| HF hub model | `Qwen/Qwen2.5-0.5B-Instruct`, `gpt2`, `meta-llama/Llama-3.2-1B`, … |
| Local folder with saved weights | `C:\path\to\my-model` or `./my-model` |
| Private/gated repo | same id + `huggingface-cli login` (or `HF_TOKEN=...`) |

Scaling flags for bigger models:

| Flag | Meaning | Example |
|---|---|---|
| `--device` | `cpu` or `cuda` | `--device cuda` |
| `--dtype` | `float32` / `bfloat16` / `float16` | `--dtype bfloat16` |
| `--layer-sample K` | hook only every K-th layer (K=1 hooks all) | `--layer-sample 4` for 7B+ |
| `--roles` | hook only these module roles | `--roles mlp.down,attn.o,lm_head` |
| `--rounds N` / `--probes-per-round N` | more probing = stronger signal | `--rounds 4 --probes-per-round 40` |
| `--save-max-units` | cap units stored per module in `activations.npz` | `--save-max-units 8192` |

```bash
# 7B on GPU, hooking every 4th layer
weight-whisperer --model Qwen/Qwen2.5-7B-Instruct --device cuda --dtype bfloat16 \
    --layer-sample 4 --roles mlp.down,attn.o,lm_head --rounds 2 --out out_qwen7b
```

## Connect your own agent

The "agent" only decides **which probes to fire next**, based on which categories are
under-covered so far. Three ways to drive it:

**1. Built-in offline agent (default, zero config).** `TemplateAgent` mutates the
probe-bank templates — no API key, no network:

```bash
weight-whisperer --model Qwen/Qwen2.5-0.5B-Instruct --rounds 3 --out out_qwen
```

**2. LLM-backed agent via any OpenAI-compatible endpoint** (OpenAI, NVIDIA NIM, or a
local vLLM/Ollama server). The CLI passes `--llm-endpoint/--llm-key/--llm-model` to an
`LLMAgent` that asks the LLM to write targeted probes; it falls back to the offline
agent automatically if the endpoint is unreachable:

```bash
# cloud endpoint
weight-whisperer --model Qwen/Qwen2.5-0.5B-Instruct --rounds 3 --out out_qwen \
    --llm-endpoint https://api.openai.com/v1/chat/completions \
    --llm-key $OPENAI_API_KEY --llm-model gpt-4o-mini

# local server (vLLM / Ollama / LM Studio)
weight-whisperer --model Qwen/Qwen2.5-0.5B-Instruct --rounds 3 --out out_qwen \
    --llm-endpoint http://localhost:8000/v1/chat/completions \
    --llm-key dummy --llm-model qwen2.5-7b-instruct
```

**3. Your own agent from Python.** Any object with
`propose(category_scores: dict[str, float], n: int) -> list[dict]` plugs straight into
the pipeline — `category_scores` maps probe category → mean neuron selectivity from the
previous rounds (low = under-covered = probe it), and each returned probe is
`{"text": ..., "category": ..., "hypothesis": ...}`. See
[`examples/custom_agent.py`](examples/custom_agent.py) and
[`examples/probe_from_python.py`](examples/probe_from_python.py):

```python
from dataset_recipe_inferrer.run import probe_model
from dataset_recipe_inferrer.agent import TemplateAgent

class MyAgent(TemplateAgent):          # or implement propose() from scratch
    def propose(self, category_scores, n=30):
        weak = sorted(category_scores, key=category_scores.get)[:5]
        # ... call your own LLM / policy / retrieval here ...
        return super().propose(category_scores, n)

probe_model("Qwen/Qwen2.5-0.5B-Instruct", agent=MyAgent(), rounds=3, out="out_qwen")
```

## What you get

Everything lands in `--out`:

- `recipe_report.md` — human-readable recipe hypothesis (open this first)
- `recipe.json` — full structured results (task mixture, signatures, format, layers)
- `activations.npz` + `activation_meta.json` — raw per-unit activation matrices
  (probes × units per module) for your own analysis (clustering, SAE training, …)

## Reading the results

The report's **Training method signatures** table directly answers "what training
methods are inside this model":

| Signature | Evidence used |
|---|---|
| `reasoning_step_by_step` | CoT/proof/debug probe selectivity + step-marker count in generated solutions |
| `instruction_following` | strict/multi-constraint instruction probes + behavioral compliance rate |
| `safety_alignment` | harmful-request probes + measured refusal rate |
| `agent_environment_interaction` | ReAct/tool-loop/Gymnasium/text-game probes + ReAct continuation rate |
| `preference_optimization` | pairwise preference (A vs B) probe selectivity |

- **High selectivity + many selective units** for a category → strong evidence the
  model saw lots of that data type.
- **Early-layer firing** → surface/lexical features (language, syntax, format tokens).
- **Late-layer firing** → task/format semantics (instruction following, structure).
- **Chat template tokens in the vocab** (ChatML, [INST], Gemma turns…) → direct
  evidence of the SFT/RLHF data format.
- Random-weight models show no meaningful selectivity — a built-in negative control.

## Distillation loop: prove the recipe is real

Turn the inferred recipe into a student model's actual training data, then re-probe
the student and compare signatures to the teacher:

```bash
weight-whisperer-distill all \
    --teacher Qwen/Qwen2.5-0.5B-Instruct --student Qwen/Qwen2.5-0.5B \
    --recipe out_qwen/recipe.json --out distill_out --samples 256 --epochs 1
# stages can also run separately: gen | train | verify
```

- **gen**: teacher generates synthetic chat samples, category-weighted by the recipe's
  selectivity, rendered with the teacher's own chat template
- **train**: fine-tunes the student on the synthetic JSONL
- **verify**: re-probes the student → `distillation_report.md` with a **signature match
  rate**. Our run: **100% (5/5)** — see [`results/CROSS_FAMILY_COMPARISON.md`](results/CROSS_FAMILY_COMPARISON.md).

## Results so far

- Qwen2.5-0.5B-Instruct: ChatML format detected exactly; top mixture = agent tool loops,
  few-shot, classification; instruction-following signature separates it from base models
- GPT-2 / GPT-Neo-125M: no chat template, low instruction selectivity — consistent with
  pre-chat web pretraining
- Distillation: teacher → recipe → 125 synthetic samples → base student → re-probe =
  **100% signature match (5/5)**; random-weight negative controls give 0%
- Full details: [`results/CROSS_FAMILY_COMPARISON.md`](results/CROSS_FAMILY_COMPARISON.md),
  per-model reports in `out_*/recipe_report.md`

## Paper

Draft in [`paper/`](paper) (`main.tex`, compiled `main.pdf`, arXiv-ready source tarball
`weight-whisperer-arxiv.tar.gz`). Build with any LaTeX: `tectonic paper/main.tex`.

## How it works

1. **Hook everything** (`capture.py`) — forward hooks on every attention projection
   (q/k/v/o), every MLP projection, layer norms, embeddings, lm_head. Each probe
   produces a per-unit firing vector for every module — "touching all the individual
   weights".
2. **Probe bank** (`probes.py`) — 37 categories × templated inputs, each tagged with a
   dataset hypothesis: chat/instructions, system prompts, code (Python/JS/SQL), math +
   CoT, JSON/structured output, Markdown, function calling, ReAct agents, tool loops,
   Gymnasium envs, reward feedback, preference pairs, safety refusals, summarization,
   QA, creative writing, dialogue, news, Wikipedia style, and more.
3. **Agent loop** (`agent.py`) — after each round, scores categories by neuron
   selectivity, then proposes new probes biased toward weakly-covered categories
   (offline templates by default; optional LLM backend — see above).
4. **Neuron targeting** (`targeting.py`) — per-unit top-activating probes, per-category
   selectivity (between/within variance + selective-unit fraction), per-module coverage.
5. **Token signals** (`logit_analysis.py`) — chat-template special tokens in the vocab,
   code/markdown/reasoning tokens, top predicted tokens per category.
6. **Synthesis** (`synthesizer.py`) — fuses everything into `recipe.json` +
   `recipe_report.md`: task-mixture evidence ranked by selectivity, early/mid/late layer
   specialization, detected data format, behavioral samples, confidence-gated
   training-method signatures.

## Limitations

- Activation selectivity is *correlational evidence*, not proof of dataset contents.
- Results depend on probe coverage; increase `--rounds` / `--probes-per-category` for
  stronger signal.
- Behavioral heuristics (e.g. compliance) can false-positive on base models, so
  confidence is evidence-gated (capped at "low" with <4 probes per signature).
- CPU-friendly up to ~1B params with full hooking; bigger models need `--device cuda`
  plus `--layer-sample`/`--roles`.
