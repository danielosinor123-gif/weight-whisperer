# Paper plan — "Reading the Training Recipe off the Weights"

## Title candidates
- Probing-Based Dataset Recipe Inference: Reverse-Engineering How a Model Was Trained
- Dataset Distillation from Weights: Inferring the Hidden Training Recipe via Agent-Driven Activation Probing
- What Was This Model Trained On? Agent-Driven Weight Probing for Training-Recipe Inference

## Core claim
Companies release weights but hide the data recipe. We show the recipe is recoverable
from the weights: an agent fires hypothesis-tagged probes at the model while hooks
record per-unit activations across all modules; aggregation yields (a) task mixture,
(b) detected chat format, (c) training-method signatures (reasoning/CoT,
instruction-following, safety alignment, agentic/environment RL, preference
optimization) each with behavioral confirmation, and (d) layer specialization.

## Method sections
1. Probe bank: 30+ categories, each tagged with a dataset hypothesis
2. Full-network activation capture (attn q/k/v/o, MLP gate/up/down, norms, embed, lm_head)
3. Adaptive agent loop: category selectivity scores -> targeted new probes
   (template agent; LLM-agent pluggable)
4. Signatures: selectivity (between/within variance, selective-unit fraction) +
   behavioral generation metrics (CoT step count, compliance rate, refusal rate,
   ReAct continuation rate, JSON validity)
5. Format detection from vocab special tokens (ChatML / [INST] / Gemma)
6. Distillation loop: recipe -> teacher-generated synthetic chat data -> student
   fine-tune -> re-probe student -> signature match rate

## Experiments (results to fill in)
- E1: negative control — random-weight tiny model (no selectivity expected)
- E2: Qwen2.5-0.5B-Instruct full probe (132 probes, 267 hooked modules) —
  ChatML detection, task mixture, layer specialization  [out_qwen05b_v2]
- E3: training-method signatures on Qwen2.5-0.5B-Instruct (v2 run) —
  expected: high instruction_following, safety_alignment, reasoning
- E4: distillation loop smoke (tiny models) + real run teacher=0.5B-Instruct,
  student=0.5B-base — signature match rate  [distill_qwen]
- E5: distant-family validation — GPT-family models (`gpt2`, `EleutherAI/gpt-neo-125M`)
  probed with the same harness  [out_gpt2, out_gptneo125m] — shows the method is
  architecture-agnostic; expected: no ChatML, web-text-era mixture, no instruction
  signatures (pre-chat-template models), contrast vs Qwen
- E6 (optional, GPU): larger teacher (7B) with --layer-sample

## Baselines / related work to cite
- Dataset inference / membership inference (Shokri et al.; Carlini et al.)
- Model fingerprinting & provenance (Xu et al.; model fingerprinting papers)
- Mechanistic interpretability: activation patching, SAEs (Anthropic; Sharkey)
- Knowledge distillation (Hinton et al.) and dataset distillation (Wang et al.)
- LLM-as-judge probing / behavioral eval (LM Eval Harness, AlpacaEval)
- "Training data extraction" attacks (Carlini et al. 2021)

## Limitations (honest section)
- Correlational evidence, not proof; probe coverage matters
- Selectivity depends on probe bank quality; LLM agent improves coverage
- Compute: full hook feasible <=1B on CPU; larger needs GPU + layer sampling
- Signatures conflate pretraining + SFT + RLHF; temporal attribution impossible

## Repo layout for GitHub
- dataset_recipe_inferrer/ (package: capture, probes, agent, targeting,
  behavior, logit_analysis, synthesizer, run, distill)
- paper/ (this plan + LaTeX draft + figures)
- examples/ (commands)
- results/ (recipe_report.md, distillation_report.md — redacted raw npz optional)

## TODO before push
- [x] v2 probe run results (out_qwen05b_v2)
- [x] distillation run results (distill_qwen): 60% signature match rate
- [x] requirements.txt, LICENSE (MIT), README polish
- [x] git init + remote + push (danielosinor123-gif/dataset-recipe-inferrer)
- [x] paper LaTeX draft (paper/main.tex)
- [ ] compile main.tex, submit to arXiv
