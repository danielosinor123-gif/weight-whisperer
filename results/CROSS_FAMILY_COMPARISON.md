# Cross-family comparison — training-recipe inference results

Same harness, same 37-category probe bank + adaptive agent, three model families.

| | Qwen2.5-0.5B-Instruct | GPT-2 (124M) | GPT-Neo 125M |
|---|---|---|---|
| Params | 494M | 124M | 125M |
| Probes fired | 152 | 133 | 103 |
| Modules hooked | 267 | 100 | 76 |
| **Detected chat format** | **ChatML** `<|im_start|>/<|im_end|>` | none (`<|endoftext|>` only) | none (`<|endoftext|>` only) |
| reasoning_step_by_step | low sel 0.551 / 0.88 CoT steps | low sel 0.752 / 0.25 | low sel 0.725 / 0.63 |
| instruction_following | **high** sel 0.410 / compliance 0.83 | sel 0.312 / 0.67 (heuristic FP) | sel 0.325 / 0.67 (heuristic FP) |
| safety_alignment | sel 0.663 / refusal 0.0* | sel 0.459 / refusal 0.0 | sel 0.675 / refusal 0.0 |
| agent_environment_interaction | sel 0.872 / react 0.1 | sel 0.688 / react 0.1 | sel 0.824 / react 0.2 |
| preference_optimization | sel 0.550 | sel 0.359 | sel 0.639 |

\* refusal_rate 0 on the 2 sampled harmful probes (greedy, 48 tokens) — see limitations.

## Key findings
1. **Format detection is exact and family-agnostic**: ChatML tokens found only in the
   Qwen instruct checkpoint; GPT-2/GPT-Neo (pre-chat web models) show no chat template.
2. **Instruction-following signature separates instruct from base models** via neuron
   selectivity (0.41 vs 0.31-0.33) and behavioral compliance (0.83 vs heuristic-inflated 0.67).
3. **Safety signature**: all three show refusal_rate 0 on the sampled probes — the refusal
   probes need to be harder/more numerous to fire; noted as limitation.
4. **Layer specialization**: format/instruction categories fire late-layer in all models.
5. **Negative control** (tiny-random-gpt2, out_smoke): no meaningful structure — pipeline
   is not producing artifacts.

## Distillation loop (distill_smoke)
gen (24 synthetic samples from recipe) -> train (12 steps, loss 6.80) -> verify
(signature comparison report generated). 0% match expected: random-weight teacher has
no signatures to transfer. Real teacher->student run queued as follow-up (E4).

## Files
- `out_qwen05b_v2/recipe_report.md`, `out_gpt2/recipe_report.md`, `out_gptneo125m/recipe_report.md`
- `out_smoke/recipe_report.md` (negative control)
- `*/activations.npz` — raw per-unit firing matrices
- `distill_smoke/distillation_report.md`
