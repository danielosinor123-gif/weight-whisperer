"""Activation capture: hooks every module of a HF causal LM and records per-unit activations.

The capture records, for every hooked module and every probe:
  - mean activation per output unit across the sequence
  - activation at the last token (where next-token prediction lives)
  - overall magnitude stats

Units are the individual weights' output dimensions (neurons in MLP, heads/dims in
attention projections), which is what "touching individual model weights" means here:
each probe produces a full per-unit firing vector for every module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

# module-name patterns -> role label
ROLE_PATTERNS = [
    # modern (Qwen/Llama/Mistral): per-projection
    (re.compile(r"\.self_attn\.q_proj$"), "attn.q"),
    (re.compile(r"\.self_attn\.k_proj$"), "attn.k"),
    (re.compile(r"\.self_attn\.v_proj$"), "attn.v"),
    (re.compile(r"\.self_attn\.o_proj$"), "attn.o"),
    # GPT-2 style fused attention
    (re.compile(r"\.attn\.c_attn$"), "attn.c_attn"),
    (re.compile(r"\.attn\.c_proj$"), "attn.c_proj"),
    (re.compile(r"\.self_attn$"), "attn"),
    (re.compile(r"\.attn$"), "attn"),
    # MLP variants: SwiGLU (gate/up/down), GELU (fc1/fc2), GPT-2 (c_fc/c_proj)
    (re.compile(r"\.mlp\.gate_proj$"), "mlp.gate"),
    (re.compile(r"\.mlp\.up_proj$"), "mlp.up"),
    (re.compile(r"\.mlp\.down_proj$"), "mlp.down"),
    (re.compile(r"\.mlp\.fc1$"), "mlp.fc1"),
    (re.compile(r"\.mlp\.fc2$"), "mlp.fc2"),
    (re.compile(r"\.mlp\.c_fc$"), "mlp.fc1"),
    (re.compile(r"\.mlp\.c_proj$"), "mlp.fc2"),
    (re.compile(r"\.mlp$"), "mlp"),
    (re.compile(r"input_layernorm$"), "ln1"),
    (re.compile(r"post_attention_layernorm$"), "ln2"),
    (re.compile(r"\.ln_1$"), "ln1"),
    (re.compile(r"\.ln_2$"), "ln2"),
    (re.compile(r"embed_tokens$"), "embed"),
    (re.compile(r"\.wte$"), "embed"),
    (re.compile(r"\.wpe$"), "pos_embed"),
    (re.compile(r"lm_head$"), "lm_head"),
    (re.compile(r"model\.norm$"), "final_norm"),
    (re.compile(r"\.ln_f$"), "final_norm"),
]


def role_of(name: str) -> str | None:
    for pat, role in ROLE_PATTERNS:
        if pat.search(name):
            return role
    return None


@dataclass
class ProbeRecord:
    probe_id: int
    category: str
    text: str
    # module_name -> {"mean": per-unit mean over seq, "last": per-unit last-token, "norm": scalar}
    activations: dict = field(default_factory=dict)
    # lm_head top tokens at last position
    top_tokens: list = field(default_factory=list)
    n_tokens: int = 0


class ActivationCapture:
    """Context manager that hooks all interesting modules of a model.

    For very large models, pass layer_sample (hook only every k-th layer) and/or
    role_filter (e.g. only MLP + attention projections) to bound memory/time.
    """

    def __init__(self, model, max_seq_stats: int = 4096,
                 layer_sample: int = 1, role_filter: set | None = None):
        self.model = model
        self.handles = []
        self.record: ProbeRecord | None = None
        self.max_seq_stats = max_seq_stats
        self.layer_sample = max(1, layer_sample)
        self.role_filter = role_filter
        self.hooked_modules: dict = {}

    def _make_hook(self, name: str):
        def hook(module, args, output):
            if self.record is None:
                return
            out = output
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                return
            if out.dim() == 3:  # (B, T, H)
                seq = out[0]
                if seq.shape[0] > self.max_seq_stats:
                    seq = seq[-self.max_seq_stats:]
                mean = seq.mean(dim=0).float().cpu()
                last = seq[-1].float().cpu()
                rec = self.record.activations.setdefault(
                    name, {"mean": None, "last": None, "norm": None}
                )
                rec["mean"] = torch.cat([rec["mean"], mean]) if rec["mean"] is not None else mean
                # keep last-token as running last
                rec["last"] = last
                rec["norm"] = float(seq.float().norm())
            elif out.dim() == 2:  # (B, H) e.g. lm_head
                vec = out[0].float().cpu()
                self.record.activations.setdefault(
                    name, {"mean": vec, "last": vec, "norm": float(vec.norm())}
                )
        return hook

    def __enter__(self):
        for name, module in self.model.named_modules():
            role = role_of(name)
            if role is None:
                continue
            if self.role_filter and role not in self.role_filter:
                continue
            if self.layer_sample > 1:
                digits = [int(p) for p in name.split(".") if p.isdigit()]
                if digits and digits[0] % self.layer_sample != 0:
                    continue
            self.hooked_modules[name] = (module, role)
            self.handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def start_probe(self, probe_id: int, category: str, text: str) -> ProbeRecord:
        self.record = ProbeRecord(probe_id=probe_id, category=category, text=text)
        return self.record

    def finish_probe(self, record: ProbeRecord, tokenizer, logits: torch.Tensor | None, n_tokens: int):
        record.n_tokens = n_tokens
        if logits is not None:
            top = logits[0, -1].float().topk(10)
            record.top_tokens = [
                {"token": tokenizer.convert_ids_to_tokens(int(i)), "logit": float(v)}
                for v, i in zip(top.values, top.indices)
            ]
        self.record = None
        return record
