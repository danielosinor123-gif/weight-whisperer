"""Neuron targeting: for each unit of each hooked module, find the probes that
activate it most, and measure per-category selectivity.

This is the 'touch all individual weights' step: every output unit of every
hooked Linear/LayerNorm module gets a firing profile across the probe bank.
"""
from __future__ import annotations

import numpy as np
import torch


def build_activation_matrix(records, module: str, key: str = "last"):
    """(n_probes, n_units) matrix for one module. Units with no data are dropped."""
    rows, cats = [], []
    for rec in records:
        act = rec.activations.get(module)
        if act is None or act.get(key) is None:
            continue
        v = act[key]
        if isinstance(v, torch.Tensor):
            v = v.numpy()
        rows.append(np.asarray(v, dtype=np.float32))
        cats.append(rec.category)
    if not rows:
        return None, []
    width = min(len(r) for r in rows)
    return np.stack([r[:width] for r in rows]), cats


def top_probes_per_unit(matrix: np.ndarray, records, cats, top_k: int = 3, max_units: int = 512):
    """For a sample of units, return the probes that fire them hardest."""
    n_probes, n_units = matrix.shape
    unit_idx = np.linspace(0, n_units - 1, min(max_units, n_units)).astype(int)
    results = []
    for u in unit_idx:
        col = matrix[:, u]
        order = np.argsort(-col)[:top_k]
        results.append({
            "unit": int(u),
            "top_probes": [
                {"probe_id": records[i].probe_id, "category": cats[i],
                 "text": records[i].text[:120], "activation": float(col[i])}
                for i in order
            ],
            "mean": float(col.mean()),
            "std": float(col.std()),
        })
    return results


def category_selectivity(matrix: np.ndarray, cats) -> dict:
    """Per-category: mean |activation|, and a selectivity score = between-category
    variance / total variance across units (high => specialized units exist)."""
    cats = np.array(cats)
    out = {}
    overall_var = matrix.var(axis=0).mean() + 1e-12
    for c in sorted(set(cats)):
        sub = matrix[cats == c]
        rest = matrix[cats != c]
        if len(sub) == 0 or len(rest) == 0:
            continue
        diff = sub.mean(axis=0) - rest.mean(axis=0)
        # fraction of units that respond notably more to this category
        thresh = 2 * matrix.std(axis=0) + 1e-9
        frac_selective = float((np.abs(diff) > thresh).mean())
        out[c] = {
            "mean_abs_activation": float(np.abs(sub).mean()),
            "n_probes": int(len(sub)),
            "frac_selective_units": frac_selective,
            "selectivity": float((diff ** 2).mean() / overall_var),
        }
    return out


def layer_coverage(records, modules: list[str]) -> dict:
    """Per-module: how much of its unit population any probe manages to fire."""
    coverage = {}
    for m in modules:
        matrix, cats = build_activation_matrix(records, m)
        if matrix is None:
            continue
        std = matrix.std(axis=0)
        active = (std > 1e-6).mean()
        coverage[m] = {
            "n_units": int(matrix.shape[1]),
            "frac_units_variable": float(active),
            "mean_norm": float(np.abs(matrix).mean()),
        }
    return coverage
