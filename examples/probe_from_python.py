"""Minimal example: probe any HuggingFace model from Python and get a recipe report.

Run from the folder that CONTAINS the dataset_recipe_inferrer package
(or after `pip install -e .` from anywhere):

    python examples/probe_from_python.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make the package importable

from dataset_recipe_inferrer.run import probe_model

out = probe_model(
    "Qwen/Qwen2.5-0.5B-Instruct",  # any HF hub id, or a local path to saved weights
    rounds=2,                      # adaptive agent rounds after the seed bank
    out="out_qwen",                # results land here
)

print("\nOpen the report:", out / "recipe_report.md")
print("Structured recipe:", out / "recipe.json")
print("Raw per-unit activations:", out / "activations.npz")
