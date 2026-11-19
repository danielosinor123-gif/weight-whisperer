"""Bring your own probe agent.

The prober just needs an object with:

    propose(category_scores: dict[str, float], n: int) -> list[dict]

- `category_scores` maps each probe category to its mean neuron selectivity from
  the previous rounds (LOW score = under-covered = worth probing next).
- return up to `n` probes, each a dict: {"text": str, "category": str, "hypothesis": str}.

Subclass TemplateAgent (offline default) or implement the method from scratch —
e.g. call your own LLM, an RL policy, a retrieval system, anything.

Run from the folder that CONTAINS the dataset_recipe_inferrer package:

    python examples/custom_agent.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make the package importable

from dataset_recipe_inferrer.agent import TemplateAgent
from dataset_recipe_inferrer.run import probe_model


class MyAgent(TemplateAgent):
    def propose(self, category_scores, n=30):
        # TODO: your logic here. Example: route the weakest categories to your own LLM:
        # weak = sorted(category_scores, key=category_scores.get)[:5]
        # probes = my_llm.write_probes(weak, n)
        # return [{"text": t, "category": c, "hypothesis": "mine"} for t, c in probes]
        return super().propose(category_scores, n)  # fallback: template mutation


if __name__ == "__main__":
    probe_model(
        "hf-internal-testing/tiny-random-gpt2",  # tiny smoke model, no download cost
        agent=MyAgent(seed=0),
        rounds=1,
        probes_per_category=1,
        probes_per_round=5,
        out="out_custom_agent",
    )
