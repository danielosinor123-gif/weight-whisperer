"""Probe agent: adaptively decides which inputs to fire at the model next.

Loop:
  1. Fire current probes, capture per-unit activations.
  2. Score each category by response strength (activation magnitude, output
     confidence) and coverage (how many units in each layer respond selectively).
  3. Find weakly-covered layers/categories and generate targeted probes for them.

Two backends:
  - TemplateAgent (default, offline): slot mutation + category cross-products.
  - LLMAgent (optional): uses an external LLM API to write new probes; disabled
    automatically if no working endpoint/key is configured.
"""
from __future__ import annotations

import random

from .probes import PROBE_SPECS, _fill


class TemplateAgent:
    """Offline agent: mutates templates and targets weak categories."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.seen_texts: set[str] = set()
        self.round = 0

    def propose(self, category_scores: dict[str, float], n: int = 40) -> list[dict]:
        """Propose n new probes, biased toward low-scoring (under-covered) categories."""
        self.round += 1
        cats = list(PROBE_SPECS.keys())
        # weights: inverse of score, floor so every category stays explorable
        weights = [1.0 / (category_scores.get(c, 0.5) + 0.05) for c in cats]
        total = sum(weights)
        weights = [w / total for w in weights]

        probes = []
        attempts = 0
        while len(probes) < n and attempts < n * 30:
            attempts += 1
            cat = self.rng.choices(cats, weights=weights, k=1)[0]
            spec = PROBE_SPECS[cat]
            template = self.rng.choice(spec["templates"])
            text = _fill(template, spec["slots"], self.rng)
            if not text or text in self.seen_texts:
                continue
            # cross-product mutation: prepend a persona/instruction wrapper
            if self.rng.random() < 0.25:
                wrapper = self.rng.choice([
                    "As an expert, ", "Briefly, ", "In detail, ",
                    "You are a teacher. ", "Without explanation, ",
                ])
                text = wrapper + text[0].lower() + text[1:]
            if text in self.seen_texts:
                continue
            self.seen_texts.add(text)
            probes.append({"text": text, "category": cat, "hypothesis": spec["hypothesis"]})
        return probes


class LLMAgent(TemplateAgent):
    """Optional LLM-backed probe writer; falls back to templates on any failure."""

    def __init__(self, endpoint: str | None = None, api_key: str | None = None,
                 model: str = "gpt-4o-mini", seed: int = 0):
        super().__init__(seed=seed)
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.enabled = bool(endpoint and api_key)

    def propose(self, category_scores: dict[str, float], n: int = 40) -> list[dict]:
        if not self.enabled:
            return super().propose(category_scores, n)
        try:
            import requests
            weak = sorted(category_scores.items(), key=lambda kv: kv[1])[:5]
            prompt = (
                "You are designing probes to discover what data a language model was trained on. "
                f"The weakest responding categories are: {[c for c, _ in weak]}. "
                f"Write {n} short, diverse input prompts (one per line) that would strongly activate "
                "neurons associated with these categories. No numbering, no commentary."
            )
            r = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.9},
                timeout=30,
            )
            lines = [l.strip() for l in r.json()["choices"][0]["message"]["content"].splitlines() if l.strip()]
            probes = [{"text": l, "category": weak[i % len(weak)][0],
                       "hypothesis": "LLM-generated targeted probe"} for i, l in enumerate(lines[:n])]
            if probes:
                return probes
        except Exception:
            self.enabled = False  # permanent fallback for this run
        return super().propose(category_scores, n)
