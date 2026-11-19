"""Token-level signals: what the vocabulary and raw logits reveal about the
training data recipe (chat template markers, code tokens, format tokens).
"""
from __future__ import annotations

import torch

# token substrings that signal specific dataset components
SIGNAL_TOKENS = {
    "chat_template": ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|assistant|>",
                      "<|user|>", "<|system|>", "[INST]", "[/INST]", "<<SYS>>",
                      "<start_of_turn>", "<end_of_turn>"],
    "code": ["def", "function", "import", "```python", "```javascript", "SELECT",
             "console.log", "public static", "#include"],
    "markdown": ["```", "##", "**", "|---", "- [", "![", "###"],
    "reasoning": ["step by step", "Step 1", "Therefore", "Let's think", "\\boxed",
                  "Solution:", "Q:", "A:"],
    "structured": ['{"', '"}', "name\":", "arguments", "function_call"],
    "dialogue": ["User:", "Assistant:", "Human:", "AI:", "System:"],
}


def analyze_vocab(tokenizer) -> dict:
    """Check which signal tokens exist as single tokens / how they tokenize."""
    vocab = tokenizer.get_vocab()
    results = {}
    for group, tokens in SIGNAL_TOKENS.items():
        hits = []
        for t in tokens:
            # exact vocab match (raw or byte-level encoded)
            in_vocab = t in vocab or t.replace(" ", "Ġ") in vocab
            n_pieces = len(tokenizer.encode(t, add_special_tokens=False))
            hits.append({"token": t, "in_vocab": in_vocab, "n_pieces": n_pieces})
        results[group] = hits
    return results


def logit_salience(model, tokenizer, prompts: dict[str, list[str]]) -> dict:
    """Run short prompts and record the top predicted tokens per category."""
    out = {}
    model.eval()
    with torch.no_grad():
        for category, texts in prompts.items():
            rows = []
            for text in texts[:3]:
                ids = tokenizer(text, return_tensors="pt")
                logits = model(**ids).logits
                top = logits[0, -1].float().topk(8)
                rows.append({
                    "text": text[:100],
                    "predictions": [
                        {"token": tokenizer.convert_ids_to_tokens(int(i)), "logit": round(float(v), 3)}
                        for v, i in zip(top.values, top.indices)
                    ],
                })
            out[category] = rows
    return out
