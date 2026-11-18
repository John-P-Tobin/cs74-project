"""Alignment: how often a sentence's nearest English neighbor is its own translation (validation split)."""
import re

import numpy as np

from embedding import MODELS, load_embeddings

TARGETS = ["es-ES", "ja-JP"]


# Read "model src -> tgt test: accuracy X ..." lines into {(model, src, tgt): accuracy}
def read_accuracies(path):
    pattern = re.compile(r"^(\w+)\s+(\S+) -> (\S+) test: accuracy ([\d.]+)")
    out = {}
    with open(path) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                model, src, tgt, acc = m.groups()
                out[(model, src, tgt)] = float(acc)
    return out


in_language = read_accuracies("results_lr.txt")              # B: trained and tested on the target
zero_shot = read_accuracies("results_transfer_matrix.txt")   # C: English -> target rows

lines = ["# retrieval = share of target validation sentences whose nearest English sentence is their translation",
         "# B, C = test accuracy (in-language, zero-shot from English); gap = B - C",
         f"{'model':<6} {'target':<6} {'retrieval':>9} {'B':>6} {'C':>6} {'gap':>6}"]
for model_name in MODELS:
    X_en, _, id_en = load_embeddings(model_name, "en-US", "validation")
    X_en = X_en / np.linalg.norm(X_en, axis=1, keepdims=True)

    for target in TARGETS:
        X_tg, _, id_tg = load_embeddings(model_name, target, "validation")
        X_tg = X_tg / np.linalg.norm(X_tg, axis=1, keepdims=True)
        assert set(id_tg) == set(id_en), "validation sets are not parallel"

        # Cosine similarity of every target sentence to every English sentence
        sims = X_tg @ X_en.T
        # Correct if the most similar English sentence is its own translation (same id)
        retrieval = (id_en[sims.argmax(axis=1)] == id_tg).mean()

        b = in_language[(model_name, target, target)]
        c = zero_shot[(model_name, "en-US", target)]
        line = f"{model_name:<6} {target:<6} {retrieval:>9.3f} {b:>6.3f} {c:>6.3f} {b - c:>6.3f}"
        print(line)
        lines.append(line)

with open("results_alignment.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nSaved results_alignment.txt")