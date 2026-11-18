"""Condition D: few-shot transfer. Train on English + k target examples per intent, test on the target."""
import csv
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from embedding import MODELS, load_embeddings
from LR import select_model, score

TARGETS = ["es-ES", "ja-JP"]
KS = [8, 32, 128]   #target examples per intent
SEEDS = [0, 1, 2]   #three random draws per k
N_JOBS = 8          #models trained at the same time


#Pick up to k random examples of each intent (all of them if an intent has fewer)
def sample_k_per_intent(y, k, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for intent in np.unique(y):
        rows = np.flatnonzero(y == intent)
        idx.extend(rng.choice(rows, size=min(k, len(rows)), replace=False))
    return np.array(idx)


#Train on English + k target examples per intent, score on the target's test set
def fewshot_run(model_name, target, k, seed, C, cw):
    X_en, y_en, _ = load_embeddings(model_name, "en-US", "train")
    X_tg, y_tg, _ = load_embeddings(model_name, target, "train")
    X_te, y_te, _ = load_embeddings(model_name, target, "test")

    idx = sample_k_per_intent(y_tg, k, seed)
    X = np.vstack([X_en, X_tg[idx]])
    y = np.concatenate([y_en, y_tg[idx]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(C=C, class_weight=cw, max_iter=2000).fit(X, y)
    return score(clf, X_te, y_te)


rows = []
lines = ["# English train + k target examples per intent; k=0 is zero-shot. Mean +/- std over draws."]
for model_name in MODELS:
    #Choose C and class weight on English validation only (no target data)
    best, _ = select_model(model_name, "en-US", verbose=False)
    C, cw = best["C"], best["class_weight"]
    print(f"\n=== {model_name} | C={C}, class_weight={cw} (chosen on English validation) ===")

    for target in TARGETS:
        #k = 0: the English-only model on the target (zero-shot)
        X_te, y_te, _ = load_embeddings(model_name, target, "test")
        results = {0: [score(best["clf"], X_te, y_te)]}

        #k > 0: one model per (k, draw), trained in parallel
        jobs = [(k, s) for k in KS for s in SEEDS]
        scores = Parallel(n_jobs=N_JOBS)(
            delayed(fewshot_run)(model_name, target, k, s, C, cw) for k, s in jobs)
        for (k, _), sc in zip(jobs, scores):
            results.setdefault(k, []).append(sc)

        #Average over the draws and record
        for k, sc in results.items():
            accs, f1s = np.array(sc).T
            rows.append([model_name, target, k, accs.mean(), accs.std(), f1s.mean(), f1s.std()])
            line = (f"{model_name:<6} {target} k={k:<3} test: "
                    f"accuracy {accs.mean():.3f} +/- {accs.std():.3f} | "
                    f"macro-F1 {f1s.mean():.3f} +/- {f1s.std():.3f}  [C={C}, class_weight={cw}, {len(sc)} draw(s)]")
            print(line)
            lines.append(line)

#Save a readable summary and a table for plotting
with open("results_fewshot.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
with open("results_fewshot.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "target", "k", "acc_mean", "acc_std", "f1_mean", "f1_std"])
    writer.writerows(rows)
print("\nSaved results_fewshot.txt and results_fewshot.csv")