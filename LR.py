"""Condition B: in-language upper bound with logistic regression on frozen embeddings.

For each (embedding model, locale):
  1. Fit LR on that locale's train embeddings for every grid combination.
  2. Score each on that locale's validation split; keep the best by macro-F1.
  3. Evaluate the winning classifier on the chosen test set(s) (default: own language).

Usage:
  python LR.py                                   # all models, all locales, own-language test
  python LR.py --models gemma --locales ja-JP
  python LR.py --test-locales en-US es-ES ja-JP  # also score every winner cross-lingually
"""
import argparse
import itertools
import warnings

from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from embedding import LOCALES, MODELS, load_embeddings

GRID = {
    "C": [0.01, 0.1, 1, 10],
    "class_weight": [None, "balanced"],
}


def score(clf, X, y):
    pred = clf.predict(X)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def fit_one(X_train, y_train, X_val, y_val, C, class_weight, max_iter=2000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter)
        clf.fit(X_train, y_train)
    acc, f1 = score(clf, X_val, y_val)
    return {"C": C, "class_weight": class_weight, "val_acc": acc, "val_f1": f1, "clf": clf}


def select_model(model_name, locale, grid=GRID, n_jobs=-1, verbose=True):
    """Grid-search on train, pick the winner by validation macro-F1. Returns (best, all_runs)."""
    X_tr, y_tr, _ = load_embeddings(model_name, locale, "train")
    X_va, y_va, _ = load_embeddings(model_name, locale, "validation")

    combos = list(itertools.product(grid["C"], grid["class_weight"]))
    runs = Parallel(n_jobs=n_jobs)(
        delayed(fit_one)(X_tr, y_tr, X_va, y_va, C, cw) for C, cw in combos
    )

    if verbose:
        for r in runs:
            print(f"  C={r['C']:<5} class_weight={str(r['class_weight']):<9}"
                  f"  val acc {r['val_acc']:.3f} | val macro-F1 {r['val_f1']:.3f}")

    best = max(runs, key=lambda r: r["val_f1"])
    return best, runs


def evaluate(clf, model_name, test_locales):
    """Score a fitted classifier on each locale's test embeddings."""
    out = {}
    for loc in test_locales:
        X_te, y_te, _ = load_embeddings(model_name, loc, "test")
        out[loc] = score(clf, X_te, y_te)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--locales", nargs="+", default=LOCALES, choices=LOCALES,
                    help="locales to train + validate on")
    ap.add_argument("--test-locales", nargs="+", default=None, choices=LOCALES,
                    help="test sets to score each winner on (default: its own locale)")
    ap.add_argument("--out", default="results_lr.txt")
    args = ap.parse_args()

    lines = []
    for model_name in args.models:
        for loc in args.locales:
            print(f"\n=== {model_name} | train/val {loc} ===")
            best, _ = select_model(model_name, loc)
            print(f"  -> best: C={best['C']}, class_weight={best['class_weight']} "
                  f"(val macro-F1 {best['val_f1']:.3f})")

            test_locs = args.test_locales or [loc]
            for test_loc, (acc, f1) in evaluate(best["clf"], model_name, test_locs).items():
                line = (f"{model_name:<6} {loc} -> {test_loc} test: accuracy {acc:.3f} | "
                        f"macro-F1 {f1:.3f}  [C={best['C']}, class_weight={best['class_weight']}, "
                        f"val macro-F1 {best['val_f1']:.3f}]")
                print("  " + line)
                lines.append(line)

    print("\n=== Summary ===")
    print("\n".join(lines))
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
