from sklearn.feature_extraction.text import TfidfVectorizer

from embedding import LOCALES, load_massive
from LR import GRID, fit_one, score

# Sentences and labels for one split
def get_split(ds, split):
    return list(ds[split]["utt"]), list(ds[split]["intent"])

data = {} #dataset per language
chosen = {} #best (vectorizer, model) per language

for loc in LOCALES:
    ds = load_massive(loc)
    data[loc] = ds
    train_x, train_y = get_split(ds, "train")
    val_x, val_y = get_split(ds, "validation")

    #Build the chunk vocabulary from train, then convert train and val
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
    X_train = vec.fit_transform(train_x)
    X_val = vec.transform(val_x)

    #Train one model per setting and score it on validation
    print(f"\n=== tfidf | {loc} ({X_train.shape[1]} features) ===")
    runs = []
    for C in GRID["C"]:
        for cw in GRID["class_weight"]:
            r = fit_one(X_train, train_y, X_val, val_y, C, cw)
            print(f"  C={C:<5} class_weight={str(cw):<9}"
                  f"  val acc {r['val_acc']:.3f} | val macro-F1 {r['val_f1']:.3f}")
            runs.append(r)

    #Keep the setting with the best validation macro-F1
    best = max(runs, key=lambda r: r["val_f1"])
    chosen[loc] = (vec, best)
    print(f"  -> best: C={best['C']}, class_weight={best['class_weight']}")

#Score each winner on test once; the English winner is also tested on the other languages
lines = []
for loc in LOCALES:
    vec, best = chosen[loc]
    test_locs = LOCALES if loc == "en-US" else [loc]
    for test_loc in test_locs:
        test_x, test_y = get_split(data[test_loc], "test")
        acc, f1 = score(best["clf"], vec.transform(test_x), test_y)
        line = (f"tfidf  {loc} -> {test_loc} test: accuracy {acc:.3f} | macro-F1 {f1:.3f}  "
                f"[C={best['C']}, class_weight={best['class_weight']}, val macro-F1 {best['val_f1']:.3f}]")
        print(line)
        lines.append(line)

#Save the results
with open("results_tfidf_LR.txt", "w") as f:
    f.write("\n".join(lines) + "\n")