from datasets import load_dataset

LOCALES = ["en-US", "es-ES", "ja-JP"]

def load_massive(locale):
    return load_dataset("AmazonScience/massive",
                        revision="refs/convert/parquet",
                        data_dir=locale)

data = {}
for loc in LOCALES:
    ds = load_massive(loc)
    data[loc] = {
        "train_x": list(ds["train"]["utt"]),
        "train_y": list(ds["train"]["intent"]),
        "val_x":   list(ds["validation"]["utt"]),
        "val_y":   list(ds["validation"]["intent"]),
    }
    print(loc, len(data[loc]["train_x"]), "train,", len(data[loc]["val_x"]), "val")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

results = {}
for loc in LOCALES:
    d = data[loc]

    # 1. Turn sentences into TF-IDF numbers
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
    X_train = vec.fit_transform(d["train_x"])
    X_val = vec.transform(d["val_x"])

    # 2. Train the classifier on this language's training set
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, d["train_y"])

    # 3. Score it on this language's validation set
    pred = clf.predict(X_val)
    acc = accuracy_score(d["val_y"], pred)
    f1 = f1_score(d["val_y"], pred, average="macro")
    results[loc] = (acc, f1)
    print(f"{loc}: {X_train.shape[1]} features | accuracy {acc:.3f} | macro-F1 {f1:.3f}")



print("\nZero-shot TF-IDF (trained on English only):")
en = data["en-US"]
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
clf = LogisticRegression(max_iter=1000)
clf.fit(vec.fit_transform(en["train_x"]), en["train_y"])

for loc in ["es-ES", "ja-JP"]:
    d = data[loc]
    pred = clf.predict(vec.transform(d["val_x"]))
    acc = accuracy_score(d["val_y"], pred)
    f1 = f1_score(d["val_y"], pred, average="macro")
    print(f"en-US -> {loc}: accuracy {acc:.3f} | macro-F1 {f1:.3f}")