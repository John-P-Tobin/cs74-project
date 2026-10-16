from datasets import load_dataset

def load_massive(locale):
    return load_dataset("AmazonScience/massive",
                        revision="refs/convert/parquet",
                        data_dir=locale)

ds = load_massive("en-US")
print(ds)


train = ds["train"].to_pandas()

intent_names = ds["train"].features["intent"].names
train["intent_name"] = train["intent"].map(lambda i: intent_names[i])

print(train[["utt", "intent_name"]].head(3))

counts = train["intent_name"].value_counts()
print("Number of intents:", len(counts))
print("Most common:\n", counts.head(5))
print("Least common:\n", counts.tail(5))
print("Imbalance ratio (max/min):", round(counts.max() / counts.min(), 1))
print("Median per intent:", counts.median())

print("Missing values per column:\n", train.isna().sum())
print("Duplicate utterances:", train["utt"].duplicated().sum())

dupes = train[train["utt"].duplicated(keep=False)]
conflicts = dupes.groupby("utt")["intent_name"].nunique()
print("Duplicated texts with conflicting labels:", (conflicts > 1).sum())
print(dupes[dupes["utt"].isin(conflicts[conflicts > 1].index)][["utt", "intent_name"]].head(6))

lengths = train["utt"].str.split().str.len()
print("Words per utterance:\n", lengths.describe())


example_id = train["id"].iloc[0]
print("en-US:", train.loc[train["id"] == example_id, "utt"].iloc[0])

for loc in ["es-ES", "ja-JP"]:
    other = load_massive(loc)["train"].to_pandas()
    print(loc + ":", other.loc[other["id"] == example_id, "utt"].iloc[0])


import matplotlib.pyplot as plt

counts.plot.bar(figsize=(14, 4))
plt.title("MASSIVE en-US train: examples per intent")
plt.ylabel("Training examples")
plt.tight_layout()
plt.savefig("intent_counts.png")
print("Saved plot")