from pathlib import Path

import numpy as np
from datasets import load_dataset

LOCALES = ["en-US", "es-ES", "ja-JP"]
SPLITS = ["train", "validation", "test"]
CACHE_DIR = Path("embeddings")

# short name -> (Hugging Face model id, prompt prepended to every utterance)
MODELS = {
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", None),
    "mpnet":  ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", None),
    # LLM-based encoder; expects a task prompt (this is its built-in "Classification" prompt).
    # Gated model: accept the license on Hugging Face and run `hf auth login` first.
    "gemma":  ("google/embeddinggemma-300m", "task: classification | query: "),
}


def load_massive(locale):
    return load_dataset("AmazonScience/massive",
                        revision="refs/convert/parquet",
                        data_dir=locale)


def cache_path(model_name, locale, split):
    return CACHE_DIR / model_name / f"{locale}_{split}.npz"


def embed_all(model_name, batch_size=64):
    from sentence_transformers import SentenceTransformer

    model_id, prompt = MODELS[model_name]
    model = None  # only load the model if something actually needs embedding

    for loc in LOCALES:
        ds = load_massive(loc)
        for split in SPLITS:
            path = cache_path(model_name, loc, split)
            if path.exists():
                print(f"{model_name} {loc} {split}: cached")
                continue
            if model is None:
                model = SentenceTransformer(model_id)

            # 1. Encode utterances (L2-normalized, so dot product = cosine similarity)
            X = model.encode(list(ds[split]["utt"]),
                             prompt=prompt,
                             batch_size=batch_size,
                             normalize_embeddings=True,
                             convert_to_numpy=True,
                             show_progress_bar=True)

            # 2. Save with labels and ids; ids line up across locales (parallel translations)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path,
                     X=X.astype(np.float32),
                     y=np.array(ds[split]["intent"]),
                     id=np.array(ds[split]["id"]))
            print(f"{model_name} {loc} {split}: {X.shape} -> {path}")


def load_embeddings(model_name, locale, split):
    """Returns (X, y, ids) from the cache. Run this file first to build it."""
    d = np.load(cache_path(model_name, locale, split))
    return d["X"], d["y"], d["id"]


if __name__ == "__main__":
    for name in MODELS:
        embed_all(name)
