"""Step 4: encode title + abstract with frozen pretrained encoders (no weights are trained).

Also stores a title-only and an abstract-only embedding for the title-abstract coherence feature.
Usage: python 04_embed.py minilm|specter2
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROC

NAMES = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "specter2": "allenai/specter2_base"}
which = sys.argv[1]
if which == "specter2":  # plain BERT checkpoint: CLS pooling as in the SPECTER papers
    word = models.Transformer(NAMES[which], max_seq_length=320)
    model = SentenceTransformer(modules=[word, models.Pooling(word.get_word_embedding_dimension(), "cls")])
else:
    model = SentenceTransformer(NAMES[which])
    model.max_seq_length = 320
model = model.to("cuda").half()

docs = pd.read_pickle(PROC / "docs.pkl")
sep = model.tokenizer.sep_token or ". "
full = (docs.title.fillna("") + f" {sep} " + docs.abstract.fillna("")).tolist()

t0 = time.time()
emb = model.encode(full, batch_size=128, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
sec = time.time() - t0
title = model.encode(docs.title.fillna("").tolist(), batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
abst = model.encode(docs.abstract.fillna("").tolist(), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)

np.save(PROC / f"emb_{which}.npy", emb.astype(np.float32))
np.save(PROC / f"emb_{which}_title.npy", title.astype(np.float32))
np.save(PROC / f"emb_{which}_abstract.npy", abst.astype(np.float32))
(PROC / f"emb_{which}_ms_per_doc.txt").write_text(f"{1000 * sec / len(full):.4f}")
print(which, emb.shape, f"{1000 * sec / len(full):.3f} ms/doc on", torch.cuda.get_device_name(0))
