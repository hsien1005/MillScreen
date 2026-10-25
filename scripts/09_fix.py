"""Fix motivated by the error analysis: long abstracts are truncated by the encoder (47% exceed 320 tokens),
and false negatives are longer than true positives.

Each abstract is split at sentence boundaries into chunks of <= CHUNK_TOKENS tokens, each prefixed with
the title, and every chunk is encoded. Two ways to combine, chosen on validation PR-AUC only:
  mean: document embedding = normalised mean of its chunk embeddings, then MillScreen as before
  max : references use the mean embedding; a query's score is the max MillScreen score over its chunks
The test set is scored once with the selected variant; compared with the original MillScreen (paired t-test).
Recall is also compared at equal test flag rates (1 / 2 / 5 %), since the fix changes how many papers pass tau.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROC, RESULTS
from src.methods import load_docs, load_split, metrics, millscreen_score, recall_at_rate, tau_budget

SEEDS, CHUNK_TOKENS = range(5), 200
docs = load_docs()
cfg = json.loads((RESULTS / "t2_config.json").read_text())
K = cfg["k"]
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to("cuda").half()
enc.max_seq_length = 320
tok, sep = enc.tokenizer, enc.tokenizer.sep_token

# ---- chunk and encode (cached) ----
cache = PROC / f"emb_minilm_chunks{CHUNK_TOKENS}.npz"
if cache.exists():
    z = np.load(cache)
    C, owner = z["C"], z["owner"]
else:
    texts, owner = [], []
    for i, (t, a) in enumerate(zip(docs.title.fillna(""), docs.abstract)):
        cur, n = [], 0
        for s in (x for x in re.split(r"(?<=[.!?])\s+", a) if x.strip()):
            m = len(tok.tokenize(s))
            if cur and n + m > CHUNK_TOKENS:
                texts.append(f"{t} {sep} " + " ".join(cur)), owner.append(i)
                cur, n = [], 0
            cur.append(s)
            n += m
        texts.append(f"{t} {sep} " + " ".join(cur)), owner.append(i)
    owner = np.array(owner)
    t0 = time.time()
    C = enc.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True,
                   convert_to_numpy=True).astype(np.float32)
    print(f"{len(texts)} chunks, {len(texts) / len(docs):.2f} per doc, {1000 * (time.time() - t0) / len(docs):.3f} ms/doc")
    np.savez(cache, C=C, owner=owner)
M = np.zeros((len(docs), C.shape[1]), np.float32)
np.add.at(M, owner, C)
M /= np.linalg.norm(M, axis=1, keepdims=True)
n_chunks = np.bincount(owner, minlength=len(docs))
print("docs with >1 chunk:", (n_chunks > 1).mean().round(3))


def score(variant, normal, mill, idx):
    if variant == "mean":
        return millscreen_score(M[normal], M[mill], M[idx], K)
    rows = np.flatnonzero(np.isin(owner, idx))
    per = millscreen_score(M[normal], M[mill], C[rows], K)
    best = pd.Series(per).groupby(owner[rows]).max()
    return best.reindex(idx).to_numpy()


# ---- select the variant on validation ----
sel = []
for s in SEEDS:
    sp = load_split(docs, "temporal", s)
    vi, vy = sp["val"]
    for v in ("mean", "max"):
        sel.append({"variant": v, "seed": s,
                    "val_pr_auc": average_precision_score(vy, score(v, sp["train_normal"], sp["train_anomaly"], vi))})
sel = pd.DataFrame(sel)
base_val = [average_precision_score(load_split(docs, "temporal", s)["val"][1],
                                    np.load(RESULTS / "scores" / f"millscreen_temporal_seed{s}.npz")["val"]) for s in SEEDS]
print(sel.groupby("variant").val_pr_auc.agg(["mean", "std"]).round(4), "\noriginal (truncated):", np.round(np.mean(base_val), 4))
best = sel.groupby("variant").val_pr_auc.mean().idxmax()
print("selected:", best)

# ---- test once ----
rows, eq = [], []
for s in SEEDS:
    sp = load_split(docs, "temporal", s)
    ti, ty = sp["test"]
    long = n_chunks[ti] > 1
    z = np.load(RESULTS / "scores" / f"millscreen_temporal_seed{s}.npz")
    new_v = score(best, sp["train_normal"], sp["train_anomaly"], sp["val"][0])
    new_t = score(best, sp["train_normal"], sp["train_anomaly"], ti)
    np.savez(RESULTS / "scores" / f"millscreen_chunk{best}_temporal_seed{s}.npz", val=new_v, test=new_t)
    for name, v, t in (("MillScreen (truncated)", z["val"], z["test"]), (f"MillScreen + chunk-{best} (fix)", new_v, new_t)):
        tau = tau_budget(v)
        rows.append({"method": name, "seed": s, **metrics(ty, t, tau),
                     "long_recall": ((t[long] > tau) & (ty[long] == 1)).sum() / max(1, ty[long].sum()),
                     "short_recall": ((t[~long] > tau) & (ty[~long] == 1)).sum() / max(1, ty[~long].sum()),
                     "long_pr_auc": average_precision_score(ty[long], t[long])})
    for rate in (0.01, 0.02, 0.05):
        eq.append({"seed": s, "flag_rate": rate, "recall_original": recall_at_rate(ty, z["test"], rate),
                   "recall_fix": recall_at_rate(ty, new_t, rate)})
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "fix_per_seed.csv", index=False)
cols = ["pr_auc", "roc_auc", "recall_at_tau", "precision_at_tau", "flag_rate", "long_recall", "short_recall", "long_pr_auc"]
pd.set_option("display.width", 250)
print(df.groupby("method", sort=False)[cols].agg(["mean", "std"]).round(4).to_string())
a, b = (df[df.method == m].sort_values("seed") for m in df.method.unique())
for c in ("pr_auc", "recall_at_tau", "roc_auc"):
    t, p = ttest_rel(b[c], a[c])
    print(f"fix - original {c}: {np.mean(b[c].to_numpy() - a[c].to_numpy()):+.4f}, paired t {t:.2f}, p {p:.4f}")
eq = pd.DataFrame(eq)
eq.to_csv(RESULTS / "fix_equal_rate.csv", index=False)
for rate, g in eq.groupby("flag_rate"):
    t, p = ttest_rel(g.recall_fix, g.recall_original)
    print(f"equal flag rate {rate:.0%}: recall {g.recall_original.mean():.4f} -> {g.recall_fix.mean():.4f}, "
          f"diff {np.mean(g.recall_fix - g.recall_original):+.4f}, paired t {t:.2f}, p {p:.4f}")
