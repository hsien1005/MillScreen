"""Tier 1 (tortured phrases, always-normal) and Tier 2 (MillScreen), plus the normal-only kNN variant.

MillScreen (Tier 2): s(x) = d_k(x, train normals) - d_k(x, train-era known mills), frozen embeddings.
One-class kNN (planned design, kept as a reported variant): s(x) = d_k(x, train normals).
Model selection (encoder, k) uses validation PR-AUC only; the test set is scored once afterwards.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROC, RESULTS
from src.methods import knn_score, knn_score_multi_k, load_docs, load_split

SEEDS, KS, ENCS = range(5), [1, 5, 10, 20, 50], ["minilm", "specter2"]
docs = load_docs()
out = RESULTS / "scores"
out.mkdir(exist_ok=True)
embs = {e: np.load(PROC / f"emb_{e}.npy") for e in ENCS}

# ---- model selection on validation, for both variants ----
sel = []
for e in ENCS:
    for s in SEEDS:
        sp = load_split(docs, "temporal", s)
        vi, vy = sp["val"]
        dn = knn_score_multi_k(embs[e][sp["train_normal"]], embs[e][vi], KS)
        dm = knn_score_multi_k(embs[e][sp["train_anomaly"]], embs[e][vi], KS)
        for k in KS:
            sel.append({"variant": "oneclass_knn", "encoder": e, "k": k, "seed": s, "val_pr_auc": average_precision_score(vy, dn[k])})
            sel.append({"variant": "millscreen", "encoder": e, "k": k, "seed": s, "val_pr_auc": average_precision_score(vy, dn[k] - dm[k])})
sel = pd.DataFrame(sel)
grid = sel.groupby(["variant", "encoder", "k"]).val_pr_auc.agg(["mean", "std"]).reset_index()
grid.to_csv(RESULTS / "t2_selection_val.csv", index=False)
cfgs = {}
for v in ("millscreen", "oneclass_knn"):
    g = grid[grid.variant == v]
    b = g.loc[g["mean"].idxmax()]
    cfgs[v] = {"encoder": b.encoder, "k": int(b.k)}
(RESULTS / "t2_config.json").write_text(json.dumps(cfgs["millscreen"]))
(RESULTS / "t2_configs_all.json").write_text(json.dumps(cfgs))
print(grid.round(4).to_string(), "\nselected:", cfgs)

# ---- final scoring (val + full test pool) ----
for s in SEEDS:
    sp = load_split(docs, "temporal", s)
    vi, _ = sp["val"]
    ti, _ = sp["test_full"]
    n_test = len(sp["test"][0])
    for v, c in cfgs.items():
        E = embs[c["encoder"]]
        enc_ms = float((PROC / f"emb_{c['encoder']}_ms_per_doc.txt").read_text())
        refs = [E[sp["train_normal"]]] + ([E[sp["train_anomaly"]]] if v == "millscreen" else [])

        def score(Q):
            d = knn_score(refs[0], Q, c["k"])
            return d - knn_score(refs[1], Q, c["k"]) if len(refs) == 2 else d

        val = score(E[vi])
        t1 = time.time()
        test_full = score(E[ti])
        knn_ms = 1000 * (time.time() - t1) / len(ti)
        np.savez(out / f"{v}_temporal_seed{s}.npz", val=val, test=test_full[:n_test], test_full=test_full,
                 train_sec=0.0, ms_per_doc=enc_ms + knn_ms)
    tp = docs.tp_rate.to_numpy()
    np.savez(out / f"tortured_temporal_seed{s}.npz", val=tp[vi], test=tp[ti][:n_test], test_full=tp[ti],
             train_sec=0.0, ms_per_doc=0.0)
    z = np.zeros(len(ti))
    np.savez(out / f"always_normal_temporal_seed{s}.npz", val=np.zeros(len(vi)), test=z[:n_test], test_full=z,
             train_sec=0.0, ms_per_doc=0.0)
    print("seed", s, "done", flush=True)
