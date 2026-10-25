"""Controlled ablations; each changes exactly one factor relative to the full MillScreen setting.

A1 split protocol (temporal vs random)        - T2 and T3
A2 scoring function                           - same embeddings: one-class (kNN / Mahalanobis / IF) vs
                                                mill-proximity vs MillScreen (both terms)
A3 normal-reference contamination (0/1/5 %)   - unlabeled train-era mills mixed into the normal set
A6 label budget (25 / 100 / 400 / all known mills in the mill reference; the same subsets also train the
                                                same-encoder logistic regression, C chosen on val)
A4 metadata shortcut                          - supervised LR on text vs text + publisher/country
A5 threshold rule (budget / best-F1 / 3-sigma)
R  test anomaly ratio (0.5 / 1 / 5 %)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROC, RESULTS
from src.methods import (fit_lr, iforest_score, knn_score, load_docs, load_split, mahalanobis_score, metrics,
                         millscreen_score, subset_ratio, tau_best_f1, tau_budget)

SEEDS = range(5)
LR = sys.argv[1] if len(sys.argv) > 1 else "2e-05"
docs = load_docs()
cfg = json.loads((RESULTS / "t2_config.json").read_text())
E, K = np.load(PROC / f"emb_{cfg['encoder']}.npy"), cfg["k"]
sc = RESULTS / "scores"
rows = []


def add(ablation, setting, method, s, y, test, val):
    rows.append({"ablation": ablation, "setting": setting, "method": method, "seed": s,
                 **metrics(y, test, tau_budget(val))})


for s in SEEDS:
    sp = {k: load_split(docs, k, s) for k in ("temporal", "random")}
    t = sp["temporal"]
    vi, vy = t["val"]
    ti, ty = t["test"]
    tr, mi = E[t["train_normal"]], E[t["train_anomaly"]]

    # A1 split protocol
    for kind in ("temporal", "random"):
        p = sp[kind]
        pn, pm = E[p["train_normal"]], E[p["train_anomaly"]]
        add("A1 split", kind, "T2 MillScreen", s, p["test"][1],
            millscreen_score(pn, pm, E[p["test"][0]], K), millscreen_score(pn, pm, E[p["val"][0]], K))
        f = sc / f"scibert_{kind}_seed{s}_lr{float(LR):g}.npz"
        if f.exists():  # SciBERT on the random split was run for seeds 0-2 only (compute budget)
            z = np.load(f)
            add("A1 split", kind, "T3 SciBERT", s, p["test"][1], z["test"], z["val"])

    # A2 scoring function
    add("A2 scoring", "MillScreen: d(normals) - d(mills) (ours)", "T2", s, ty, millscreen_score(tr, mi, E[ti], K), millscreen_score(tr, mi, E[vi], K))
    add("A2 scoring", "mill proximity only: -d(mills)", "T2", s, ty, -knn_score(mi, E[ti], K), -knn_score(mi, E[vi], K))
    add("A2 scoring", "one-class kNN: d(normals)", "T2", s, ty, knn_score(tr, E[ti], K), knn_score(tr, E[vi], K))
    add("A2 scoring", "one-class Mahalanobis (Ledoit-Wolf)", "T2", s, ty, mahalanobis_score(tr, E[ti]), mahalanobis_score(tr, E[vi]))
    add("A2 scoring", "one-class Isolation Forest", "T2", s, ty, iforest_score(tr, E[ti], s), iforest_score(tr, E[vi], s))

    # A3 contamination of the normal reference. Known mills are split in two disjoint halves:
    # half A is the mill reference in every setting, half B supplies the undetected (unlabeled) mills.
    rng = np.random.default_rng(s)
    perm = rng.permutation(t["train_anomaly"])
    half_a, half_b = perm[: len(perm) // 2], perm[len(perm) // 2:]
    for c in (0.0, 0.01, 0.03):
        n_add = int(round(c / (1 - c) * len(tr)))
        extra = half_b[:n_add]
        trc = E[np.concatenate([t["train_normal"], extra]).astype(int)]
        add("A3 contamination", f"{c:.0%}", "T2", s, ty, millscreen_score(trc, E[half_a], E[ti], K),
            millscreen_score(trc, E[half_a], E[vi], K))

    # A6 label budget: how many known mills does the mill reference need?
    for nb in (25, 100, 400, len(t["train_anomaly"])):
        mb = E[rng.choice(t["train_anomaly"], nb, replace=False)]
        kk = min(K, nb)
        add("A6 label budget", f"{nb} known mills", "T2", s, ty, millscreen_score(tr, mb, E[ti], kk), millscreen_score(tr, mb, E[vi], kk))
        lr, _ = fit_lr(np.r_[tr, mb], np.r_[np.zeros(len(tr)), np.ones(nb)], E[vi], vy)
        add("A6 label budget", f"{nb} known mills", "T3 LR (same embeddings)", s, ty, lr.decision_function(E[ti]), lr.decision_function(E[vi]))

    # A4 metadata shortcut (supervised logistic regression)
    meta = np.c_[docs.publisher.fillna("").str.contains("Hindawi").to_numpy(),
                 docs.countries.fillna("").str.contains("CN").to_numpy()].astype(np.float32)
    for kind in ("temporal", "random"):
        p = sp[kind]
        tri = np.concatenate([p["train_normal"], p["train_anomaly"]])
        try_ = np.r_[np.zeros(len(p["train_normal"])), np.ones(len(p["train_anomaly"]))]
        for feat_name, X in (("text", E), ("text + publisher/country", np.c_[E, 3 * meta])):
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X[tri], try_)
            add(f"A4 metadata ({kind})", feat_name, "LR supervised", s, p["test"][1],
                clf.decision_function(X[p["test"][0]]), clf.decision_function(X[p["val"][0]]))

    # A5 threshold rule (same MillScreen scores)
    val_s, test_s = millscreen_score(tr, mi, E[vi], K), millscreen_score(tr, mi, E[ti], K)
    loo = knn_score(tr, tr, K, exclude_self=True) - knn_score(mi, tr, K)  # normals scored label-free (self excluded)
    for rule, tau in (("review budget 2% (ours)", tau_budget(val_s)), ("best F1 on val", tau_best_f1(val_s, vy)),
                      ("3-sigma of train-normal scores", float(loo.mean() + 3 * loo.std()))):
        rows.append({"ablation": "A5 threshold", "setting": rule, "method": "T2", "seed": s, **metrics(ty, test_s, tau)})

    # R test anomaly ratio
    for name, f in (("T1 Tortured phrases", "tortured"), ("T2 MillScreen", "millscreen"), ("T3 SciBERT", "scibert")):
        fn = f"{f}_temporal_seed{s}" + (f"_lr{float(LR):g}" if f == "scibert" else "") + ".npz"
        z = np.load(sc / fn)
        for r in (0.005, 0.01, 0.05):
            keep = subset_ratio(t["test_full"], r)
            add("R test ratio", f"{r:.1%}", name, s, t["test_full"][1][keep], z["test_full"][keep], z["val"])
    print("seed", s, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "ablations_per_seed.csv", index=False)
agg = df.groupby(["ablation", "method", "setting"], sort=False)[["pr_auc", "roc_auc", "recall_at_tau", "precision_at_tau", "flag_rate"]].agg(["mean", "std"])
agg.to_csv(RESULTS / "ablations.csv")
pd.set_option("display.width", 250)
print(agg.round(4).to_string())
