"""Main results table: all tiers on the identical temporal test split, 5 seeds, tau from the val review budget.

Significance: paired t-test over seeds, plus a journal-clustered bootstrap 95% CI of the PR-AUC gap
(papers in the same journal are not independent).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import RESULTS, tortured_hits
from src.methods import load_docs, load_split, metrics, tau_budget

SEEDS = range(5)
LR = sys.argv[1] if len(sys.argv) > 1 else "2e-05"
METHODS = {"T1 Always normal": "always_normal_temporal_seed{s}.npz",
           "T1 Tortured phrases": "tortured_temporal_seed{s}.npz",
           "T2 variant: one-class kNN (normal-only)": "oneclass_knn_temporal_seed{s}.npz",
           "T2 MillScreen (semi-supervised kNN)": "millscreen_temporal_seed{s}.npz",
           "T3 SciBERT (supervised)": "scibert_temporal_seed{s}_lr" + f"{float(LR):g}" + ".npz"}
docs = load_docs()
sc = RESULTS / "scores"

# tortured-phrase latency (regex over raw text)
sample = docs.abstract.iloc[:2000].tolist()
t0 = time.time()
for a in sample:
    tortured_hits(a)
tp_ms = 1000 * (time.time() - t0) / len(sample)

rows = []
for name, pat in METHODS.items():
    for s in SEEDS:
        z = np.load(sc / pat.format(s=s))
        sp = load_split(docs, "temporal", s)
        tau = tau_budget(z["val"])
        m = metrics(sp["test"][1], z["test"], tau)
        ms = tp_ms if "Tortured" in name else float(z["ms_per_doc"])
        rows.append({"method": name, "seed": s, "tau": tau, **m, "ms_per_doc": ms, "train_sec": float(z["train_sec"])})
per_seed = pd.DataFrame(rows)
per_seed.to_csv(RESULTS / "main_per_seed.csv", index=False)
cols = ["pr_auc", "roc_auc", "recall_at_tau", "precision_at_tau", "flag_rate", "ms_per_doc", "train_sec"]
agg = per_seed.groupby("method", sort=False)[cols].agg(["mean", "std"])
agg.to_csv(RESULTS / "main_table.csv")
pd.set_option("display.width", 250)
print(agg.round(4).to_string())
sp0 = load_split(docs, "temporal", 0)
print("test size per seed:", len(sp0["test"][1]), "anomalies:", int(sp0["test"][1].sum()),
      "| val:", len(sp0["val"][1]), "anomalies:", int(sp0["val"][1].sum()),
      "| train normals:", len(sp0["train_normal"]), "train mills (T3 only):", len(sp0["train_anomaly"]))

# ---- significance ----
t2, t3, t1 = (per_seed[per_seed.method == m].sort_values("seed") for m in
              ["T2 MillScreen (semi-supervised kNN)", "T3 SciBERT (supervised)", "T1 Tortured phrases"])
sig = []
for a, b, an, bn in ((t2, t3, "T2", "T3"), (t2, t1, "T2", "T1")):
    for col in ("pr_auc", "recall_at_tau"):
        t, p = ttest_rel(a[col].to_numpy(), b[col].to_numpy())
        sig.append({"compare": f"{an} - {bn}", "metric": col, "mean_diff": (a[col].to_numpy() - b[col].to_numpy()).mean(),
                    "paired_t": t, "p": p})

# journal-clustered bootstrap on each seed, pooled percentiles
rng = np.random.default_rng(0)
diffs = []
for s in SEEDS:
    sp = load_split(docs, "temporal", s)
    idx, y = sp["test"]
    a = np.load(sc / METHODS["T2 MillScreen (semi-supervised kNN)"].format(s=s))["test"]
    b = np.load(sc / METHODS["T3 SciBERT (supervised)"].format(s=s))["test"]
    j = docs.journal.fillna("NA").to_numpy()[idx]
    groups = pd.Series(np.arange(len(idx))).groupby(j).apply(np.array).tolist()
    for _ in range(400):
        pick = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        if y[pick].sum() == 0:
            continue
        diffs.append(average_precision_score(y[pick], a[pick]) - average_precision_score(y[pick], b[pick]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
sig.append({"compare": "T2 - T3", "metric": "pr_auc (journal-cluster bootstrap 95% CI)",
            "mean_diff": float(np.mean(diffs)), "paired_t": lo, "p": hi})
sig = pd.DataFrame(sig)
sig.to_csv(RESULTS / "main_significance.csv", index=False)
print(sig.round(4).to_string())
print(f"cluster bootstrap T2-T3 PR-AUC gap: mean {np.mean(diffs):.4f}, 95% CI [{lo:.4f}, {hi:.4f}], n={len(diffs)}")

# ---- review-budget sensitivity: tau at the (1 - b) quantile of val scores, b in {1%, 2%, 5%} ----
bud = []
for name, pat in METHODS.items():
    if "Always" in name:
        continue
    for s in SEEDS:
        z = np.load(sc / pat.format(s=s))
        y = load_split(docs, "temporal", s)["test"][1]
        for b in (0.01, 0.02, 0.05):
            m = metrics(y, z["test"], float(np.quantile(z["val"], 1 - b)))
            bud.append({"method": name, "seed": s, "budget": b, **m})
bud = pd.DataFrame(bud)
bud.to_csv(RESULTS / "budget_sensitivity.csv", index=False)
print(bud.groupby(["method", "budget"], sort=False)[["recall_at_tau", "precision_at_tau", "flag_rate"]]
      .agg(["mean", "std"]).round(4).to_string())
