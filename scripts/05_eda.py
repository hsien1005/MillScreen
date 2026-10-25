"""Step 5: exploratory analysis on the TRAINING portion only (temporal split, seed 0; years 2015-19).

Note: anomalies are used here only to describe separability; the Tier 2 model itself never sees them.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import FIGS, RESULTS
from src.methods import load_docs, load_split
from src.common import PROC, type_token_ratio

ENC = sys.argv[1] if len(sys.argv) > 1 else "minilm"
docs = load_docs()
sp = load_split(docs, "temporal", 0)
idx = np.concatenate([sp["train_normal"], sp["train_anomaly"]])
y = np.concatenate([np.zeros(len(sp["train_normal"]), int), np.ones(len(sp["train_anomaly"]), int)])
d = docs.iloc[idx].copy()
d["y"] = y

emb = np.load(PROC / f"emb_{ENC}.npy")
et, ea = np.load(PROC / f"emb_{ENC}_title.npy"), np.load(PROC / f"emb_{ENC}_abstract.npy")
d["ta_sim"] = (et[idx] * ea[idx]).sum(1)
words = d.abstract.str.split()
d["n_words"] = words.str.len()
d["ttr"] = d.abstract.map(type_token_ratio)
d["sent_len"] = d.abstract.map(lambda s: np.mean([len(x.split()) for x in s.split(". ") if x.strip()] or [0]))
d["has_tp"] = (d.tp_rate > 0).astype(int)

feats = ["tp_rate", "has_tp", "ta_sim", "n_words", "ttr", "sent_len"]
rows = []
mi = mutual_info_classif(d[feats].to_numpy(), y, discrete_features=[False, True, False, False, False, False], random_state=0)
for f, m in zip(feats, mi):
    a, b = d.loc[d.y == 1, f], d.loc[d.y == 0, f]
    rows.append({"feature": f, "normal_mean": b.mean(), "mill_mean": a.mean(), "normal_median": b.median(),
                 "mill_median": a.median(), "KS": ks_2samp(a, b).statistic, "KS_p": ks_2samp(a, b).pvalue,
                 "wasserstein": wasserstein_distance(a, b), "MI": m})
sep = pd.DataFrame(rows)
sep.to_csv(RESULTS / "eda_separability.csv", index=False)
d[feats].corr(method="spearman").to_csv(RESULTS / "eda_feature_corr.csv")
print(f"train: {len(d)} docs, mills {y.sum()} ({y.mean():.2%})")
print(sep.round(4).to_string())
print("share of docs with >=1 tortured phrase: normal %.3f%%, mill %.3f%%" % (
    100 * d.loc[d.y == 0, "has_tp"].mean(), 100 * d.loc[d.y == 1, "has_tp"].mean()))
print("Spearman corr:\n", d[feats].corr(method="spearman").round(2))

# H3 (train-only proxy): do mills drift faster than normals between 2015-17 and 2018-19?
def centroid(m):
    v = emb[idx[m]].mean(0)
    return v / np.linalg.norm(v)
early, late = d.year.to_numpy() <= 2017, d.year.to_numpy() >= 2018
drift = {g: 1 - centroid((d.y.to_numpy() == g) & early) @ centroid((d.y.to_numpy() == g) & late) for g in (0, 1)}
gap = {p: 1 - centroid((d.y.to_numpy() == 0) & m) @ centroid((d.y.to_numpy() == 1) & m) for p, m in (("2015-17", early), ("2018-19", late))}
pd.Series({"drift_normal": drift[0], "drift_mill": drift[1], **{f"gap_{k}": v for k, v in gap.items()}}).to_csv(RESULTS / "eda_drift.csv")
print("centroid cosine drift 2015-17 -> 2018-19: normal %.4f, mill %.4f" % (drift[0], drift[1]))
print("normal-vs-mill centroid gap:", {k: round(v, 4) for k, v in gap.items()})

# H5: are mills outliers (far from normals) or a dense cluster (close to each other)?
from src.methods import knn_score
yv = d.y.to_numpy()
En, Em = emb[idx[yv == 0]], emb[idx[yv == 1]]
rng0 = np.random.default_rng(0)
sn, sm = rng0.choice(len(En), 1500, replace=False), rng0.choice(len(Em), 1000, replace=False)
dens = {"pairwise_sim_normal": float((En[sn] @ En[sn].T)[np.triu_indices(1500, 1)].mean()),
        "pairwise_sim_mill": float((Em[sm] @ Em[sm].T)[np.triu_indices(1000, 1)].mean()),
        "knn10_dist_to_normals_normal_LOO": float(knn_score(En, En, 10, exclude_self=True).mean()),
        "knn10_dist_to_normals_mill": float(knn_score(En, Em, 10).mean()),
        "knn10_dist_to_mills_normal": float(knn_score(Em, En, 10).mean()),
        "knn10_dist_to_mills_mill_LOO": float(knn_score(Em, Em, 10, exclude_self=True).mean())}
from sklearn.metrics import roc_auc_score
dn = np.r_[knn_score(En, En, 10, exclude_self=True), knn_score(En, Em, 10)]
dens["roc_auc_dist_to_normals_train"] = float(roc_auc_score(np.r_[np.zeros(len(En)), np.ones(len(Em))], dn))
pd.Series(dens).to_csv(RESULTS / "eda_density.csv")
print("density / outlierness (train only):", {k: round(v, 4) for k, v in dens.items()})

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].hist(dn[: len(En)], bins=60, density=True, alpha=.6, label="Normal (leave-one-out)", color="#4C72B0")
ax[0].hist(dn[len(En):], bins=60, density=True, alpha=.6, label="Paper mill", color="#DD8452")
ax[0].set_xlabel("Mean cosine distance to 10 nearest train normals"); ax[0].set_ylabel("Density"); ax[0].legend(fontsize=8)
dm = np.r_[knn_score(Em, En, 10), knn_score(Em, Em, 10, exclude_self=True)]
ax[1].hist(dm[: len(En)], bins=60, density=True, alpha=.6, label="Normal", color="#4C72B0")
ax[1].hist(dm[len(En):], bins=60, density=True, alpha=.6, label="Paper mill (leave-one-out)", color="#DD8452")
ax[1].set_xlabel("Mean cosine distance to 10 nearest train mills"); ax[1].set_ylabel("Density"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "eda_density.pdf"); fig.savefig(FIGS / "eda_density.png", dpi=150)

# ---- figures ----
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for g, lab, c in ((0, "Normal", "#4C72B0"), (1, "Paper mill", "#DD8452")):
    s = d[d.y == g]
    ax[0].hist(s.ta_sim, bins=50, density=True, alpha=.6, label=lab, color=c)
    ax[1].hist(s.n_words.clip(upper=500), bins=50, density=True, alpha=.6, label=lab, color=c)
    ax[2].hist(s.ttr, bins=50, density=True, alpha=.6, label=lab, color=c)
ax[0].set_xlabel("Title-abstract cosine similarity"); ax[1].set_xlabel("Abstract length (words, clipped at 500)")
ax[2].set_xlabel("Type-token ratio")
for a in ax:
    a.set_ylabel("Density")
ax[0].legend()
fig.tight_layout(); fig.savefig(FIGS / "eda_distributions.pdf"); fig.savefig(FIGS / "eda_distributions.png", dpi=150)

rng = np.random.default_rng(0)
sub = np.concatenate([rng.choice(np.where(y == 0)[0], 3000, replace=False), np.where(y == 1)[0]])
Z = TSNE(n_components=2, init="pca", random_state=0, perplexity=40).fit_transform(PCA(50, random_state=0).fit_transform(emb[idx[sub]]))
fig, ax = plt.subplots(figsize=(5, 4.2))
ys = y[sub]
ax.scatter(*Z[ys == 0].T, s=3, alpha=.35, label="Normal (3,000 sampled)", color="#4C72B0")
ax.scatter(*Z[ys == 1].T, s=4, alpha=.6, label="Paper mill", color="#DD8452")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(markerscale=3)
fig.tight_layout(); fig.savefig(FIGS / "eda_tsne.pdf"); fig.savefig(FIGS / "eda_tsne.png", dpi=150)
print("figures saved")
