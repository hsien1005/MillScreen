"""Split loading, the three tiers' scorers (except SciBERT, see scripts/06b), thresholds and metrics."""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common import PROC, n_words, tortured_hits

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BUDGET = 0.02  # editors deep-review 20 of every 1,000 submissions


def load_docs():
    docs = pd.read_pickle(PROC / "docs.pkl")
    docs["tp_hits"] = docs.abstract.map(tortured_hits)
    docs["tp_rate"] = docs.tp_hits.map(len) / docs.abstract.map(n_words).clip(lower=1) * 100
    return docs


def load_split(docs, kind="temporal", seed=0, ratio=0.01):
    """Return positional indices into docs. Val/test anomalies are the first k of a fixed ordering."""
    sp = json.loads((PROC / "splits" / f"{kind}_seed{seed}.json").read_text())
    pos = pd.Series(np.arange(len(docs)), index=docs.doi)

    def part(name, with_ratio):
        n = pos[sp[name]["normal"]].to_numpy()
        a = pos[sp[name]["anomaly"]].to_numpy()
        if with_ratio:
            a = a[: int(round(ratio / (1 - ratio) * len(n)))]
        idx = np.concatenate([n, a])
        y = np.concatenate([np.zeros(len(n), int), np.ones(len(a), int)])
        return idx, y

    tr_idx, tr_y = part("train", with_ratio=False)
    return {"train_normal": tr_idx[tr_y == 0], "train_anomaly": tr_idx[tr_y == 1],
            "val": part("val", True), "test": part("test", True),
            "test_full": part("test", False)}  # normals + whole anomaly pool, for ratio sensitivity


def subset_ratio(test_full, ratio):
    """Slice test_full (normals first, then ordered anomalies) down to a given anomaly ratio."""
    idx, y = test_full
    n = int((y == 0).sum())
    k = int(round(ratio / (1 - ratio) * n))
    return np.arange(n + k)


# ---------- Tier 2 scorers (all fit on train normals only) ----------
def knn_score(train_emb, query_emb, k=10, exclude_self=False, chunk=4096):
    N = torch.from_numpy(train_emb).to(DEV, torch.float16)
    out = []
    for i in range(0, len(query_emb), chunk):
        Q = torch.from_numpy(query_emb[i:i + chunk]).to(DEV, torch.float16)
        sims = (Q @ N.T).float()
        top = sims.topk(k + int(exclude_self), dim=1).values
        if exclude_self:
            top = top[:, 1:]
        out.append((1 - top).mean(1).cpu().numpy())
    return np.concatenate(out)


def knn_score_multi_k(train_emb, query_emb, ks, chunk=4096):
    N = torch.from_numpy(train_emb).to(DEV, torch.float16)
    out = {k: [] for k in ks}
    for i in range(0, len(query_emb), chunk):
        Q = torch.from_numpy(query_emb[i:i + chunk]).to(DEV, torch.float16)
        top = (Q @ N.T).float().topk(max(ks), dim=1).values
        for k in ks:
            out[k].append((1 - top[:, :k]).mean(1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def millscreen_score(normal_emb, mill_emb, query_emb, k=10):
    """MillScreen: farther from the normal reference AND closer to known mills -> more suspicious.

    s(x) = d_k(x, normals) - d_k(x, known mills), with d_k the mean cosine distance to the k nearest.
    Paper mills turned out to be a dense template cluster rather than outliers (see EDA), so the
    second term carries most of the signal.
    """
    return knn_score(normal_emb, query_emb, k) - knn_score(mill_emb, query_emb, k)


def mahalanobis_score(train_emb, query_emb):
    lw = LedoitWolf().fit(train_emb)
    d = query_emb - lw.location_
    return np.einsum("ij,jk,ik->i", d, lw.precision_, d)


def iforest_score(train_emb, query_emb, seed=0):
    f = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(train_emb)
    return -f.score_samples(query_emb)


# ---------- Tier 3 reference on the same frozen embeddings ----------
LR_CS = (0.01, 0.1, 1.0, 10.0)


def fit_lr(X_train, y_train, X_val, y_val, cs=LR_CS):
    """Class-weighted logistic regression; C is chosen on validation PR-AUC. Returns (model, C)."""
    best = None
    for c in cs:
        m = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(X_train, y_train)
        ap = average_precision_score(y_val, m.decision_function(X_val))
        if best is None or ap > best[0]:
            best = (ap, c, m)
    return best[2], best[1]


# ---------- thresholds & metrics ----------
def tau_budget(val_scores, budget=BUDGET):
    return float(np.quantile(val_scores, 1 - budget))


def tau_best_f1(val_scores, val_y):
    best, tau = -1, None
    for t in np.unique(np.quantile(val_scores, np.linspace(0.5, 0.9995, 400))):
        p = val_scores >= t
        tp = (p & (val_y == 1)).sum()
        f1 = 2 * tp / (p.sum() + (val_y == 1).sum())
        if f1 > best:
            best, tau = f1, float(t)
    return tau


def journal_bootstrap_gaps(y, a, b, journals, rng, n=400):
    """PR-AUC gap a - b on n resamples of whole journals (papers in one journal are not independent)."""
    groups = pd.Series(np.arange(len(y))).groupby(journals).apply(np.array).tolist()
    out = []
    for _ in range(n):
        pick = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        if y[pick].sum() == 0:
            continue
        out.append(average_precision_score(y[pick], a[pick]) - average_precision_score(y[pick], b[pick]))
    return out


def recall_at_rate(y, s, rate):
    """Recall when exactly the top `rate` share of papers is flagged (equal review load across methods)."""
    k = int(round(rate * len(s)))
    return float(y[np.argsort(-s)[:k]].sum() / max(1, y.sum()))


def metrics(y, s, tau):
    flag = s > tau  # strict: tied scores at tau (e.g. all-zero rule scores) are not flagged
    tp = int((flag & (y == 1)).sum())
    return {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "recall_at_tau": tp / max(1, (y == 1).sum()), "precision_at_tau": tp / max(1, flag.sum()),
            "flag_rate": float(flag.mean()), "tp": tp, "fp": int((flag & (y == 0)).sum()),
            "fn": int((~flag & (y == 1)).sum())}
