"""Where does SciBERT's advantage come from? A linear model on word n-grams of the same text.

Logistic regression on TF-IDF 1-3 grams of title + abstract (and, as controls, of only the first two abstract
sentences or only the rest), fitted on the train era with C chosen on validation PR-AUC, scored once on test,
on the same 5 temporal splits. Compared with SciBERT and MillScreen by paired t-tests over seeds.
Usage: python 10_wording_check.py 2e-05
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import RESULTS
from src.methods import load_docs, load_split, metrics, tau_budget

LR = sys.argv[1] if len(sys.argv) > 1 else "2e-05"
docs = load_docs()
sents = docs.abstract.fillna("").map(lambda a: re.split(r"(?<=[.!?])\s+", a))
TEXTS = {"n-grams, title + abstract": (docs.title.fillna("") + " . " + docs.abstract.fillna("")).to_numpy(),
         "n-grams, first two sentences": sents.map(lambda x: " ".join(x[:2])).to_numpy(),
         "n-grams, rest of abstract": sents.map(lambda x: " ".join(x[2:])).to_numpy()}
sc = RESULTS / "scores"
rows, top = [], {}
for s in range(5):
    sp = load_split(docs, "temporal", s)
    tr = np.r_[sp["train_normal"], sp["train_anomaly"]]
    ytr = np.r_[np.zeros(len(sp["train_normal"])), np.ones(len(sp["train_anomaly"]))]
    vi, vy = sp["val"]
    ti, ty = sp["test"]
    for name, text in TEXTS.items():
        vec = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=200000, sublinear_tf=True).fit(text[tr])
        Xtr, Xv, Xt = vec.transform(text[tr]), vec.transform(text[vi]), vec.transform(text[ti])
        best = None
        for C in (0.1, 1.0, 10.0):
            mdl = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
            ap = average_precision_score(vy, mdl.decision_function(Xv))
            if best is None or ap > best[0]:
                best = (ap, C, mdl)
        mdl = best[2]
        rows.append({"method": name, "seed": s, "C": best[1],
                     **metrics(ty, mdl.decision_function(Xt), tau_budget(mdl.decision_function(Xv)))})
        if s == 0 and name.endswith("title + abstract"):
            vocab = np.array(vec.get_feature_names_out())
            top = list(vocab[np.argsort(-mdl.coef_[0])[:30]])
    for name, f in (("SciBERT", f"scibert_temporal_seed{s}_lr{float(LR):g}.npz"), ("MillScreen", f"millscreen_temporal_seed{s}.npz")):
        z = np.load(sc / f)
        rows.append({"method": name, "seed": s, **metrics(ty, z["test"], tau_budget(z["val"]))})
    print("seed", s, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "wording_check_per_seed.csv", index=False)
pv = df.pivot(index="seed", columns="method", values="pr_auc")
ref = "n-grams, title + abstract"
sig = [{"compare": f"{ref} - {o}", "diff": float(np.mean(pv[ref] - pv[o])), "paired_t_p": float(ttest_rel(pv[ref], pv[o]).pvalue)}
       for o in pv.columns if o != ref]
pd.DataFrame(sig).to_csv(RESULTS / "wording_check_significance.csv", index=False)
pd.Series(top, name="top_mill_ngrams_seed0").to_csv(RESULTS / "wording_check_top_ngrams.csv", index=False)
print(df.groupby("method")[["pr_auc", "roc_auc", "recall_at_tau"]].agg(["mean", "std"]).round(3).to_string())
print(pd.DataFrame(sig).round(4).to_string())
print("top mill n-grams (seed 0):", top)
