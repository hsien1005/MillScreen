"""Error analysis, interpretability, practical impact and the Hindawi slice (the fix is in 09_fix.py).

All on the temporal split. Error cases come from seed 0; aggregate statements use all 5 seeds.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROC, RESULTS
from src.methods import knn_score, load_docs, load_split, metrics, millscreen_score, tau_budget

SEEDS = range(5)
LR = sys.argv[1] if len(sys.argv) > 1 else "2e-05"
docs = load_docs()
cfg = json.loads((RESULTS / "t2_config.json").read_text())
E, K = np.load(PROC / f"emb_{cfg['encoder']}.npy"), cfg["k"]
ta = (np.load(PROC / f"emb_{cfg['encoder']}_title.npy") * np.load(PROC / f"emb_{cfg['encoder']}_abstract.npy")).sum(1)
docs["ta_sim"] = ta
docs["n_words"] = docs.abstract.str.split().str.len()
docs["ttr"] = docs.abstract.str.split().map(lambda w: len({x.lower() for x in w}) / max(1, len(w)))
sc = RESULTS / "scores"
report = {}

# ---------- error cases (seed 0) ----------
sp = load_split(docs, "temporal", 0)
ti, ty = sp["test"]
z2 = np.load(sc / "millscreen_temporal_seed0.npz")
z3 = np.load(sc / f"scibert_temporal_seed0_lr{float(LR):g}.npz")
tau = tau_budget(z2["val"])
d = docs.iloc[ti].copy()
d["y"], d["s"], d["t3"] = ty, z2["test"], z3["test"]
d["pct"] = d.s.rank(pct=True)
d["kind"] = np.select([(d.y == 1) & (d.s > tau), (d.y == 1), (d.s > tau)], ["TP", "FN", "FP"], "TN")
tr, trm = sp["train_normal"], sp["train_anomaly"]
Ntr, Mtr = E[tr], E[trm]


def nearest(i):
    sn, sm = Ntr @ E[i], Mtr @ E[i]
    jn, jm = int(np.argmax(sn)), int(np.argmax(sm))
    return docs.title.iloc[tr[jn]][:90], float(sn[jn]), docs.title.iloc[trm[jm]][:90], float(sm[jm])


cases = pd.concat([
    d[d.kind == "FN"].nlargest(1, "s").assign(case="FN near tau"),
    d[d.kind == "FN"].nsmallest(1, "s").assign(case="FN far below tau"),
    d[d.kind == "FP"].nsmallest(1, "s").assign(case="FP just above tau"),
    d[d.kind == "FP"].nlargest(1, "s").assign(case="FP far above tau"),
    d[(d.kind == "TP")].nlargest(1, "s").assign(case="TP top score"),
])
cases["nn_normal"], cases["nn_normal_sim"], cases["nn_mill"], cases["nn_mill_sim"] = zip(*[nearest(docs.index.get_loc(i)) for i in cases.index])
cases["tau"] = tau
cases["tp_hits"] = cases.tp_hits.map(lambda h: "; ".join(h))
cols = ["case", "doi", "title", "journal", "publisher", "year", "field", "countries", "s", "tau", "pct", "t3",
        "ta_sim", "ttr", "n_words", "tp_hits", "nn_normal", "nn_normal_sim", "nn_mill", "nn_mill_sim"]
cases[cols].to_csv(RESULTS / "error_cases.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 80)
print(cases[cols].T.to_string())

# shared patterns: FN vs TP and FP vs TN feature contrasts (pooled over seeds)
pool = []
for s in SEEDS:
    p = load_split(docs, "temporal", s)
    z = np.load(sc / f"millscreen_temporal_seed{s}.npz")
    t = tau_budget(z["val"])
    q = docs.iloc[p["test"][0]][["field", "publisher", "countries", "ta_sim", "ttr", "n_words", "tp_rate", "year"]].copy()
    q["y"], q["flag"] = p["test"][1], z["test"] > t
    pool.append(q)
pool = pd.concat(pool)
pool["hindawi"] = pool.publisher.fillna("").str.contains("Hindawi")
pool["cn"] = pool.countries.fillna("").str.contains("CN")
pool["has_tp"] = pool.tp_rate > 0
pat = []
for grp, a_m, b_m in (("FN vs TP", (pool.y == 1) & ~pool.flag, (pool.y == 1) & pool.flag),
                      ("FP vs TN", (pool.y == 0) & pool.flag, (pool.y == 0) & ~pool.flag)):
    a, b = pool[a_m], pool[b_m]
    for f in ("ta_sim", "ttr", "n_words", "has_tp", "hindawi", "cn", "year"):
        pat.append({"contrast": grp, "feature": f, "n_a": len(a), "n_b": len(b), "mean_a": a[f].mean(),
                    "mean_b": b[f].mean(), "mwu_p": mannwhitneyu(a[f].astype(float), b[f].astype(float)).pvalue})
pat = pd.DataFrame(pat)
pat.to_csv(RESULTS / "error_patterns.csv", index=False)
print(pat.round(4).to_string())
fn_fields = pool[(pool.y == 1) & ~pool.flag].field.value_counts(normalize=True).head(5)
tp_fields = pool[(pool.y == 1) & pool.flag].field.value_counts(normalize=True).head(5)
fp_fields = pool[(pool.y == 0) & pool.flag].field.value_counts(normalize=True).head(5)
print("FN fields:\n", fn_fields.round(3), "\nTP fields:\n", tp_fields.round(3), "\nFP fields:\n", fp_fields.round(3))
pd.concat({"FN": fn_fields, "TP": tp_fields, "FP": fp_fields}).to_csv(RESULTS / "error_fields.csv")

# ---------- interpretability: sentence leave-one-out on one flagged paper ----------
if cfg["encoder"] == "specter2":
    w = models.Transformer("allenai/specter2_base", max_seq_length=320)
    enc = SentenceTransformer(modules=[w, models.Pooling(w.get_word_embedding_dimension(), "cls")]).to("cuda")
else:
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to("cuda")
    enc.max_seq_length = 320  # same truncation as scripts/04_embed.py
sep = enc.tokenizer.sep_token or ". "
target = "10.1155/2021/6635947"
row = d[(d.doi == target) & (d.y == 1)]
row = row.iloc[0] if len(row) else d[d.kind == "TP"].nlargest(1, "s").iloc[0]
sents = [x for x in re.split(r"(?<=[.!?])\s+", row.abstract) if x.strip()]
texts = [row.title + f" {sep} " + row.abstract] + [
    row.title + f" {sep} " + " ".join(sents[:i] + sents[i + 1:]) for i in range(len(sents))]
V = enc.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
base = millscreen_score(Ntr, Mtr, V[:1], K)[0]
drops = base - millscreen_score(Ntr, Mtr, V[1:], K)
loo = pd.DataFrame({"sentence": sents, "score_drop": drops, "share_of_excess": drops / max(1e-9, base - tau)})
loo.to_csv(RESULTS / "interpret_loo.csv", index=False)
report["interpret"] = {"doi": row.doi, "title": row.title, "score": float(base), "tau": tau,
                       "t3_prob": float(row.t3), "ta_sim": float(row.ta_sim), "tortured": row.tp_hits}
print("\nInterpretability:", report["interpret"])
print(loo.sort_values("score_drop", ascending=False).round(4).to_string())

# ---------- practical impact & Hindawi slice (5 seeds) ----------
imp = []
for s in SEEDS:
    p = load_split(docs, "temporal", s)
    idx, y = p["test"]
    hind = docs.publisher.fillna("").str.contains("Hindawi").to_numpy()[idx]
    for name, fn in (("T1 Tortured phrases", f"tortured_temporal_seed{s}.npz"),
                     ("T2 MillScreen", f"millscreen_temporal_seed{s}.npz"),
                     ("T3 SciBERT", f"scibert_temporal_seed{s}_lr{float(LR):g}.npz")):
        z = np.load(sc / fn)
        m = metrics(y, z["test"], tau_budget(z["val"]))
        hm = {"hindawi_pr_auc": average_precision_score(y[hind], z["test"][hind]) if y[hind].sum() else np.nan,
              "hindawi_n": int(hind.sum()), "hindawi_mills": int(y[hind].sum()),
              "non_hindawi_pr_auc": average_precision_score(y[~hind], z["test"][~hind])}
        # per 1,000 submissions with 1% prevalence (10 mills) and 20 deep reviews
        imp.append({"method": name, "seed": s, "mills_caught_per_1000": 10 * m["recall_at_tau"],
                    "real_mills_among_reviews": m["tp"] / max(1, m["tp"] + m["fp"]) * 1000 * m["flag_rate"],
                    "reviews_per_1000": 1000 * m["flag_rate"], **hm})
imp = pd.DataFrame(imp)
imp.to_csv(RESULTS / "impact_per_seed.csv", index=False)
print(imp.groupby("method").agg(["mean", "std"]).round(3).T.to_string())

(RESULTS / "analysis_report.json").write_text(json.dumps(report, indent=1, default=str))
