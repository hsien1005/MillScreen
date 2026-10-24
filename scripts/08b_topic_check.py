"""Confound check: is the detector finding mill *templates* or just mill-favoured *topics*?

Controls were matched only at the OpenAlex field level. Here every test paper gets its fine-grained
OpenAlex topic (read from the API cache, no new requests), and ROC-AUC is computed inside each topic
that has >= 5 mills and >= 20 normals, then averaged (weighted by mills). A detector that only
recognises topics should drop to ~0.5 within topics.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import CACHE, RESULTS
from src.methods import load_docs, load_split

LR = sys.argv[1] if len(sys.argv) > 1 else "2e-05"
topic = {}
for f in CACHE.glob("*.json"):
    for w in json.loads(f.read_text(encoding="utf-8")).get("results", []):
        pt = w.get("primary_topic") or {}
        topic[(w.get("doi") or "").lower().replace("https://doi.org/", "")] = pt.get("display_name")
docs = load_docs()
docs["topic"] = docs.doi.map(topic)
print("topic coverage:", docs.topic.notna().mean().round(4))

rows = []
for s in range(5):
    sp = load_split(docs, "temporal", s)
    idx, y = sp["test"]
    t = docs.topic.to_numpy()[idx]
    for name, fn in (("T2 MillScreen", f"millscreen_temporal_seed{s}.npz"),
                     ("T3 SciBERT", f"scibert_temporal_seed{s}_lr{float(LR):g}.npz"),
                     ("T2 variant: one-class kNN", f"oneclass_knn_temporal_seed{s}.npz")):
        sc = np.load(RESULTS / "scores" / fn)["test"]
        per = []
        for tp in pd.unique(t[y == 1]):
            m = t == tp
            if y[m].sum() >= 5 and (y[m] == 0).sum() >= 20:
                per.append((roc_auc_score(y[m], sc[m]), y[m].sum()))
        a = np.array(per)
        rows.append({"method": name, "seed": s, "overall_roc_auc": roc_auc_score(y, sc),
                     "within_topic_roc_auc": np.average(a[:, 0], weights=a[:, 1]), "n_topics": len(a),
                     "mills_covered": int(a[:, 1].sum()), "mills_total": int(y.sum())})
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "topic_check_per_seed.csv", index=False)
print(df.groupby("method")[["overall_roc_auc", "within_topic_roc_auc", "n_topics", "mills_covered", "mills_total"]]
      .agg(["mean", "std"]).round(4).to_string())
