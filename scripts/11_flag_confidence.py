"""Can an editor trust a single flag? Precision of MillScreen's flags by how far they clear tau and by how close
the nearest known mill is (the explanation an editor reads), pooled over the 5 temporal test sets.
Flags are split into thirds by margin s(x) - tau and, separately, by nearest-known-mill cosine similarity.
Also reports the realised test flag rate against the 2% validation budget.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROC, RESULTS
from src.methods import DEV, load_docs, load_split, tau_budget

docs = load_docs()
cfg = json.loads((RESULTS / "t2_config.json").read_text())
E = np.load(PROC / f"emb_{cfg['encoder']}.npy")
rows = []
for s in range(5):
    sp = load_split(docs, "temporal", s)
    ti, ty = sp["test"]
    z = np.load(RESULTS / "scores" / f"millscreen_temporal_seed{s}.npz")
    tau = tau_budget(z["val"])
    M = torch.from_numpy(E[sp["train_anomaly"]]).to(DEV)
    Q = torch.from_numpy(E[ti]).to(DEV)
    nn_sim = (Q @ M.T).max(1).values.cpu().numpy()
    for k in range(len(ti)):
        rows.append({"seed": s, "y": int(ty[k]), "s": float(z["test"][k]), "tau": tau, "nn_mill_sim": float(nn_sim[k])})
d = pd.DataFrame(rows)
d["flag"] = d.s > d.tau
d["margin"] = d.s - d.tau
f = d[d.flag].copy()
out = []
for col in ("margin", "nn_mill_sim"):
    f["bin"] = pd.qcut(f[col], 3, labels=["lowest third", "middle third", "highest third"])
    g = f.groupby("bin", observed=True).agg(n=("y", "size"), precision=("y", "mean"), lo=(col, "min"), hi=(col, "max")).reset_index()
    g.insert(0, "by", col)
    out.append(g)
res = pd.concat(out)
res.to_csv(RESULTS / "flag_confidence.csv", index=False)
rate = d.groupby("seed").flag.mean()
summary = {"test_flag_rate_mean": float(rate.mean()), "test_flag_rate_std": float(rate.std(ddof=1)),
           "flags_pooled": int(d.flag.sum()), "precision_all_flags": float(f.y.mean())}
(RESULTS / "flag_confidence_summary.json").write_text(json.dumps(summary, indent=1))
print(res.round(3).to_string()); print(summary)
