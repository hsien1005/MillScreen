"""Step 3: build the shared train/val/test splits (5 seeds) for every method.

temporal (main): train 2015-19, val 2020, test 2021-23 (publication year)
random (ablation A1): the same pools mixed over years, split at random with identical sizes

Per seed, 80% of each normal pool is drawn. Anomalies are kept as an ordered list per split, so the
first k give any anomaly ratio (1% main, 0.5%/5% for the sensitivity analysis) without re-splitting.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROC, clean_titles

SEEDS = [0, 1, 2, 3, 4]
FRAC = 0.8
ERAS = {"train": range(2015, 2020), "val": [2020], "test": range(2021, 2024)}

an = pd.read_pickle(PROC / "anomalies.pkl")
no = pd.read_pickle(PROC / "normals.pkl")
docs = pd.concat([an, no], ignore_index=True)
docs = docs[["doi", "label", "year", "title", "abstract", "field", "domain", "journal", "publisher", "countries"]]
raw = docs.title.fillna("")
docs["title"] = clean_titles(docs.title)
print("retraction markers removed from titles:", (raw != docs.title).groupby(docs.label).sum().to_dict())
docs.to_pickle(PROC / "docs.pkl")

out = PROC / "splits"
out.mkdir(exist_ok=True)
summary = []
for split_kind in ["temporal", "random"]:
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s)
        sp = {}
        if split_kind == "temporal":
            for name, yrs in ERAS.items():
                n_pool = no[no.year.isin(yrs)].doi.to_numpy()
                a_pool = an[an.year.isin(yrs)].doi.to_numpy()
                sp[name] = {"normal": rng.choice(n_pool, int(FRAC * len(n_pool)), replace=False).tolist(),
                            "anomaly": rng.permutation(a_pool).tolist()}
        else:
            ref = json.loads((out / f"temporal_seed{s}.json").read_text())
            n_perm, a_perm = rng.permutation(no.doi.to_numpy()), rng.permutation(an.doi.to_numpy())
            i = j = 0
            for name in ["train", "val", "test"]:
                nn, na = len(ref[name]["normal"]), len(ref[name]["anomaly"])
                sp[name] = {"normal": n_perm[i:i + nn].tolist(), "anomaly": a_perm[j:j + na].tolist()}
                i, j = i + nn, j + na
        (out / f"{split_kind}_seed{s}.json").write_text(json.dumps(sp))
        for name in sp:
            summary.append({"split": split_kind, "seed": s, "part": name,
                            "normal": len(sp[name]["normal"]), "anomaly_pool": len(sp[name]["anomaly"])})

summary = pd.DataFrame(summary)
summary.to_csv(PROC / "split_summary.csv", index=False)
print(summary[summary.seed == 0])
