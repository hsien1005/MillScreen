"""Step 2: sample non-retracted control papers from OpenAlex, matched to the anomalies' field mix per year.

Per publication year we draw a fixed number of controls and spread them over OpenAlex fields in
proportion to that year's paper-mill field shares (fields under 2% are dropped and the rest renormalised).
Usage: python 02_build_normals.py [--dry-run]
"""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import RAW, PROC, SELECT, fetch, flatten

PER_YEAR = {2015: 5000, 2016: 5000, 2017: 5000, 2018: 5000, 2019: 5000,
            2020: 10000, 2021: 12000, 2022: 12000, 2023: 12000}
MIN_SHARE = 0.02
SEED = 2026

an = pd.read_pickle(PROC / "anomalies.pkl")
plan = []
for year, total in PER_YEAR.items():
    share = an[an.year == year].groupby("field_id").size()
    share = share / share.sum()
    share = share[share >= MIN_SHARE]
    share = share / share.sum()
    for fid, s in share.items():
        plan.append({"year": year, "field_id": fid, "n": int(round(total * s))})
plan = pd.DataFrame(plan)
plan["requests"] = (plan.n / 200).apply(math.ceil)
print(plan.groupby("year")[["n", "requests"]].sum(), "\nTotal requests:", plan.requests.sum())
if "--dry-run" in sys.argv:
    sys.exit()

rw = pd.read_csv(RAW / "retraction_watch.csv", encoding="utf-8-sig", low_memory=False)
rw_dois = set(rw.OriginalPaperDOI.fillna("").str.lower().str.strip())

rows = []
for _, p in plan.iterrows():
    flt = (f"publication_year:{p.year},primary_topic.field.id:{p.field_id},type:article,"
           "has_abstract:true,is_retracted:false")
    for page in range(1, p.requests + 1):
        data = fetch({"filter": flt, "sample": p.n, "seed": SEED, "per-page": 200, "page": page, "select": SELECT})
        rows += [flatten(w) for w in data["results"]]
    print(p.year, p.field_id, p.n, "->", len(rows), flush=True)

df = pd.DataFrame(rows).drop_duplicates("doi")
log = {"sampled": len(rows), "unique": len(df)}
df = df[df.doi.str.startswith("10.") & ~df.doi.isin(rw_dois)]
log["not_in_retraction_watch"] = len(df)
df = df[df.abstract.str.split().str.len() >= 30]
log["with_abstract_ge30w"] = len(df)
df = df[~df.doi.isin(set(an.doi))]
df["label"] = 0
df.to_pickle(PROC / "normals.pkl")
pd.Series(log).to_csv(PROC / "normals_log.csv")
plan.to_csv(PROC / "normals_plan.csv", index=False)
print(pd.Series(log))
print(df.groupby("year").size())
