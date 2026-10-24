"""Step 1: select paper-mill retractions from Retraction Watch and fetch their abstracts from OpenAlex."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import RAW, PROC, SELECT, fetch, flatten

rw = pd.read_csv(RAW / "retraction_watch.csv", encoding="utf-8-sig", low_memory=False)
log = {"rw_rows": len(rw)}
r = rw[rw.RetractionNature == "Retraction"].copy()
log["retractions"] = len(r)
r = r[r.Reason.fillna("").str.contains("Paper Mill")]
log["paper_mill"] = len(r)
r["doi"] = r.OriginalPaperDOI.fillna("").str.lower().str.strip()
r = r[r.doi.str.startswith("10.")].drop_duplicates("doi")
log["with_doi"] = len(r)
r["rw_year"] = pd.to_datetime(r.OriginalPaperDate, errors="coerce").dt.year
r = r[r.rw_year.between(2015, 2023)]
log["year_2015_2023"] = len(r)

rows = []
dois = r.doi.tolist()
for i in range(0, len(dois), 100):
    batch = dois[i:i + 100]
    data = fetch({"filter": "doi:" + "|".join(batch), "per-page": 100, "select": SELECT})
    rows += [flatten(w) for w in data["results"]]
    if i % 2000 == 0:
        print(f"{i}/{len(dois)}", flush=True)

oa = pd.DataFrame(rows).drop_duplicates("doi")
keep = ["doi", "rw_year", "Journal", "Publisher", "Country", "Subject", "Reason", "RetractionDate"]
df = r[keep].merge(oa, on="doi", how="inner")
log["found_in_openalex"] = len(df)
df = df[df.abstract.str.split().str.len() >= 30]
log["with_abstract_ge30w"] = len(df)
df["label"] = 1
df["year"] = df["rw_year"].astype(int)  # Retraction Watch publication date is the reference year
df.to_pickle(PROC / "anomalies.pkl")
pd.Series(log).to_csv(PROC / "anomalies_log.csv")
print(pd.Series(log))
print(df.groupby("year").size())
print(df.field.value_counts().head(12))
