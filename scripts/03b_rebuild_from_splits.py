"""Step 3b (reproduce instead of re-sample): rebuild docs.pkl from the normals recorded in the repo.

OpenAlex keeps changing, so scripts/02 with the same seed no longer returns the same normal papers. This
step keeps data/processed/splits/*.json as committed and fetches exactly those normals by their OpenAlex
work ID (data/processed/normals_openalex_ids.csv). IDs are used rather than DOIs because one DOI can belong
to several OpenAlex works (e.g. a supplement of conference abstracts) and some DOIs contain characters
that break the OR-ed filter. The anomalies come from step 1; titles are cleaned as in step 3.
Run it instead of 02 + 03.
Usage: python 03b_rebuild_from_splits.py [--keep-markers]   (--keep-markers: leave retraction markers in
titles, which reproduces the label-leakage comparison in results/leaky_titles/)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROC, SELECT, clean_titles, fetch, fetch_doi, flatten

need = set()
for f in (PROC / "splits").glob("*.json"):
    for part in json.loads(f.read_text()).values():
        need |= set(part["normal"])
an = pd.read_pickle(PROC / "anomalies.pkl")
ids = pd.read_csv(PROC / "normals_openalex_ids.csv")
assert set(ids.doi) == need, "normals_openalex_ids.csv does not match the split files"
wids = sorted(ids.openalex_id)
rows = []
for i in range(0, len(wids), 100):  # OpenAlex accepts up to 100 OR-ed values; one ID is one work
    rows += [flatten(w) for w in fetch({"filter": "openalex:" + "|".join(wids[i:i + 100]), "per-page": 100, "select": SELECT})["results"]]
    if i % 5000 == 0:
        print(f"{i}/{len(wids)}", flush=True)
# a work merged into another since the IDs were recorded is fetched by its DOI instead
got = {r["openalex_id"] for r in rows}
for d in ids[~ids.openalex_id.isin(got)].doi:
    w = fetch_doi(d)
    if w:
        rows.append(flatten(w))
print("works not found by ID, fetched by DOI:", int((~ids.openalex_id.isin(got)).sum()))
no = pd.DataFrame(rows).drop_duplicates("doi")
no = no[no.doi.isin(need)].assign(label=0)
print("normals in splits:", len(need), "| fetched:", len(no), "| missing:", len(need - set(no.doi)),
      "| abstract now under 30 words:", int((no.abstract.str.split().str.len() < 30).sum()))
no.to_pickle(PROC / "normals.pkl")
docs = pd.concat([an, no], ignore_index=True)
docs = docs[["doi", "label", "year", "title", "abstract", "field", "domain", "journal", "publisher", "countries"]]
if "--keep-markers" not in sys.argv:
    raw = docs.title.fillna("")
    docs["title"] = clean_titles(docs.title)
    print("retraction markers removed from titles:", (raw != docs.title).groupby(docs.label).sum().to_dict())
docs.to_pickle(PROC / "docs.pkl")
print(docs.groupby(["label", "year"]).size().unstack(0))
