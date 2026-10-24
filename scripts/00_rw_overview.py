"""Step 0: describe the two paper-mill waves in Retraction Watch (motivation table; no OpenAlex calls).

Counts every paper-mill retraction by original publication year (2015-19 = wave 1 / train era,
2020 = val, 2021-23 = wave 2 / test era): size, publisher and country mix, subject areas,
share flagged for computer-aided content, and time from publication to retraction.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import RAW, RESULTS

rw = pd.read_csv(RAW / "retraction_watch.csv", encoding="utf-8-sig", low_memory=False)
r = rw[(rw.RetractionNature == "Retraction") & rw.Reason.fillna("").str.contains("Paper Mill")].copy()
r["pub"] = pd.to_datetime(r.OriginalPaperDate, errors="coerce")
r["ret"] = pd.to_datetime(r.RetractionDate, errors="coerce")
r["era"] = pd.cut(r.pub.dt.year, [2014, 2019, 2020, 2023], labels=["2015-19", "2020", "2021-23"])
r = r[r.era.notna()]
r["has_doi"] = r.OriginalPaperDOI.fillna("").str.lower().str.strip().str.startswith("10.")
r["hindawi"] = r.Publisher.eq("Hindawi")
r["china"] = r.Country.fillna("").str.contains("China")
r["computer_aided"] = r.Reason.str.contains("Computer-Aided Content")
r["years_to_retraction"] = (r.ret - r.pub).dt.days / 365.25
r["areas"] = r.Subject.fillna("").str.findall(r"\(([^)]+)\)").apply(lambda l: sorted(set(l)))

g = r.groupby("era", observed=True)
waves = pd.DataFrame({"n": g.size(), "with_doi": g.has_doi.sum(),
                      "hindawi_pct": g.hindawi.mean() * 100, "china_pct": g.china.mean() * 100,
                      "computer_aided_pct": g.computer_aided.mean() * 100,
                      "median_years_to_retraction": g.years_to_retraction.median()})
# subject areas are multi-label (a paper can be BLS and HSC); shares are of papers, not of labels
for area in ("BLS", "HSC", "B/T", "PHY", "SOC", "ENV", "HUM"):
    waves[f"{area}_pct"] = g.areas.apply(lambda s: s.map(lambda l: area in l).mean() * 100)
waves = waves.round(1)
waves.to_csv(RESULTS / "rw_waves.csv")
# share of subject *labels* per era (each label counted once), the version quoted in the draft's wave table
lab = r[["era", "areas"]].explode("areas")
lab = pd.crosstab(lab.areas, lab.era, normalize="columns").mul(100).round(1)
lab = lab[lab.max(axis=1) > 3]
lab.to_csv(RESULTS / "rw_waves_subject_labels.csv")
pd.set_option("display.width", 200)
print(waves.T.to_string())
print("\nshare of subject labels (%):\n" + lab.to_string())
print("\ntop journals per era:")
for era, d in g:
    print(era, d.Journal.value_counts().head(3).to_dict())
print("median years to retraction, all paper-mill 2015-23:", round(r.years_to_retraction.median(), 2),
      "| median of (retraction year - publication year):", (r.ret.dt.year - r.pub.dt.year).median())
