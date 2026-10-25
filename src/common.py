"""Shared helpers: paths, OpenAlex fetching with an on-disk cache, text utilities."""
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
for d in (CACHE, PROC, RESULTS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

OPENALEX = "https://api.openalex.org/works"
SELECT = "id,doi,title,publication_year,type,primary_topic,primary_location,authorships,is_retracted,abstract_inverted_index"


def fetch(params: dict) -> dict:
    """GET OpenAlex /works with params; every response is cached so reruns cost no API budget."""
    url = OPENALEX + "?" + urllib.parse.urlencode(params, safe=":|,/")
    key = hashlib.sha1(url.encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nycu-dm-hw2"})
            with urllib.request.urlopen(req, timeout=90) as f:
                data = json.load(f)
            path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.12)
            return data
        except Exception as e:  # network hiccup or 429
            if attempt == 7:
                raise
            time.sleep(5 * 2 ** min(attempt, 5))  # back off longer on 429 (rate limit)
            print("retry", attempt, e)


def fetch_doi(doi: str) -> dict | None:
    """GET one work by DOI (for DOIs whose characters, such as '|' or ',', break the OR-ed doi filter).
    Cached like fetch(); returns None when OpenAlex has no record."""
    url = OPENALEX + "/doi:" + urllib.parse.quote(doi, safe="") + "?select=" + SELECT
    path = CACHE / f"{hashlib.sha1(url.encode()).hexdigest()}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data or None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nycu-dm-hw2"})
            with urllib.request.urlopen(req, timeout=90) as f:
                data = json.load(f)
            path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.12)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                path.write_text("{}", encoding="utf-8")
                return None
            if attempt == 7:
                raise
            time.sleep(5 * 2 ** min(attempt, 5))
            print("retry", attempt, e)


def rebuild_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))


def flatten(w: dict) -> dict:
    pt = w.get("primary_topic") or {}
    src = ((w.get("primary_location") or {}).get("source")) or {}
    countries = sorted({c for a in (w.get("authorships") or []) for c in (a.get("countries") or [])})
    return {
        "openalex_id": w["id"].rsplit("/", 1)[-1],
        "doi": (w.get("doi") or "").lower().replace("https://doi.org/", ""),
        "title": w.get("title") or "",
        "abstract": rebuild_abstract(w.get("abstract_inverted_index")),
        "year": w.get("publication_year"),
        "type": w.get("type"),
        "field": ((pt.get("field") or {}).get("display_name")),
        "field_id": ((pt.get("field") or {}).get("id") or "").rsplit("/", 1)[-1],
        "domain": ((pt.get("domain") or {}).get("display_name")),
        "journal": src.get("display_name"),
        "publisher": src.get("host_organization_name"),
        "countries": ";".join(countries),
        "oa_is_retracted": w.get("is_retracted"),
    }


# Tortured phrases from Table 1 of Cabanac, Labbe & Magazinov (2021), arXiv:2107.06751.
_TP = [
    r"profound neural organi[sz]ation", r"(fake|counterfeit) neural organi[sz]ation",
    r"versatile organi[sz]ation", r"organi[sz]ation (ambush|assault)", r"organi[sz]ation association",
    r"(enormous|huge|immense|colossal) information", r"information (stockroom|distribution center)",
    r"(counterfeit|human-made) consciousness", r"elite figuring", r"haze figuring",
    r"designs preparing unit", r"focal preparing unit", r"work process motor",
    r"facial acknowledg(e)?ment", r"discourse acknowledg(e)?ment", r"mean square (mistake|blunder)",
    r"mean (outright|supreme) (mistake|blunder)",
    r"(motion|flag|indicator|sign|signal) to (clamor|commotion|noise)", r"worldwide parameters",
    r"(arbitrary|irregular) get right of passage to", r"(arbitrary|irregular) (backwoods|timberland|lush territory)",
    r"(arbitrary|irregular) esteem", r"subterranean insect (state|province|area|region|settlement)",
    r"underground creepy crawly (state|province|area|region|settlement)", r"leftover vitality",
    r"territorial normal vitality", r"motor vitality", r"(credulous|innocent|gullible) bayes",
    r"individual computerized collaborator",
]
TORTURED = re.compile(r"\b(" + "|".join(_TP) + r")\b", re.IGNORECASE)


def tortured_hits(text: str) -> list[str]:
    return [m.group(0) for m in TORTURED.finditer(text or "")]


# Retraction Watch / publishers prepend retraction markers to titles ("RETRACTED: ...", "[Retracted] ...").
# Only retracted papers carry them, so leaving them in leaks the label (found during error analysis).
_MARK_HEAD = re.compile(r"^\s*(\[\s*retracted\s*\]|\(\s*retracted\s*\)|retracted\s*(article)?\s*[:.\-–]"
                        r"|retraction\s*(note)?\s*[:.\-–]|withdrawn\s*[:.\-–])\s*", re.I)


def clean_titles(titles):
    """Strip retraction markers from a pandas Series of titles."""
    t = titles.fillna("")
    for _ in range(2):
        t = t.str.replace(_MARK_HEAD, "", regex=True)
    t = t.str.replace(r"\s*[\[(]\s*retracted\s*[\])]\s*$", "", regex=True, flags=re.I)
    t = t.str.replace(r"^\s*\[\s*retracted article\s*\]\s*", "", regex=True, flags=re.I)
    t = t.str.replace(r"\s*Retraction in\s+\S+\s*$", "", regex=True)
    t = t.mask(t.str.fullmatch(r"\s*retraction\s*", case=False), "")
    return t.str.replace(r"^\s*\[ARTICLE WITHDRAWN\]\s*", "", regex=True)


def n_words(text: str) -> int:
    return len((text or "").split())


def type_token_ratio(text: str) -> float:
    """Distinct words / words of an abstract (0 for an empty abstract)."""
    w = (text or "").lower().split()
    return len(set(w)) / max(1, len(w))
