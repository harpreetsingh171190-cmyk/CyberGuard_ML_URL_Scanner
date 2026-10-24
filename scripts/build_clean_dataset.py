"""Build a cleaner training CSV from PhishTank (phishing) + Tranco (popular benign domains).

Download (check each licence and terms yourself):
  PhishTank : https://phishtank.org/developer_info.php  -> verified_online.csv
  Tranco    : https://tranco-list.eu                    -> top-1m.csv (columns: rank,domain; no header)

Usage:
  python scripts/build_clean_dataset.py --phishtank verified_online.csv --tranco top-1m.csv \
         --per-class 20000 --out data/clean_urls.csv

IMPORTANT: Tranco lists domains only, but phishing URLs usually have paths. If benign URLs have no path
and phishing URLs do, a model just learns "has a path = phishing". So by default (--host-only) BOTH
classes are reduced to their host name, which is a fair comparison. Use --full-urls only if you add real
benign URLs with paths from another source.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cyberguard.features import scheme_free  # noqa: E402


def to_host(url: str) -> str:
    try:
        host = urlparse("http://" + scheme_free(url)).hostname or ""
    except ValueError:
        return ""
    return host.lower()


def load_phishtank(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "url" not in cols:
        raise ValueError(f"PhishTank file needs a 'url' column. Found: {list(df.columns)}")
    out = pd.DataFrame({"url": df[cols["url"]].astype(str)})
    date_col = cols.get("submission_time") or cols.get("verification_time")
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True) if date_col else pd.NaT
    out["label"] = 1
    return out


def load_tranco(path: str, top_n: int = 100_000) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["rank", "domain"], nrows=top_n)
    return pd.DataFrame({"url": df["domain"].astype(str), "date": pd.NaT, "label": 0})


def build(phish: pd.DataFrame, benign: pd.DataFrame, per_class: int, host_only: bool = True, seed: int = 42):
    parts = []
    for d in (phish, benign):
        d = d.copy()
        if host_only:
            d["url"] = d["url"].map(to_host)
        d = d[d["url"].str.len() > 0].drop_duplicates(subset="url")
        parts.append(d)
    n = min(per_class, *(len(p) for p in parts))
    if n == 0:
        raise ValueError("One class is empty after cleaning; check your input files")
    out = pd.concat([p.sample(n, random_state=seed) for p in parts], ignore_index=True)
    # A URL that appears in both classes is ambiguous: drop it from both.
    dup = out["url"].duplicated(keep=False) & out.groupby("url")["label"].transform("nunique").gt(1)
    return out[~dup].sample(frac=1, random_state=seed).reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phishtank", required=True)
    p.add_argument("--tranco", required=True)
    p.add_argument("--per-class", type=int, default=20_000)
    p.add_argument("--tranco-top", type=int, default=100_000)
    p.add_argument("--full-urls", action="store_true", help="keep paths (only if benign URLs also have paths)")
    p.add_argument("--out", default="data/clean_urls.csv")
    a = p.parse_args(argv)
    df = build(load_phishtank(a.phishtank), load_tranco(a.tranco, a.tranco_top), a.per_class, not a.full_urls)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df):,} rows ({df['label'].mean():.0%} malicious) -> {a.out}")


if __name__ == "__main__":
    main()
