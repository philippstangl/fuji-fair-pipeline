"""Marks datasets whose F-UJI run did not resolve (resolved_url == "not defined")
as `failed` in a repository's status table, so the next stage-2 run retries them.

Run from the project root:
    python scripts/mark_failed.py --repo nfdi4cat_dataverse
"""
from __future__ import annotations
from pathlib import Path

import argparse
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.common import atomic_write_parquet  # noqa: E402
from pipeline.stage2_evaluate import is_unresolved  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo", required=True,
        help="Repository name - the data/<repo> subfolder.",
    )
    ap.add_argument(
        "--data-dir", default="data",
        help="Root data directory (default: data).",
    )
    args = ap.parse_args()

    eval_dir = Path(args.data_dir) / args.repo / "02_evaluations"
    status_path = eval_dir / "status.parquet"
    raw_dir = eval_dir / "raw"

    s = pd.read_parquet(status_path)
    # Identify datasets whose landing page F-UJI could not resolve.
    failed: list[str] = []
    for p in raw_dir.glob("*.json"):
        with open(p) as f:
            d = json.load(f)
        if is_unresolved(d):
            failed.append((d.get("request") or {}).get("object_identifier"))

    s.loc[s["pid"].isin(failed), "status"] = "failed"
    atomic_write_parquet(s, status_path)
    print(f"Marked {len(failed)} datasets for re-evaluation in {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
