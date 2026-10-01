"""Extract the cohort of low-scoring datasets for manual investigation.

Run from the project root:
    python scripts/inspect_low_scorers.py --repo nfdi4cat_dataverse

Optional flags:
    --threshold 50           score cutoff in percent (default: 50)
    --out failing.csv        write filtered rows to a CSV
"""
from __future__ import annotations
from pathlib import Path

import argparse
import sys
import duckdb

SUMMARY_PARQUET: Path | None = None

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo", required=True,
        help="Repository name - the data/<repo> subfolder to inspect.",
    )
    ap.add_argument(
        "--data-dir", default="data",
        help="Root data directory (default: data).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="FAIR %% cutoff; rows below this are considered failing (default: 50)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV output path (e.g. data/04_analysis/failing.csv)",
    )
    args = ap.parse_args()

    global SUMMARY_PARQUET
    SUMMARY_PARQUET = Path(args.data_dir) / args.repo / "03_parsed" / "summary.parquet"

    if not SUMMARY_PARQUET.exists():
        print(f"ERROR: {SUMMARY_PARQUET} not found. Run the pipeline first.",
              file=sys.stderr)
        return 1

    sql = """
        SELECT
            global_id,
            name,
            name_of_dataverse,
            published_at,
            fair_percent,
            f_percent,
            a_percent,
            i_percent,
            r_percent
        FROM read_parquet(?)
        WHERE fair_percent < ?
        ORDER BY name_of_dataverse NULLS LAST, published_at
    """

    con = duckdb.connect(":memory:")
    # Inline params with proper quoting via DuckDB's prepared statement
    rel = con.sql(sql, params=[SUMMARY_PARQUET.as_posix(), args.threshold])
    print(f"\nDatasets with fair_percent < {args.threshold}:\n")
    rel.show(max_width=10_000, max_rows=200)

    print("\nBreakdown by sub-dataverse:\n")
    con.sql(
        """
        SELECT
            COALESCE(name_of_dataverse, '(unassigned)') AS dataverse,
            COUNT(*)                                    AS n_failing,
            ROUND(AVG(fair_percent), 1)                 AS mean_fair_pct
        FROM read_parquet(?)
        WHERE fair_percent < ?
        GROUP BY dataverse
        ORDER BY n_failing DESC
        """,
        params=[SUMMARY_PARQUET.as_posix(), args.threshold],
    ).show(max_width=10_000, max_rows=50)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_path = args.out.as_posix().replace("'", "''")
        parquet_path = SUMMARY_PARQUET.as_posix().replace("'", "''")
        con.execute(f"""
            COPY (
                SELECT
                    global_id, name, name_of_dataverse, published_at,
                    fair_percent, f_percent, a_percent, i_percent, r_percent
                FROM read_parquet('{parquet_path}')
                WHERE fair_percent < {float(args.threshold)}
                ORDER BY name_of_dataverse NULLS LAST, published_at
            ) TO '{out_path}' (HEADER, DELIMITER ',')
        """)
        print(f"\n wrote {args.out}")

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
