"""DuckDB analysis of F-UJI evaluation results.

Inputs (Parquet, produced by stage 3 of the pipeline, per repository):
    data/<repo>/03_parsed/summary.parquet     one row per dataset
    data/<repo>/03_parsed/metrics.parquet     one row per (dataset, metric)

Outputs (under data/<repo>/04_analysis/):
    CSVs:
        headline.csv                   overall FAIR % distribution
        histogram.csv                  binned FAIR % counts
        dimension_gap.csv              mean F / A / I / R scores
        worst_metrics.csv              10 lowest pass-rate metrics
        metric_pass_rates.csv          all metrics ranked by pass rate
        dataverse_breakdown.csv        mean FAIR % per sub-dataverse
        year_trend.csv                 mean FAIR % by publication year
    Plots:
        plots/histogram.{png,svg}
        plots/dimension_gap.{png,svg}
        plots/metric_pass_rates.{png,svg}
        plots/dataverse_breakdown.{png,svg}
        plots/year_trend.{png,svg}

Run from the project root:
    pip install duckdb matplotlib
    python -m pipeline.analysis.analyze_with_plots --repo nfdi4cat_dataverse

Plotting is optional: if matplotlib is not installed, CSVs are produced as
normal and plot generation is skipped with a warning.

Optional:
    --db data/repo4cat.duckdb          persist the in-memory DB
"""
from __future__ import annotations

import argparse
import sys
import duckdb

from pathlib import Path
from textwrap import dedent
from typing import Callable

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False

PARSED_DIR: Path | None = None
OUT_DIR: Path | None = None
PLOT_DIR: Path | None = None
SUMMARY_PARQUET: Path | None = None
METRICS_PARQUET: Path | None = None

PRINCIPLE_COLORS = {
    "F": "#1f77b4", 
    "A": "#2ca02c", 
    "I": "#d62728",
    "R": "#9467bd", 
}

def section(title: str) -> None:
    """Print a visible section header to stdout."""
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def run_and_capture(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    title: str,
    csv_name: str | None,
    plot_fn: Callable[[list[tuple], list[str]], "plt.Figure"] | None = None,
    plot_name: str | None = None,
) -> duckdb.DuckDBPyRelation:
    section(title)
    rel = con.sql(sql)
    rows = rel.fetchall()
    cols = rel.columns

    # Re-issue the query for nice tabular display
    con.sql(sql).show(max_width=10_000, max_rows=200)

    if csv_name:
        out_path = OUT_DIR / csv_name
        # Re-execute via COPY so DuckDB streams to disk without going through Python
        con.sql(f"COPY ({sql}) TO '{out_path.as_posix()}' (HEADER, DELIMITER ',')")
        print(f"wrote {out_path}")

    if plot_fn is not None and plot_name is not None:
        _render_plot(plot_fn, rows, cols, plot_name)

    return rel


def has_subdataverse_info(con: duckdb.DuckDBPyConnection) -> bool:
    """Does repository summary carry sub-dataverse membership?

    Only Dataverse harvests populate `name_of_dataverse`. For Zenodo (and any
    future backend) it is null on every row, so the breakdown would result in
    a single '(unassigned)' bar.
    """
    cols = set(con.sql("SELECT * FROM summary LIMIT 0").columns)
    if "name_of_dataverse" not in cols:
        return False
    n = con.sql("SELECT COUNT(name_of_dataverse) FROM summary").fetchone()[0]
    return bool(n)


def _render_plot(
    plot_fn: Callable[[list[tuple], list[str]], "plt.Figure"],
    rows: list[tuple],
    cols: list[str],
    plot_name: str,
) -> None:
    if not _HAS_MPL:
        print(f"matplotlib not installed. Skipping plot '{plot_name}'")
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fig = plot_fn(rows, cols)
    except Exception as e:
        print(f"(plot '{plot_name}' failed: {type(e).__name__}: {e})")
        return
    png_path = PLOT_DIR / f"{plot_name}.png"
    svg_path = PLOT_DIR / f"{plot_name}.svg"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(PLOT_DIR / f"{plot_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path} (+ .svg, .pdf)")


# Plot functions

def _col(cols: list[str], name: str) -> int:
    """Find a column index by name; raises KeyError if missing."""
    try:
        return cols.index(name)
    except ValueError as e:
        raise KeyError(f"plot expected column '{name}', got {cols}") from e


def plot_histogram(rows, cols):
    """Bar chart of the binned FAIR % distribution."""
    i_low = _col(cols, "pct_bin_low")
    i_n = _col(cols, "n")
    bins = [r[i_low] for r in rows]
    counts = [r[i_n] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([b + 5 for b in bins], counts, width=9, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("FAIR % (10-pp bins)")
    ax.set_ylabel("Number of datasets")
    ax.set_title("Distribution of overall FAIR % across datasets")
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.grid(axis="y", alpha=0.3)
    for b, c in zip(bins, counts):
        ax.text(b + 5, c, str(c), ha="center", va="bottom", fontsize=9)
    return fig


def plot_dimension_gap(rows, cols):
    """Bar chart of mean F/A/I/R scores with min/max range as error bars."""
    i_p = _col(cols, "principle")
    i_mean = _col(cols, "mean_pct")
    i_min = _col(cols, "min_pct")
    i_max = _col(cols, "max_pct")
    principles = [r[i_p] for r in rows]
    means = [r[i_mean] for r in rows]
    # Error bars expressed as distance from the mean
    err_lo = [r[i_mean] - r[i_min] for r in rows]
    err_hi = [r[i_max] - r[i_mean] for r in rows]
    colors = [PRINCIPLE_COLORS.get(p, "#888") for p in principles]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(principles, means, color=colors, edgecolor="white",
           yerr=[err_lo, err_hi], capsize=6, ecolor="#333")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Mean score by FAIR principle (error bars = min–max range)")
    ax.grid(axis="y", alpha=0.3)
    for p, m in zip(principles, means):
        ax.text(p, m, f" {m:.1f}", ha="center", va="bottom", fontsize=10)
    return fig


def plot_metric_pass_rates(rows, cols):
    """Horizontal bar chart of all metrics ranked by mean points earned."""
    i_id = _col(cols, "metric_identifier")
    i_pr = _col(cols, "principle")
    i_rate = _col(cols, "earned_pct")

    # Sort ascending so the worst-performing metrics is at the TOP of the chart
    data = sorted(rows, key=lambda r: (r[i_rate], r[i_id]))
    labels = [r[i_id] for r in data]
    rates = [r[i_rate] for r in data]
    colors = [PRINCIPLE_COLORS.get(r[i_pr], "#888") for r in data]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(labels))))
    ax.barh(labels, rates, color=colors, edgecolor="white")
    ax.set_xlabel("Mean share of points earned (%)")
    # Extra horizontal room so the "100%" annotations don't run into the
    # right axis or the legend.
    ax.set_xlim(0, 118)
    ax.set_title("Per-metric points earned across all datasets")
    ax.invert_yaxis()  # worst at top
    ax.grid(axis="x", alpha=0.3)
    for i, r in enumerate(rates):
        ax.text(r + 1, i, f"{r:.0f}%", va="center", fontsize=8)
    # Legend for principle colors
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PRINCIPLE_COLORS.values()]
    ax.legend(handles, PRINCIPLE_COLORS.keys(), title="Principle",
              loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize=9,
              borderaxespad=0.0)
    return fig


def plot_dataverse_breakdown(rows, cols):
    """Horizontal bar of mean FAIR % per sub-dataverse, sorted descending.
    Labels truncated because some dataverse names are very long."""
    i_dv = _col(cols, "dataverse")
    i_mean = _col(cols, "mean_fair_pct")
    i_n = _col(cols, "n_datasets")

    # Sort so the highest mean is rendered at the TOP of the chart
    data = sorted(rows, key=lambda r: (r[i_mean] is None, r[i_mean] or 0))
    labels = [(r[i_dv][:55] + "…") if len(r[i_dv]) > 56 else r[i_dv] for r in data]
    means = [r[i_mean] for r in data]
    ns = [r[i_n] for r in data]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
    ax.barh(labels, means, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Mean FAIR % (n shown in label)")
    ax.set_xlim(0, 115)  # extra room for "(n=NN)" annotations
    ax.set_title("FAIR % by sub-dataverse")
    ax.grid(axis="x", alpha=0.3)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text((m or 0) + 1, i, f"{m:.1f}% (n={n})" if m is not None else f"(n={n})",
                va="center", fontsize=8)
    return fig


def plot_year_trend(rows, cols):
    """Line plot of mean FAIR % over publication year, with dataset counts."""
    i_y = _col(cols, "year")
    i_mean = _col(cols, "mean_fair_pct")
    i_n = _col(cols, "n_datasets")
    years = [r[i_y] for r in rows]
    means = [r[i_mean] for r in rows]
    ns = [r[i_n] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(years, means, marker="o", linewidth=2, color="#4c72b0")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Mean FAIR %")
    ax.set_ylim(0, 105)
    ax.set_title("Mean FAIR % by publication year")
    ax.grid(alpha=0.3)
    for y, m, n in zip(years, means, ns):
        ax.annotate(f"n={n}", (y, m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    # Integer x-ticks because years are discrete.
    if years:
        ax.set_xticks(list(range(int(min(years)), int(max(years)) + 1)))
    return fig


def run(repo: str, data_dir: str = "data", db: str = ":memory:") -> None:
    """Analyze one repository's stage-3 output."""
    global PARSED_DIR, OUT_DIR, PLOT_DIR, SUMMARY_PARQUET, METRICS_PARQUET
    base = Path(data_dir) / repo
    PARSED_DIR = base / "03_parsed"
    OUT_DIR = base / "04_analysis"
    PLOT_DIR = OUT_DIR / "plots"
    SUMMARY_PARQUET = PARSED_DIR / "summary.parquet"
    METRICS_PARQUET = PARSED_DIR / "metrics.parquet"

    # Sanity check inputs
    for p in (SUMMARY_PARQUET, METRICS_PARQUET):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run the pipeline first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db)

    # Register the parquet files as views
    con.execute(f"CREATE OR REPLACE VIEW summary AS SELECT * FROM read_parquet('{SUMMARY_PARQUET.as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW metrics AS SELECT * FROM read_parquet('{METRICS_PARQUET.as_posix()}')")

    # Sanity check
    section("Dataset inventory")
    con.sql(dedent("""
        SELECT
            (SELECT COUNT(*) FROM summary) AS datasets_evaluated,
            (SELECT COUNT(*) FROM metrics) AS metric_observations,
            (SELECT COUNT(DISTINCT metric_identifier) FROM metrics) AS distinct_metrics,
            (SELECT MIN(fuji_timestamp) FROM summary) AS first_eval,
            (SELECT MAX(fuji_timestamp) FROM summary) AS last_eval,
            (SELECT ANY_VALUE(metric_version) FROM summary) AS metric_version
    """)).show(max_width=10_000)

    # Headline: overall FAIR % distribution
    run_and_capture(
        con,
        dedent("""
            SELECT
                COUNT(*)                                                          AS n_datasets,
                ROUND(MIN(fair_percent), 1)                                       AS min_pct,
                ROUND(QUANTILE_CONT(fair_percent, 0.25), 1)                       AS p25_pct,
                ROUND(MEDIAN(fair_percent), 1)                                    AS median_pct,
                ROUND(AVG(fair_percent), 1)                                       AS mean_pct,
                ROUND(QUANTILE_CONT(fair_percent, 0.75), 1)                       AS p75_pct,
                ROUND(QUANTILE_CONT(fair_percent, 0.90), 1)                       AS p90_pct,
                ROUND(MAX(fair_percent), 1)                                       AS max_pct,
                ROUND(STDDEV_SAMP(fair_percent), 1)                               AS sd_pct
            FROM summary
        """),
        "Overall FAIR% across all datasets",
        "headline.csv",
    )

    run_and_capture(
        con,
        dedent("""
            WITH bins AS (
                SELECT
                    CAST(FLOOR(fair_percent / 10) * 10 AS INTEGER) AS bin_low
                FROM summary
                WHERE fair_percent IS NOT NULL
            )
            SELECT
                bin_low                                AS pct_bin_low,
                bin_low + 10                           AS pct_bin_high,
                COUNT(*)                               AS n,
                REPEAT('█', CAST(COUNT(*) AS INTEGER)) AS bar
            FROM bins
            GROUP BY bin_low
            ORDER BY bin_low
        """),
        "FAIR% distribution (10-pp bins)",
        "histogram.csv",
        plot_fn=plot_histogram,
        plot_name="histogram",
    )

    # F A I R
    # UNPIVOT keeps the SQL terse
    # CASE in ORDER BY enforces the canonical F-A-I-R reading order
    run_and_capture(
        con,
        dedent("""
            WITH long AS (
                UNPIVOT summary
                ON f_percent, a_percent, i_percent, r_percent
                INTO NAME dim VALUE pct
            )
            SELECT
                UPPER(LEFT(dim, 1))                  AS principle,
                COUNT(pct)                           AS n,
                ROUND(AVG(pct), 1)                   AS mean_pct,
                ROUND(MEDIAN(pct), 1)                AS median_pct,
                ROUND(STDDEV_SAMP(pct), 1)           AS sd_pct,
                ROUND(MIN(pct), 1)                   AS min_pct,
                ROUND(MAX(pct), 1)                   AS max_pct
            FROM long
            GROUP BY principle
            ORDER BY CASE principle
                WHEN 'F' THEN 1 WHEN 'A' THEN 2 WHEN 'I' THEN 3 WHEN 'R' THEN 4
            END
        """),
        "F / A / I / R dimension scores",
        "dimension_gap.csv",
        plot_fn=plot_dimension_gap,
        plot_name="dimension_gap",
    )

    # Maturity by dimension is a complementary view
    run_and_capture(
        con,
        dedent("""
            SELECT
                fair_principle                       AS principle,
                COUNT(*)                             AS n_obs,
                ROUND(AVG(maturity), 2)              AS mean_maturity
            FROM metrics
            WHERE fair_principle IS NOT NULL
            GROUP BY fair_principle
            ORDER BY CASE fair_principle
                WHEN 'F' THEN 1 WHEN 'A' THEN 2 WHEN 'I' THEN 3 WHEN 'R' THEN 4
            END
        """),
        "Mean maturity by FAIR principle",
        None,
    )

    # Per-metric points earned
    # Ranked by `earned_pct`, which is what the headline score sums.
    run_and_capture(
        con,
        dedent("""
            SELECT
                metric_identifier,
                ANY_VALUE(metric_name)                                AS metric_name,
                ANY_VALUE(fair_principle)                             AS principle,
                COUNT(*)                                              AS n_evaluated,
                ROUND(100.0 * AVG(score_earned / NULLIF(score_total, 0)), 1)
                                                                      AS earned_pct,
                SUM(CASE WHEN score_earned >= score_total THEN 1 ELSE 0 END)
                                                                      AS n_full_points,
                ROUND(100.0 * AVG(CASE WHEN score_earned >= score_total
                                       THEN 1.0 ELSE 0.0 END), 1)     AS full_points_pct,
                ROUND(100.0 * AVG(CASE WHEN test_status = 'pass'
                                       THEN 1.0 ELSE 0.0 END), 1)     AS status_pass_pct,
                ROUND(AVG(maturity), 2)                               AS mean_maturity
            FROM metrics
            GROUP BY metric_identifier
            ORDER BY earned_pct ASC, metric_identifier
        """),
        "All metrics ranked by mean points earned (ascending; "
        "status_pass_pct = F-UJI's own metric verdict, which can differ)",
        "metric_pass_rates.csv",
        plot_fn=plot_metric_pass_rates,
        plot_name="metric_pass_rates",
    )

    # the 10 worst
    run_and_capture(
        con,
        dedent("""
            SELECT
                metric_identifier,
                ANY_VALUE(metric_name)                                AS metric_name,
                ANY_VALUE(fair_principle)                             AS principle,
                COUNT(*)                                              AS n_evaluated,
                ROUND(100.0 * AVG(score_earned / NULLIF(score_total, 0)), 1)
                                                                      AS earned_pct,
                ROUND(100.0 * AVG(CASE WHEN test_status = 'pass'
                                       THEN 1.0 ELSE 0.0 END), 1)     AS status_pass_pct
            FROM metrics
            GROUP BY metric_identifier
            ORDER BY earned_pct ASC, metric_identifier
            LIMIT 10
        """),
        "10 metrics with the fewest points earned",
        "worst_metrics.csv",
    )

    # Sub-dataverse breakdown for Dataverse repositories
    _title = "FAIR% by sub-dataverse"
    if has_subdataverse_info(con):
        run_and_capture(
            con,
            dedent("""
                SELECT
                    COALESCE(name_of_dataverse, '(unassigned)')   AS dataverse,
                    COUNT(*)                                      AS n_datasets,
                    ROUND(AVG(fair_percent), 1)                   AS mean_fair_pct,
                    ROUND(MEDIAN(fair_percent), 1)                AS median_fair_pct,
                    ROUND(MIN(fair_percent), 1)                   AS min_fair_pct,
                    ROUND(MAX(fair_percent), 1)                   AS max_fair_pct,
                    ROUND(STDDEV_SAMP(fair_percent), 1)           AS sd_fair_pct
                FROM summary
                GROUP BY dataverse
                ORDER BY mean_fair_pct DESC NULLS LAST
            """),
            _title,
            "dataverse_breakdown.csv",
            plot_fn=plot_dataverse_breakdown,
            plot_name="dataverse_breakdown",
        )
    else:
        section(_title)
        print("(skipped: no sub-dataverse membership in this repository's metadata)")

    # Time trend by publication year
    run_and_capture(
        con,
        dedent("""
            WITH yr AS (
                SELECT
                    fair_percent,
                    EXTRACT(year FROM TRY_CAST(published_at AS TIMESTAMP)) AS year
                FROM summary
            )
            SELECT
                year,
                COUNT(*)                                      AS n_datasets,
                ROUND(AVG(fair_percent), 1)                   AS mean_fair_pct,
                ROUND(MEDIAN(fair_percent), 1)                AS median_fair_pct
            FROM yr
            WHERE year IS NOT NULL
            GROUP BY year
            ORDER BY year
        """),
        "FAIR% by publication year",
        "year_trend.csv",
        plot_fn=plot_year_trend,
        plot_name="year_trend",
    )

    con.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo", required=True,
        help="Repository name - the data/<repo> subfolder to analyze.",
    )
    ap.add_argument(
        "--data-dir", default="data",
        help="Root data directory (default: data).",
    )
    ap.add_argument(
        "--db",
        default=":memory:",
        help="DuckDB database file (default: in-memory). "
             "Use a path like data/repo4cat.duckdb to persist for later CLI use.",
    )
    args = ap.parse_args()

    try:
        run(args.repo, args.data_dir, args.db)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
