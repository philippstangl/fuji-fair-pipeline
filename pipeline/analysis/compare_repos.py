"""Cross-repository comparison of F-UJI FAIR scores.

Reads each enabled stage 3 output (data/<repo>/03_parsed/) and writes
a combined comparison to the comparison directory (default data/_comparison/).

Outputs:
    repo_headline.csv            per-repo FAIR % distribution
    repo_dimension_gap.csv       per-repo mean F / A / I / R
    repo_metric_pass_rates.csv   per-repo x metric: mean points earned, full-points
                                 rate, and F-UJI status-pass rate (long)
    score_decomposition_by_repo.csv     per-repo platform/depositor shares + floors
    plots/fair_by_repo.{png,svg}        box plot of FAIR % per repository
    plots/dimension_by_repo.{png,svg}   grouped mean F/A/I/R per repository
    plots/metric_matrix.{png,svg}       repo x metric matrix of mean points earned
    plots/score_decomposition_by_repo.{png,svg}  stacked composition, all repos

Run from the project root:
    pip install duckdb matplotlib pyyaml
    python -m pipeline.analysis.compare_repos
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

import duckdb

from pipeline.common import (
    comparison_dir,
    load_config,
    repo_paths,
    select_repos,
)

# Below this many datasets, the spread statistics are
# descriptive rather than distributional
SMALL_N = 30

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.style.use(Path(__file__).with_name("paper.mplstyle"))
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    Patch = None  # type: ignore[assignment,misc]
    _HAS_MPL = False

# equals the LNI type-area width of 12.6 cm
TEXTWIDTH_IN = 12.6 / 2.54

_REPO_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]

# Paper cohort codes
_COHORT_CODES = {
    "nfdi4cat_dataverse": "DV",
    "nfdi4cat_zenodo": "Z-Cat",
    "nfdi4chem_zenodo": "Z-Chem",
    "chemotion_reactions": "Ch-R",
    "chemotion_collections": "Ch-C",
}


def _code(name: str) -> str:
    return _COHORT_CODES.get(name, name)

_BUCKET_COLORS = {"platform": "#4c72b0", "mixed": "#dd8452",
                  "depositor": "#55a868", "unclassified": "#888"}
_BUCKET_ORDER = ("platform", "depositor", "mixed", "unclassified")
_PRINCIPLE_COLORS = {"F": "#1f77b4", "A": "#2ca02c", "I": "#d62728", "R": "#9467bd"}
_PRINCIPLE_ORDER = ("F", "A", "I", "R")
_MATRIX_FILLS = {"zero": "#f0f0f0", "partial": "#c9d7ea", "full": "#4c72b0"}


def _section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def _run(con, sql: str, title: str, csv_path: Path | None) -> duckdb.DuckDBPyRelation:
    _section(title)
    rel = con.sql(sql)
    rel.show(max_width=10_000, max_rows=200)
    if csv_path is not None:
        con.sql(f"COPY ({sql}) TO '{csv_path.as_posix()}' (HEADER, DELIMITER ',')")
        print(f"wrote {csv_path}")
    return rel


def _plot_fair_by_repo(con, plot_dir: Path) -> str | None:
    if not _HAS_MPL:
        return None
    rows = con.sql(
        "SELECT repository_name, fair_percent FROM summary WHERE fair_percent IS NOT NULL"
    ).fetchall()
    groups: dict[str, list[float]] = defaultdict(list)
    for name, pct in rows:
        groups[name].append(pct)
    if not groups:
        return None
    # Mean-descending, same order as the dimension chart and the metric matrix
    names = sorted(groups, key=lambda k: -statistics.mean(groups[k]))
    data = [groups[k] for k in names]
    labels = []
    for k, vals in zip(names, data):
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        labels.append(f"{_code(k)}\n$n$ = {len(vals):,}\n"
                      f"{statistics.mean(vals):.1f} ± {sd:.1f}")

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.0), layout="constrained")
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("F-UJI score (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, plot_dir, "fair_by_repo")


def _plot_dimension_by_repo(rows: list[tuple], plot_dir: Path) -> str | None:
    """rows: (repository_name, n, mean_f, mean_a, mean_i, mean_r, mean_fair)."""
    if not _HAS_MPL or not rows:
        return None
    repos = [r[0] for r in rows]
    dims = ["F", "A", "I", "R"]
    per_repo = {r[0]: [r[2], r[3], r[4], r[5]] for r in rows}

    n_repos = len(repos)
    group_w = 0.8
    bar_w = group_w / max(n_repos, 1)
    x = list(range(len(dims)))

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.0), layout="constrained")
    for i, repo in enumerate(repos):
        offsets = [xi - group_w / 2 + bar_w * (i + 0.5) for xi in x]
        color = _REPO_COLORS[i % len(_REPO_COLORS)]
        bars = ax.bar(offsets, per_repo[repo], width=bar_w, label=_code(repo), color=color, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f", rotation=90, padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Mean score (%)")
    # Headroom for the rotated labels above the 100% bars
    ax.set_ylim(0, 118)
    ax.grid(axis="y", alpha=0.3)
    fig.legend(loc="outside lower center", ncol=len(repos), frameon=False)
    return _save(fig, plot_dir, "dimension_by_repo")


def _plot_metric_matrix(rows: list[tuple], repo_order: list[str],
                        plot_dir: Path) -> str | None:
    """Repo x metric matrix of mean points earned.

    rows: (repository_name, metric_identifier, principle, n_evaluated,
           earned_pct, full_points_pct, status_pass_pct)
    """
    if not _HAS_MPL or not rows:
        return None
    from matplotlib.patches import Rectangle

    rate = {(r[0], r[1]): r[4] for r in rows}  # earned_pct
    principle_of = {r[1]: r[2] for r in rows}
    repos = [name for name in repo_order if any(k[0] == name for k in rate)]
    blocks = [
        (p, sorted(m for m, pr in principle_of.items() if pr == p))
        for p in _PRINCIPLE_ORDER
    ]
    blocks = [(p, ms) for p, ms in blocks if ms]

    block_gap = 0.35
    y_of: dict[str, float] = {}
    y = 0.0
    for _, ms in blocks:
        for m in ms:
            y_of[m] = y
            y += 1.0
        y += block_gap
    total_h = y - block_gap

    # figure plus its multi-line caption must stay below the LNI text height
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 0.22 * len(y_of) + 1.2),
                           layout="constrained")
    for m, ym in y_of.items():
        for x, name in enumerate(repos):
            v = rate.get((name, m))
            if v is None:
                continue
            bin_ = "zero" if v == 0 else ("full" if v == 100 else "partial")
            ax.add_patch(Rectangle((x, ym), 1, 1, facecolor=_MATRIX_FILLS[bin_],
                                   edgecolor="white", linewidth=1.5))
            label = f"{v:.0f}" if v in (0.0, 100.0) else f"{v:.1f}"
            ax.text(x + 0.5, ym + 0.5, label, ha="center", va="center",
                    fontsize=8, color="white" if bin_ == "full" else "#222222")
    # Principle strip: one colored bar per block, left of the cells
    for p, ms in blocks:
        top, bottom = y_of[ms[0]], y_of[ms[-1]] + 1
        ax.add_patch(Rectangle((-0.42, top), 0.3, bottom - top,
                               facecolor=_PRINCIPLE_COLORS[p],
                               edgecolor="white", linewidth=1.5))
        ax.text(-0.27, (top + bottom) / 2, p, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

    ax.set_xlim(-0.45, len(repos))
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()
    ax.set_xticks([x + 0.5 for x in range(len(repos))])
    ax.set_xticklabels([_code(r) for r in repos], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks([ym + 0.5 for ym in y_of.values()])
    ax.set_yticklabels(list(y_of), fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    proxies = [
        Patch(facecolor=_MATRIX_FILLS["zero"], edgecolor="#cccccc",
              label="0 % of points"),
        Patch(facecolor=_MATRIX_FILLS["partial"], edgecolor="white",
              label="partial (0–100 %)"),
        Patch(facecolor=_MATRIX_FILLS["full"], edgecolor="white",
              label="100 % of points"),
    ]
    fig.legend(handles=proxies, loc="outside lower center",
               ncol=3, frameon=False, fontsize=8)
    return _save(fig, plot_dir, "metric_matrix")


def _load_decomposition_rows(cfg: dict, repos: list[dict],
                             dim_rows: list[tuple]) -> list[dict]:
    by_name = {r["name"]: r for r in repos}
    rows: list[dict] = []
    for dim in dim_rows:
        repo = by_name.get(dim[0])
        if repo is None:
            continue
        path = repo_paths(cfg, repo)["analysis"] / "decomposition_summary.csv"
        if not path.exists():
            print(f"(skipping {dim[0]} in decomposition figure: no {path}. "
                  "Run stage 4 analysis for it first)")
            continue
        with path.open(newline="") as fh:
            summary = next(csv.DictReader(fh))
        rows.append({
            "repository_name": dim[0],
            "n_records": int(summary["n_records"]),
            "mean_fair": float(dim[6]),
            "pct": {b: float(summary.get(f"pct_{b}", 0) or 0)
                    for b in _BUCKET_ORDER},
            "floor": {b: float(summary.get(f"floor_pp_{b}_excl_worst", 0) or 0)
                      for b in _BUCKET_ORDER},
        })
    return rows


def _plot_decomposition_by_repo(rows: list[dict], plot_dir: Path) -> str | None:
    """Relative 100%-stacked composition per repository."""
    if not _HAS_MPL or not rows:
        return None
    n = len(rows)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 0.5 * n + 0.7),
                           layout="constrained")
    for y, row in enumerate(rows):
        left = 0.0
        for b in _BUCKET_ORDER:
            v = row["pct"][b]
            if not v:
                continue
            color = _BUCKET_COLORS[b]
            ax.barh([y], [v], left=[left], color=color, edgecolor="white")
            fl = min(row["floor"][b], v)
            if fl > 0:
                ax.barh([y], [fl], left=[left], color=color, hatch="///",
                        edgecolor="white", linewidth=0.0)
                ax.plot([left + fl, left + fl], [y - 0.4, y + 0.4],
                        color="white", lw=1.5)
            # Below roughly 8% the label overflows its segment
            # drop the label if it does
            if v >= 8:
                ax.text(left + v / 2, y, f"{v:.1f}%", ha="center", va="center",
                        color="white", fontsize=9,
                        bbox=dict(facecolor=color, edgecolor="none",
                                  boxstyle="round,pad=0.25"))
            left += v
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{_code(r['repository_name'])} (n={r['n_records']})"
                        for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    proxies = [
        Patch(facecolor=_BUCKET_COLORS["platform"], edgecolor="white",
              label="platform"),
        Patch(facecolor=_BUCKET_COLORS["depositor"], edgecolor="white",
              label="depositor"),
        Patch(facecolor="#999", edgecolor="white", hatch="////", linewidth=0.0,
              label="earned by every deposit (invariant floor)"),
    ]
    # ncol=2: one row of all three entries is wider than the 12.6cm and would be clipped
    fig.legend(handles=proxies, loc="outside lower center",
               ncol=2, frameon=False, fontsize=8)
    return _save(fig, plot_dir, "score_decomposition_by_repo")


def _save(fig, plot_dir: Path, name: str) -> str:
    plot_dir.mkdir(parents=True, exist_ok=True)
    png = plot_dir / f"{name}.png"
    fig.savefig(png, dpi=150)
    fig.savefig(plot_dir / f"{name}.svg")
    fig.savefig(plot_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote {png} (+ .svg, .pdf)")
    return f"plots/{name}.png"


def run(cfg: dict | None = None) -> None:
    """Compare every repository that has stage-3 output."""
    if cfg is None:
        cfg = load_config()
    repos = select_repos(cfg)

    summary_paths: list[str] = []
    metrics_paths: list[str] = []
    present: list[str] = []
    for r in repos:
        p = repo_paths(cfg, r)
        sp = p["parsed"] / "summary.parquet"
        mp = p["parsed"] / "metrics.parquet"
        if sp.exists():
            summary_paths.append(sp.as_posix())
            present.append(r["name"])
            if mp.exists():
                metrics_paths.append(mp.as_posix())
        else:
            print(f"(skipping {r['name']}: no {sp}. Run stages 1-3 for it first)")

    if not summary_paths:
        raise FileNotFoundError("no parsed summaries found for any enabled repository.")

    out_dir = comparison_dir(cfg)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Comparing {len(present)} repositories: {present}")

    con = duckdb.connect(":memory:")
    summ_list = ", ".join(f"'{p}'" for p in summary_paths)
    con.execute(
        f"CREATE VIEW summary AS SELECT * FROM read_parquet([{summ_list}], union_by_name=true)"
    )
    if metrics_paths:
        met_list = ", ".join(f"'{p}'" for p in metrics_paths)
        con.execute(
            f"CREATE VIEW metrics AS SELECT * FROM read_parquet([{met_list}], union_by_name=true)"
        )

    present_repos = [r for r in repos if r["name"] in present]

    _run(
        con,
        dedent("""
            SELECT
                repository_name,
                COUNT(*)                                     AS n_datasets,
                ROUND(MIN(fair_percent), 1)                  AS min_pct,
                ROUND(QUANTILE_CONT(fair_percent, 0.25), 1)  AS p25_pct,
                ROUND(MEDIAN(fair_percent), 1)               AS median_pct,
                ROUND(AVG(fair_percent), 1)                  AS mean_pct,
                ROUND(QUANTILE_CONT(fair_percent, 0.75), 1)  AS p75_pct,
                ROUND(MAX(fair_percent), 1)                  AS max_pct,
                ROUND(STDDEV_SAMP(fair_percent), 1)          AS sd_pct
            FROM summary
            GROUP BY repository_name
            ORDER BY mean_pct DESC NULLS LAST
        """),
        "FAIR % per repository",
        out_dir / "repo_headline.csv",
    )

    small = con.sql(
        f"SELECT repository_name, COUNT(*) AS n FROM summary "
        f"GROUP BY repository_name HAVING COUNT(*) < {SMALL_N} ORDER BY n"
    ).fetchall()
    if small:
        listed = ", ".join(f"{name} (n={n})" for name, n in small)
        print(f"Small cohorts: {listed}. Below n={SMALL_N}.")

    dim_rows = _run(
        con,
        dedent("""
            SELECT
                repository_name,
                COUNT(*)                        AS n,
                ROUND(AVG(f_percent), 1)        AS mean_f,
                ROUND(AVG(a_percent), 1)        AS mean_a,
                ROUND(AVG(i_percent), 1)        AS mean_i,
                ROUND(AVG(r_percent), 1)        AS mean_r,
                ROUND(AVG(fair_percent), 1)     AS mean_fair
            FROM summary
            GROUP BY repository_name
            ORDER BY mean_fair DESC NULLS LAST
        """),
        "F / A / I / R by repository",
        out_dir / "repo_dimension_gap.csv",
    ).fetchall()

    metric_rows: list[tuple] = []
    if metrics_paths:
        metric_rows = _run(
            con,
            dedent("""
                SELECT
                    repository_name,
                    metric_identifier,
                    ANY_VALUE(fair_principle)                             AS principle,
                    COUNT(*)                                              AS n_evaluated,
                    ROUND(100.0 * AVG(score_earned / NULLIF(score_total, 0)), 1)
                                                                          AS earned_pct,
                    ROUND(100.0 * AVG(CASE WHEN score_earned >= score_total
                                           THEN 1.0 ELSE 0.0 END), 1)     AS full_points_pct,
                    ROUND(100.0 * AVG(CASE WHEN test_status = 'pass'
                                           THEN 1.0 ELSE 0.0 END), 1)     AS status_pass_pct
                FROM metrics
                GROUP BY repository_name, metric_identifier
                ORDER BY repository_name, earned_pct ASC, metric_identifier
            """),
            "Per-metric points earned by repository "
            "(earned_pct = mean share of points; status_pass_pct = F-UJI metric verdict)",
            out_dir / "repo_metric_pass_rates.csv",
        ).fetchall()

    decomp_rows = _load_decomposition_rows(cfg, present_repos, dim_rows)
    if decomp_rows:
        _section("Score decomposition by repository")
        decomp_csv = out_dir / "score_decomposition_by_repo.csv"
        fieldnames = ["repository_name", "n_records", "mean_fair",
                      "pct_platform", "pct_depositor",
                      "floor_pp_platform_excl_worst",
                      "floor_pp_depositor_excl_worst"]
        flat = [{
            "repository_name": r["repository_name"],
            "n_records": r["n_records"],
            "mean_fair": r["mean_fair"],
            "pct_platform": r["pct"]["platform"],
            "pct_depositor": r["pct"]["depositor"],
            "floor_pp_platform_excl_worst": r["floor"]["platform"],
            "floor_pp_depositor_excl_worst": r["floor"]["depositor"],
        } for r in decomp_rows]
        with decomp_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat)
        print(f"wrote {decomp_csv}")

    _section("Plots")
    _plot_fair_by_repo(con, plot_dir)
    _plot_dimension_by_repo(dim_rows, plot_dir)
    _plot_metric_matrix(metric_rows, [r[0] for r in dim_rows], plot_dir)
    _plot_decomposition_by_repo(decomp_rows, plot_dir)
    if not _HAS_MPL:
        print("matplotlib not installed. Skipping plots.")

    con.close()


def main() -> int:
    try:
        run()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
