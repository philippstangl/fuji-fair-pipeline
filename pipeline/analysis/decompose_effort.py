"""Decompose F-UJI results into 'platform-given' vs 'depositor-supplied'.

Run from the project root:
    pip install duckdb matplotlib pandas pyarrow
    python -m pipeline.analysis.decompose_effort --repo nfdi4cat_dataverse
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.common import load_config, safe_filename
from pipeline.content import filter_manifest_by_content_class

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _HAS_MPL = True
except ImportError:
    plt = None  # type: ignore[assignment]
    Patch = None  # type: ignore[assignment]
    _HAS_MPL = False

RAW_DIR: Path | None = None
PARSED: Path | None = None
OUT_DIR: Path | None = None
PLOT_DIR: Path | None = None

MANIFEST: pd.DataFrame | None = None

# Metric bucket classification
# Source: F-UJI metric catalogue at https://www.f-uji.net/index.php?action=metrics

METRIC_BUCKETS: dict[str, str] = {
    "FsF-F1-01MD":   "platform",   
    "FsF-F1-02MD":   "platform", 
    "FsF-F2-01M":    "mixed", 
    "FsF-F3-01M":    "platform", 
    "FsF-F4-01M":    "platform", 
    "FsF-A1-01M":    "depositor", 
    "FsF-A1-02MD":   "platform", 
    "FsF-A1.1-01MD": "platform", 
    "FsF-A1.2-01MD": "platform", 
    "FsF-I1-01M":    "platform", 
    "FsF-I2-01M":    "depositor", 
    "FsF-I3-01M":    "depositor", 
    "FsF-R1-01M":    "mixed", 
    "FsF-R1.1-01M":  "depositor", 
    "FsF-R1.2-01M":  "depositor", 
    "FsF-R1.3-01M":  "depositor", 
    "FsF-R1.3-02D":  "depositor",  
}

SUBTEST_BUCKETS: dict[str, dict[str, str]] = {
    "FsF-F2-01M": {
        "FsF-F2-01M-2": "platform",
        "FsF-F2-01M-3": "depositor",
        # NOTE: F-UJI 3.5.1 emits no FsF-F2-01M-1 (checked over all 3567
        # evaluations). Deliberately left unclassified: an unseen sub-test must
        # trip the coverage guard
    },
    "FsF-R1-01M": {
        "FsF-R1-01M-1": "platform",
        "FsF-R1-01M-2": "platform",
        # F-UJI scores this 0/0, so it carries no points either way.
        # But classified so that coverage passes rather than falling the
        # whole metric back to "mixed" over a zero-weight test.
        "FsF-R1-01M-3": "depositor",
    },
}

MAX_OF_ALTERNATIVES: tuple[str, ...] = (
    "FsF-I1-01M", "FsF-I3-01M", "FsF-R1.2-01M", "FsF-R1.3-01M",
)

BUCKET_ORDER: tuple[str, ...] = ("platform", "depositor", "mixed", "unclassified")

ATTRIBUTION_NOTES: dict[str, str] = {
    "FsF-R1-01M-2": (
        "tool artifact, not effort: passes iff the deposit has exactly one file "
        "on both Zenodo cohorts (21/21, perfect separation); 0% on both Chemotion "
        "cohorts, whose JSON-LD exposes no file listing"
    ),
    "FsF-R1.3-02D": (
        "contestable: 27 of 80 Dataverse records earn this only via "
        "text/tab-separated-values, the tabular-ingest derivative rather than a "
        "depositor upload; see would_earn_without_tab in content_signals.csv"
    ),
    "FsF-R1-01M-3": "scored 0/0 by F-UJI — carries no points either way",
    "FsF-F2-01M-2": "invariant: 99.9-100% across all five cohorts",
}


def _section(title: str) -> None:
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")


def _splittable_metrics(
    metrics: pd.DataFrame, tests: pd.DataFrame | None
) -> tuple[set[str], pd.DataFrame]:
    """Decide which metrics in SUBTEST_BUCKETS may be replaced by their sub-tests."""
    audit_rows = []
    splittable: set[str] = set()

    for mid, sub in SUBTEST_BUCKETS.items():
        m = metrics[metrics["metric_identifier"] == mid]
        if m.empty:
            continue
        row = {
            "metric_identifier": mid, "n_records": len(m), "n_ok": 0,
            "n_not_additive": 0, "n_unclassified_subtest": 0, "n_no_tests": 0,
            "split": False, "first_failures": "",
        }
        if tests is None or tests.empty:
            row["n_no_tests"] = len(m)
            audit_rows.append(row)
            continue

        t = tests[tests["metric_identifier"] == mid]
        agg = t.groupby("global_id").agg(
            sum_total=("metric_test_total", "sum"),
            sum_earned=("metric_test_earned", "sum"),
            n_tests=("metric_test_identifier", "size"),
            n_unclassified=("metric_test_identifier",
                            lambda s: int((~s.isin(sub)).sum())),
        )
        joined = m.set_index("global_id").join(agg, how="left")
        has_tests = joined["n_tests"].notna() & (joined["n_tests"] > 0)
        # np.isclose, not ==: FsF-F1-02MD genuinely scores 0.5 per sub-test.
        additive = has_tests & (
            np.isclose(joined["sum_total"].fillna(-1), joined["score_total"])
            & np.isclose(joined["sum_earned"].fillna(-1), joined["score_earned"])
        )
        covered = has_tests & (joined["n_unclassified"].fillna(1) == 0)
        ok = additive & covered

        row["n_ok"] = int(ok.sum())
        row["n_no_tests"] = int((~has_tests).sum())
        row["n_not_additive"] = int((has_tests & ~additive).sum())
        row["n_unclassified_subtest"] = int((has_tests & ~covered).sum())
        row["split"] = bool(ok.all()) and len(joined) > 0
        if row["split"]:
            splittable.add(mid)
        else:
            failures = list(joined.index[~ok])[:5]
            row["first_failures"] = "; ".join(str(f) for f in failures)
            print(
                f"WARNING: {mid} not split. The sub-test additivity guard failed "
                f"for {len(joined) - row['n_ok']} of {len(joined)} records "
                f"({row['n_not_additive']} non-additive, "
                f"{row['n_unclassified_subtest']} with an unclassified sub-test, "
                f"{row['n_no_tests']} with no sub-tests). Falling back to the "
                f"whole-metric bucket ('{METRIC_BUCKETS.get(mid)}') for the entire "
                f"cohort, so the shares below are the pre-split numbers.\n"
                f"First failures: {row['first_failures'] or '(none)'}"
            )
        audit_rows.append(row)

    return splittable, pd.DataFrame(audit_rows)


def scoring_units(
    metrics: pd.DataFrame, tests: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve each score into attributed units."""
    splittable, audit = _splittable_metrics(metrics, tests)

    whole = metrics[~metrics["metric_identifier"].isin(splittable)].copy()
    whole["unit_id"] = whole["metric_identifier"]
    whole["grain"] = "metric"
    whole["split_from"] = ""
    whole["bucket"] = whole["metric_identifier"].map(METRIC_BUCKETS).fillna("unclassified")
    whole = whole.rename(columns={"score_earned": "earned", "score_total": "total",
                                  "metric_name": "unit_name"})

    unclass = whole.loc[whole["bucket"] == "unclassified", "unit_id"].unique()
    if len(unclass):
        print(f"WARNING: {len(unclass)} metric IDs not in METRIC_BUCKETS "
              f"and will be counted as 'unclassified': {list(unclass)}")

    cols = ["global_id", "unit_id", "grain", "split_from", "bucket",
            "earned", "total", "unit_name"]
    frames = [whole[cols]]

    if splittable and tests is not None and not tests.empty:
        part = tests[tests["metric_identifier"].isin(splittable)].copy()
        part["unit_id"] = part["metric_test_identifier"]
        part["grain"] = "metric_test"
        part["split_from"] = part["metric_identifier"]
        part["bucket"] = [
            SUBTEST_BUCKETS[mid][tid]
            for mid, tid in zip(part["metric_identifier"], part["metric_test_identifier"])
        ]
        part = part.rename(columns={"metric_test_earned": "earned",
                                    "metric_test_total": "total",
                                    "metric_test_name": "unit_name"})
        frames.append(part[cols])

    units = pd.concat(frames, ignore_index=True)

    # Splitting must redistribute points, never create or destroy them.
    before = metrics.groupby("global_id")["score_earned"].sum()
    after = units.groupby("global_id")["earned"].sum()
    drift = (after - before).abs()
    bad = list(drift.index[drift > 1e-9])[:5]
    if bad:
        print(f"WARNING: earned score changed for {len(bad)}+ records while "
              f"splitting into units. This is a bug! First: {bad}")

    return units, audit


def score_decomposition(units: pd.DataFrame) -> pd.DataFrame:
    """Sum earned/total per bucket per dataset, derive % contributions."""
    grouped = (
        units.groupby(["global_id", "bucket"], dropna=False)[["earned", "total"]]
        .sum()
        .unstack(fill_value=0)
    )
    # Flatten the MultiIndex columns into score_<stat>_<bucket> strings
    grouped.columns = [f"score_{'earned' if stat == 'earned' else 'total'}_{bucket}"
                       for stat, bucket in grouped.columns]
    grouped = grouped.reset_index()

    # Derive percentages and guard against divide-by-zero with a fillna
    earned_total = grouped.filter(like="score_earned_").sum(axis=1)
    grouped["earned_total"] = earned_total
    for b in BUCKET_ORDER:
        col = f"score_earned_{b}"
        if col in grouped.columns:
            grouped[f"pct_of_earned_{b}"] = (
                100.0 * grouped[col] / earned_total.replace(0, pd.NA)
            ).fillna(0).round(1)

    n_split = units.loc[units["grain"] == "metric_test", "unit_id"].nunique()
    n_fallback = units.loc[
        units["unit_id"].isin(SUBTEST_BUCKETS), "unit_id"
    ].nunique()
    grouped["n_units_split"] = n_split
    grouped["n_units_fallback"] = n_fallback
    return grouped


def summarize_decomposition(decomp: pd.DataFrame) -> dict[str, float]:
    """Roll up to a single set of repository-wide percentages."""
    total_earned_by_bucket = {
        b: decomp.get(f"score_earned_{b}", pd.Series([0])).sum()
        for b in BUCKET_ORDER
    }
    total = sum(total_earned_by_bucket.values()) or 1
    return {b: round(100.0 * v / total, 1) for b, v in total_earned_by_bucket.items()}


def _unit_matrix(units: pd.DataFrame) -> pd.DataFrame:
    dupes = units.duplicated(["global_id", "unit_id"])
    if dupes.any():
        print(f"WARNING: {int(dupes.sum())} duplicate (global_id, unit_id) rows "
              f"will be summed together in the floor computation.")
    # Cast first: fill_value is a float, and pandas warns (soon raises) when it
    # cannot be held in an integer column dtype
    earned = units.assign(earned=units["earned"].astype(float))
    return earned.pivot_table(index="global_id", columns="unit_id", values="earned",
                              aggfunc="sum", fill_value=0.0)


def bucket_evidence(units: pd.DataFrame, repo_name: str,
                    audit: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per scoring unit: how much it contributes, and how much of that is a floor
    every record clears versus genuinely varying between deposits."""
    mat = _unit_matrix(units)
    n = len(mat)
    grand = float(mat.values.sum()) or 1.0

    meta = (units.groupby("unit_id")
            .agg(bucket=("bucket", "first"), grain=("grain", "first"),
                 split_from=("split_from", "first"), unit_name=("unit_name", "first"),
                 points_total_max=("total", "max"), points_total_min=("total", "min")))
    fails = {}
    if audit is not None and not audit.empty:
        for _, a in audit.iterrows():
            if not a["split"]:
                fails[a["metric_identifier"]] = int(a["n_records"] - a["n_ok"])

    rows = []
    for unit_id in mat.columns:
        col = mat[unit_id]
        m = meta.loc[unit_id]
        points = float(m["points_total_max"] or 0)
        floor = float(col.min())
        mean = float(col.mean())
        # p_earn is measured against the unit's own total, not the observed max:
        # a unit nobody earns (FsF-I2-01M, FsF-R1-01M-3) would otherwise report
        # p_earn = 1.0 because every record ties for the maximum of zero.
        p_earn = float(np.isclose(col, points).mean()) if points > 0 else 0.0
        rows.append({
            "repository_name": repo_name,
            "bucket": m["bucket"],
            "unit_id": unit_id,
            "grain": m["grain"],
            "split_from": m["split_from"],
            "unit_name": m["unit_name"],
            "points_total": points,
            "points_total_varies": bool(m["points_total_max"] != m["points_total_min"]),
            "n_records": n,
            "n_earning_full": int(np.isclose(col, points).sum()) if points > 0 else 0,
            "p_earn": round(p_earn, 4),
            "min_earned": floor,
            "median_earned": float(col.median()),
            "max_earned": float(col.max()),
            "mean_earned": round(mean, 4),
            "floor_earned": floor,
            "variable_earned": round(mean - floor, 4),
            "sum_earned": float(col.sum()),
            "pct_of_cohort_earned": round(100.0 * col.sum() / grand, 2),
            "pct_floor_of_cohort_earned": round(100.0 * floor * n / grand, 2),
            "pct_variable_of_cohort_earned": round(100.0 * (mean - floor) * n / grand, 2),
            "n_guard_failures": fails.get(unit_id, 0),
            "attribution_note": ATTRIBUTION_NOTES.get(unit_id, ""),
        })
    ev = pd.DataFrame(rows)
    order = {b: i for i, b in enumerate(BUCKET_ORDER)}
    ev["_o"] = ev["bucket"].map(order).fillna(len(order))
    return (ev.sort_values(["_o", "pct_of_cohort_earned"], ascending=[True, False])
            .drop(columns="_o").reset_index(drop=True))


def bucket_floors(units: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per bucket, the share of cohort earned score that is an invariant floor."""
    mat = _unit_matrix(units)
    n = len(mat)
    if n == 0:
        return {"floor": {}, "floor_excl_worst": {}, "worst": {}}
    bucket_of = units.groupby("unit_id")["bucket"].first()

    def _floors(sub: pd.DataFrame) -> dict[str, float]:
        grand = float(sub.values.sum()) or 1.0
        rows = len(sub)
        out: dict[str, float] = dict.fromkeys(BUCKET_ORDER, 0.0)
        for unit_id in sub.columns:
            b = bucket_of.get(unit_id, "unclassified")
            out[b] = out.get(b, 0.0) + 100.0 * float(sub[unit_id].min()) * rows / grand
        return {k: round(v, 1) for k, v in out.items()}

    per_record = mat.sum(axis=1)
    worst_pid = per_record.idxmin()
    excl = mat.drop(index=worst_pid) if n > 1 else mat
    return {
        "floor": _floors(mat),
        "floor_excl_worst": _floors(excl),
        "worst": {"pid": str(worst_pid), "earned": float(per_record.min())},
    }


# Content effort signals from raw harvested metadata

def _coalesce_metadata(raw: dict) -> dict:
    """F-UJI harvests metadata from multiple sources (json_in_html, meta_tag,
    signposting, etc.). Each source contains overlapping fields. To measure
    depositor effort, take the *richest* value for each field across sources.
    If any source has a 200-word description and another has 20 words,
    credit the depositor for the 200.
    """
    sources = raw.get("harvested_metadata") or []
    combined: dict[str, Any] = {}
    for src in sources:
        md = src.get("metadata", {})
        if not isinstance(md, dict):
            continue
        for k, v in md.items():
            if v is None:
                continue
            # Prefer longer string values; prefer larger lists.
            if k not in combined:
                combined[k] = v
            else:
                old = combined[k]
                if isinstance(v, str) and isinstance(old, str) and len(v) > len(old):
                    combined[k] = v
                elif isinstance(v, list) and isinstance(old, list) and len(v) > len(old):
                    combined[k] = v
    return combined


def _safe_len(v) -> int:
    if v is None:
        return 0
    if isinstance(v, str):
        return len(v.strip())
    if isinstance(v, list):
        return len(v)
    return 0


def _keyword_diversity(kw) -> int:
    """Number of distinct, non-generic keywords. 'Chemistry' alone is the
    Dataverse default for this repository."""
    if not isinstance(kw, list):
        return 0
    cleaned = {str(k).strip().lower() for k in kw if str(k).strip()}
    return len(cleaned)


def _has_publication_link(related) -> bool:
    """Is at least one related_resource a publication?

    A real publication link can show up several ways:
      - relation_type contains "citation" or "isReferencedBy"
      - URL contains a DOI pattern: doi.org/..., /doi/10..., or just "10."
    """
    if not isinstance(related, list):
        return False
    for r in related:
        if not isinstance(r, dict):
            continue
        url = str(r.get("related_resource", "")).lower()
        rel = str(r.get("relation_type", "")).lower()
        if "doi.org" in url or "/doi/10." in url:
            return True
        if "citation" in rel or "isreferencedby" in rel:
            return True
    return False


# Dataverse ingests tabular uploads (.csv, .xlsx, etc.) and serves a derived .tab
_INGEST_DERIVED_MIME = "text/tab-separated-values"


def _would_earn_without_tab(raw: dict) -> bool | None:
    """Would FsF-R1.3-02D still be earned if the ingest-derived .tab were discounted?

    Returns None when the metric was not earned at all, so records that never had
    the point are not counted as evidence either way.
    """
    for res in raw.get("results", []) or []:
        if res.get("metric_identifier") != "FsF-R1.3-02D":
            continue
        if not (res.get("score", {}) or {}).get("earned"):
            return None
        out = res.get("output") or []
        if not isinstance(out, list):
            return None
        preferred = {
            o.get("mime_type") for o in out
            if isinstance(o, dict) and o.get("is_preferred_format")
        }
        return bool(preferred - {_INGEST_DERIVED_MIME})
    return None


def content_signals() -> pd.DataFrame:
    """For each in-scope evaluated dataset, extract content-level effort proxies."""
    rows = []
    if MANIFEST is None:
        paths = sorted(RAW_DIR.glob("*.json"))
    else:
        paths = [
            RAW_DIR / f"{safe_filename(pid)}.json"
            for pid in MANIFEST["pid"]
            if pid and not pd.isna(pid)
        ]
    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path) as f:
                raw = json.load(f)
        except Exception as e:
            print(f"skipped {path.name}: {e}")
            continue
        doi = raw.get("request", {}).get("object_identifier", path.stem)
        md = _coalesce_metadata(raw)
        rows.append({
            "global_id": doi,
            "harvested_ok": bool(md),
            "title_len": _safe_len(md.get("title")),
            "summary_len": _safe_len(md.get("summary")),
            "n_keywords": _keyword_diversity(md.get("keywords")),
            "keywords_only_generic": (
                isinstance(md.get("keywords"), list)
                and {str(k).strip().lower() for k in md["keywords"]} == {"chemistry"}
            ),
            "n_creators": _safe_len(md.get("creator")),
            "has_license": _safe_len(md.get("license")) > 0,
            "has_publication_link": _has_publication_link(md.get("related_resources")),
            "n_related_resources": _safe_len(md.get("related_resources")),
            "n_data_files": _safe_len(md.get("object_content_identifier")),
            # Sensitivity check on the one contestable attribution
            "would_earn_without_tab": _would_earn_without_tab(raw),
        })
    return pd.DataFrame(rows)


# Report

def _metric_version() -> str:
    """The F-UJI metric spec version these results were scored against — it lives
    in summary.parquet, and pins what the bucketing was argued against."""
    try:
        s = pd.read_parquet(PARSED / "summary.parquet", columns=["metric_version"])
        vals = s["metric_version"].dropna()
        return str(vals.iloc[0]) if len(vals) else ""
    except Exception:
        return ""


def _decomposition_summary(repo_name: str, units: pd.DataFrame,
                           summary: dict[str, float], floors: dict,
                           audit: pd.DataFrame, metric_version: str,
                           subtest_split: bool) -> pd.DataFrame:
    """One row per repository — the headline claim and everything needed to audit it."""
    split_metrics = sorted(units.loc[units["grain"] == "metric_test",
                                     "split_from"].unique())
    n_fail = 0
    if audit is not None and not audit.empty:
        n_fail = int((audit["n_records"] - audit["n_ok"])[~audit["split"]].sum())
    row = {
        "repository_name": repo_name,
        "n_records": units["global_id"].nunique(),
        "cohort_earned": float(units["earned"].sum()),
        "cohort_total": float(units["total"].sum()),
    }
    for b in BUCKET_ORDER:
        row[f"pct_{b}"] = summary.get(b, 0.0)
    for b in ("platform", "depositor"):
        row[f"floor_pp_{b}"] = floors["floor"].get(b, 0.0)
        row[f"variable_pp_{b}"] = round(
            summary.get(b, 0.0) - floors["floor"].get(b, 0.0), 1)
        row[f"floor_pp_{b}_excl_worst"] = floors["floor_excl_worst"].get(b, 0.0)
    row.update({
        "worst_record_pid": floors["worst"].get("pid", ""),
        "worst_record_earned": floors["worst"].get("earned", 0.0),
        "subtest_split": bool(subtest_split and split_metrics),
        "metrics_split": ";".join(split_metrics),
        "n_guard_failures": n_fail,
        "fuji_metric_version": metric_version,
    })
    return pd.DataFrame([row])


def _write_csv(df: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(f"wrote {path}")


def _plot_score_decomposition(summary: dict[str, float],
                              floors: dict[str, float] | None = None,
                              note: str = "") -> Path | None:
    if not _HAS_MPL:
        return None
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    floors = floors or {}
    # Filter to non-zero buckets only, in a fixed order. With the sub-test split
    # in place "mixed" is normally empty and simply drops out here.
    buckets = [b for b in BUCKET_ORDER if summary.get(b, 0)]
    values = [summary[b] for b in buckets]
    colors = {"platform": "#4c72b0", "mixed": "#dd8452",
              "depositor": "#55a868", "unclassified": "#888"}
    fig, ax = plt.subplots(figsize=(8, 2.1))
    left = 0
    for b, v in zip(buckets, values):
        ax.barh([0], [v], left=[left], color=colors[b], edgecolor="white")
        fl = min(float(floors.get(b, 0.0)), v)
        if fl > 0:
            ax.barh([0], [fl], left=[left], color=colors[b], hatch="///",
                    edgecolor="white", linewidth=0.0)
            ax.plot([left + fl, left + fl], [-0.4, 0.4], color="white", lw=1.5)
        # Below approx 8 an in-bar two-line label overflows its segment; annotate above.
        if v >= 8:
            # The label usually lands on hatched ground, so give it a solid patch
            # of its own colour to sit on, otherwise the white bold text is read
            # through the stripes.
            ax.text(left + v / 2, 0, f"{b}\n{v}%", ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold",
                    bbox=dict(facecolor=colors[b], edgecolor="none",
                              boxstyle="round,pad=0.35"))
        else:
            ax.annotate(f"{b} {v}%", xy=(left + v / 2, 0.42),
                        xytext=(left + v / 2, 0.95), ha="center", fontsize=8,
                        arrowprops=dict(arrowstyle="-", lw=0.6))
        left += v
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    caption = "Share of total earned F-UJI score (%)"
    if floors:
        parts = [f"{b} {floors.get(b, 0.0):.1f} pp of {summary.get(b, 0.0):.1f}"
                 for b in buckets if summary.get(b, 0)]
        caption += "\ninvariant floor: " + " · ".join(parts)
    if note:
        caption += f" · {note}"
    ax.set_xlabel(caption, fontsize=9)
    ax.set_title("Where the F-UJI score comes from: platform vs. depositor")
    proxies = [
        Patch(facecolor="#999", edgecolor="white", label="varies across deposits"),
        Patch(facecolor="#999", edgecolor="white", hatch="////", linewidth=0.0,
              label="earned by every deposit (invariant floor)"),
    ]
    ax.legend(handles=proxies, loc="upper center", bbox_to_anchor=(0.5, -0.45),
              ncol=2, frameon=False, fontsize=8)
    out = PLOT_DIR / "score_decomposition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (+ .svg, .pdf)")
    return out


def _plot_content_signals(content: pd.DataFrame) -> Path | None:
    if not _HAS_MPL:
        return None
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(content)
    if n == 0:
        return None

    # Six effort proxies, normalised to "share of datasets exhibiting effort"
    has_keywords = content["n_keywords"] >= 1
    has_meaningful_keywords = has_keywords & (~content["keywords_only_generic"])

    proxies = [
        ("Summary > 200 chars",                  (content["summary_len"] > 200).mean()),
        ("≥ 3 keywords",                         (content["n_keywords"] >= 3).mean()),
        ("Keywords beyond 'Chemistry'",          has_meaningful_keywords.mean()),
        ("License chosen",                       content["has_license"].mean()),
        ("Linked to publication",                content["has_publication_link"].mean()),
        ("≥ 1 data file attached",               (content["n_data_files"] >= 1).mean()),
    ]
    labels, fractions = zip(*proxies)
    pcts = [round(f * 100, 1) for f in fractions]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels, pcts, color="#55a868", edgecolor="white")
    ax.set_xlim(0, 115)
    ax.set_xlabel(f"Share of datasets (n={n}, %)")
    ax.set_title("Depositor effort: content-level signals from harvested metadata")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    for bar, p in zip(bars, pcts):
        ax.text(p + 1, bar.get_y() + bar.get_height() / 2, f"{p}%",
                va="center", fontsize=9)
    out = PLOT_DIR / "content_signals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (+ .svg, .pdf)")
    return out


def _load_scoped_manifest(base: Path, repo_name: str) -> pd.DataFrame | None:
    """The manifest restricted to the repository's in-scope content classes.

    Returns None when the manifest or the config can't be read, in which case
    Part B falls back to globbing the raw directory — the pre-classification
    behaviour, so the script stays usable standalone.
    """
    manifest_path = base / "01_harvest" / "manifest.parquet"
    if not manifest_path.exists():
        print(f"(no {manifest_path}; content signals will cover every evaluation)")
        return None
    try:
        cfg = load_config()
        repo = next(
            (r for r in cfg.get("repositories") or [] if r.get("name") == repo_name), {}
        )
        classes = repo.get("content_classes")
    except Exception as e:
        print(f"(could not read config ({type(e).__name__}); no content-class filter)")
        classes = None

    log = logging.getLogger("decompose_effort")
    if not log.handlers:
        log.addHandler(logging.StreamHandler())
        log.setLevel(logging.INFO)
    manifest = pd.read_parquet(manifest_path)
    scoped, _ = filter_manifest_by_content_class(manifest, classes, log, repo_name)
    return scoped


def run(repo: str, data_dir: str = "data") -> None:
    """Decompose one repository's effort signals. Called by stage 4 and by main()."""
    # Resolve per-repository paths. Reassigning the module-level constants keeps
    # the helper functions (which read them) unchanged.
    global RAW_DIR, PARSED, OUT_DIR, PLOT_DIR, MANIFEST
    base = Path(data_dir) / repo
    RAW_DIR = base / "02_evaluations" / "raw"
    PARSED = base / "03_parsed"
    OUT_DIR = base / "04_analysis"
    PLOT_DIR = OUT_DIR / "plots"
    MANIFEST = _load_scoped_manifest(base, repo)

    if not (PARSED / "metrics.parquet").exists():
        raise FileNotFoundError(f"{PARSED}/metrics.parquet not found. Run the pipeline first.")
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"{RAW_DIR} not found. Run stage 2 of the pipeline first.")

    _section("A. Score decomposition: platform vs. depositor")
    metrics = pd.read_parquet(PARSED / "metrics.parquet")
    tests_path = PARSED / "metric_tests.parquet"
    note = ""
    if tests_path.exists():
        tests = pd.read_parquet(tests_path)
    else:
        tests = None
        note = "sub-test split unavailable"
        print(f"WARNING: {tests_path} not found. Falling back to whole-metric ")

    units, audit = scoring_units(metrics, tests)
    decomp = score_decomposition(units)
    summary = summarize_decomposition(decomp)
    floors = bucket_floors(units)
    evidence = bucket_evidence(units, repo, audit)

    split_metrics = sorted(units.loc[units["grain"] == "metric_test",
                                     "split_from"].unique())
    print(f"\nSplit at sub-test grain: {', '.join(split_metrics) or '(none)'}")
    print("\nRepository-wide split of total earned score:")
    for bucket, pct in summary.items():
        if pct:
            fl = floors["floor"].get(bucket, 0.0)
            flx = floors["floor_excl_worst"].get(bucket, 0.0)
            print(f"{bucket:14s} {pct:5.1f}%   invariant floor {fl:5.1f} pp "
                  f"({flx:5.1f} pp excluding the single lowest-scoring record)")
    print(f" lowest-scoring record: {floors['worst']['pid']} "
          f"(earned {floors['worst']['earned']:g})")

    _write_csv(decomp, "score_decomposition.csv")
    _write_csv(evidence, "bucket_evidence.csv")
    _write_csv(_decomposition_summary(repo, units, summary, floors, audit,
                                      _metric_version(), tests is not None),
               "decomposition_summary.csv")
    _plot_score_decomposition(summary, floors["floor_excl_worst"], note)

    _section("B. Content effort signals from harvested metadata")
    content = content_signals()
    if content.empty:
        print("(no harvested metadata available)")
        return

    print(f"\nAcross {len(content)} datasets:")
    print(f" median summary length:     {int(content['summary_len'].median()):>6} chars")
    print(f" max summary length:        {int(content['summary_len'].max()):>6} chars")
    print(f" median # keywords:         {int(content['n_keywords'].median()):>6}")
    print(f" share with only 'Chemistry' as keyword: "
          f"{content['keywords_only_generic'].mean() * 100:.1f}%")
    print(f" share with publication-DOI link:        "
          f"{content['has_publication_link'].mean() * 100:.1f}%")
    print(f" share with >= 1 data file:              "
          f"{(content['n_data_files'] >= 1).mean() * 100:.1f}%")
    earned_fmt = content["would_earn_without_tab"].dropna()
    if len(earned_fmt):
        print(f" of the {len(earned_fmt)} earning FsF-R1.3-02D, share that would "
              f"still earn it\n without the platform .tab derivative:  "
              f"{earned_fmt.mean() * 100:.1f}%")
    _write_csv(content, "content_signals.csv")
    _plot_content_signals(content)


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
    args = ap.parse_args()

    try:
        run(args.repo, args.data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
