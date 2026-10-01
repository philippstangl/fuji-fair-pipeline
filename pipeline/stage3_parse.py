"""Stage 3: Flatten F-UJI raw JSON into tables, per repository.

Outputs (under data/<repo name>/03_parsed/):
  summary.parquet       one row per dataset; headline FAIR score + per-letter scores
  metrics.parquet       one row per (dataset, metric_identifier); long format
  metric_tests.parquet  one row per (dataset, metric_test_identifier)
  parse_errors.json     any responses that couldn't be parsed (with reason)
"""
from __future__ import annotations

import json

import pandas as pd

from pipeline.common import (
    atomic_write_json,
    atomic_write_parquet,
    load_config,
    repo_paths,
    safe_filename,
    select_repos,
    setup_logging,
)
from pipeline.content import filter_manifest_by_content_class


def parse_all(repo: dict, cfg: dict | None = None) -> None:
    if cfg is None:
        cfg = load_config()
    log = setup_logging(f"stage3_parse_{repo['name']}", cfg["paths"]["logs_dir"])

    paths = repo_paths(cfg, repo)
    raw_dir = paths["evaluations"] / "raw"
    out_dir = paths["parsed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(paths["harvest"] / "manifest.parquet")
    manifest, out_of_scope = filter_manifest_by_content_class(
        manifest, repo.get("content_classes"), log, repo["name"]
    )
    manifest_by_pid = {r["pid"]: r for _, r in manifest.iterrows()}

    summary_rows: list[dict] = []
    metric_rows: list[dict] = []
    test_rows: list[dict] = []
    errors: list[dict] = []
    n_on_disk = sum(1 for _ in raw_dir.glob("*.json"))
    n_missing = 0

    # Driven by the manifest, not by the directory: an evaluation on disk whose
    # record is not in the (filtered) manifest is simply never asked for, so
    # excluded records stay out of the tables without anything being deleted.
    for pid, m_row in manifest_by_pid.items():
        if not pid or pd.isna(pid):
            continue
        json_path = raw_dir / f"{safe_filename(pid)}.json"
        if not json_path.exists():
            n_missing += 1
            continue
        try:
            with open(json_path) as f:
                resp = json.load(f)
            summary_rows.append(_summary_row(repo, pid, m_row, resp))
            metric_rows.extend(_metric_rows(repo, pid, resp))
            test_rows.extend(_metric_test_rows(repo, pid, resp))
        except Exception as e:
            log.error(f"Failed to parse {json_path.name}: {type(e).__name__}: {e}")
            errors.append({"file": json_path.name, "error": f"{type(e).__name__}: {e}"})

    sdf = pd.DataFrame(summary_rows)
    mdf = pd.DataFrame(metric_rows)
    tdf = pd.DataFrame(test_rows)

    # Type hygiene — coerce numerics so parquet has clean dtypes
    for col in ["fair_percent", "f_percent", "a_percent", "i_percent", "r_percent",
                "fair_earned", "fair_total"]:
        if col in sdf.columns:
            sdf[col] = pd.to_numeric(sdf[col], errors="coerce")
    for col in ["score_earned", "score_total", "metric_tests_passed", "metric_tests_total"]:
        if col in mdf.columns:
            mdf[col] = pd.to_numeric(mdf[col], errors="coerce")
    for col in ["metric_test_earned", "metric_test_total", "metric_test_maturity"]:
        if col in tdf.columns:
            tdf[col] = pd.to_numeric(tdf[col], errors="coerce")

    atomic_write_parquet(sdf, out_dir / "summary.parquet")
    atomic_write_parquet(mdf, out_dir / "metrics.parquet")
    atomic_write_parquet(tdf, out_dir / "metric_tests.parquet")
    if errors:
        atomic_write_json(errors, out_dir / "parse_errors.json")

    log.info(
        f"[{repo['name']}] Parsed {len(sdf)} summary rows, {len(mdf)} metric rows, "
        f"{len(tdf)} metric-test rows from {len(manifest_by_pid)} in-scope records "
        f"({n_on_disk} evaluation files on disk; {sum(out_of_scope.values())} out of "
        f"content scope, {n_missing} in scope but not evaluated). "
        f"{len(errors)} parse errors."
    )


def _summary_row(repo: dict, pid: str, manifest_row, resp: dict) -> dict:
    summary = resp.get("summary", {}) or {}
    pct = summary.get("score_percent", {}) or {}
    earned = summary.get("score_earned", {}) or {}
    total = summary.get("score_total", {}) or {}
    maturity = summary.get("maturity", {}) or {}

    return {
        "pid": pid,
        "global_id": pid,
        "repository_name": repo["name"],
        "repository_type": repo["type"],
        "name": _safe_get(manifest_row, "name"),
        "content_class": _safe_get(manifest_row, "content_class"),
        "name_of_dataverse": _safe_get(manifest_row, "name_of_dataverse"),
        "identifier_of_dataverse": _safe_get(manifest_row, "identifier_of_dataverse"),
        "published_at": _safe_get(manifest_row, "published_at"),
        "fair_percent": pct.get("FAIR"),
        "f_percent": pct.get("F"),
        "a_percent": pct.get("A"),
        "i_percent": pct.get("I"),
        "r_percent": pct.get("R"),
        "fair_earned": earned.get("FAIR"),
        "fair_total": total.get("FAIR"),
        "fair_maturity": maturity.get("FAIR"),
        "f_maturity": maturity.get("F"),
        "a_maturity": maturity.get("A"),
        "i_maturity": maturity.get("I"),
        "r_maturity": maturity.get("R"),
        "metric_version": resp.get("metric_version"),
        "fuji_timestamp": resp.get("timestamp"),
        "fuji_uid": resp.get("uid"),
    }


def _metric_rows(repo: dict, pid: str, resp: dict) -> list[dict]:
    out = []
    for r in resp.get("results", []) or []:
        score = r.get("score", {}) or {}
        identifier = r.get("metric_identifier") or ""
        # Identifier format is e.g. "FsF-F1-01D" — pull the letter after "FsF-"
        principle = identifier[4:5] if len(identifier) >= 5 else None
        if principle not in {"F", "A", "I", "R"}:
            principle = None

        metric_tests = r.get("metric_tests", {}) or {}
        out.append(
            {
                "pid": pid,
                "global_id": pid,
                "repository_name": repo["name"],
                "repository_type": repo["type"],
                "metric_identifier": identifier,
                "metric_name": r.get("metric_name"),
                "fair_principle": principle,
                "test_status": r.get("test_status"),
                "score_earned": score.get("earned"),
                "score_total": score.get("total"),
                "maturity": r.get("maturity"),
                "metric_tests_passed": _count_tests(metric_tests, "pass"),
                "metric_tests_total": len(metric_tests) if isinstance(metric_tests, dict) else None,
            }
        )
    return out


def _metric_test_rows(repo: dict, pid: str, resp: dict) -> list[dict]:
    """One row per sub-test F-UJI scored, e.g. "FsF-F2-01M-3" under "FsF-F2-01M"."""
    out = []
    for r in resp.get("results", []) or []:
        identifier = r.get("metric_identifier") or ""
        metric_tests = r.get("metric_tests")
        if not isinstance(metric_tests, dict):
            continue
        for test_id, t in metric_tests.items():
            if not isinstance(t, dict):
                continue
            score = t.get("metric_test_score", {}) or {}
            out.append(
                {
                    "pid": pid,
                    "global_id": pid,
                    "repository_name": repo["name"],
                    "repository_type": repo["type"],
                    "metric_identifier": identifier,
                    "metric_test_identifier": test_id,
                    "metric_test_name": t.get("metric_test_name"),
                    "metric_test_status": t.get("metric_test_status"),
                    "metric_test_earned": score.get("earned"),
                    "metric_test_total": score.get("total"),
                    "metric_test_maturity": t.get("metric_test_maturity"),
                }
            )
    return out


def _count_tests(metric_tests, want_status: str) -> int | None:
    if not isinstance(metric_tests, dict):
        return None
    n = 0
    for t in metric_tests.values():
        if isinstance(t, dict) and t.get("metric_test_status") == want_status:
            n += 1
    return n


def _safe_get(row, key, default=None):
    """Get a column from a dict-like or pandas Series, or return default."""
    if row is None:
        return default
    if hasattr(row, "get"):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


if __name__ == "__main__":
    _cfg = load_config()
    for _repo in select_repos(_cfg):
        parse_all(_repo, _cfg)
