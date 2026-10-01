"""Stage 1 — Harvest dataset/record metadata for one repository.

Each repository is harvested independently into its own tree. Outputs (under
data/<repo name>/01_harvest/):
  raw_pages/page_<n>.json   immutable raw API responses, one per fetched page
  manifest.parquet          normalized record list (one row per pid)
  harvest_meta.json         counts, timestamp, source URL/scope, label

The API-specific fetching and field mapping live in pipeline/harvesters/; this
stage only wires a harvester to per-repository persistence.
"""
from __future__ import annotations

import pandas as pd

from pipeline.common import (
    atomic_write_json,
    atomic_write_parquet,
    load_config,
    repo_paths,
    select_repos,
    setup_logging,
    utc_now_iso,
)
from pipeline.harvesters import MANIFEST_COLUMNS, get_harvester


def harvest(repo: dict, cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    log = setup_logging(f"stage1_harvest_{repo['name']}", cfg["paths"]["logs_dir"])

    harvest_dir = repo_paths(cfg, repo)["harvest"]
    raw_dir = harvest_dir / "raw_pages"
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[{repo['name']}] Harvesting {repo['type']} from {repo['base_url']}")
    harvester = get_harvester(repo, log)
    raw_pages, rows = harvester.harvest()
    filter_stats = harvester.filter_stats

    for i, body in enumerate(raw_pages):
        atomic_write_json(body, raw_dir / f"page_{i:06d}.json")

    # File-index pages (Dataverse) go in their own directory, not raw_pages/:
    # mixing them into that numbered sequence would break anything re-reading it.
    for i, body in enumerate(getattr(harvester, "raw_file_pages", []) or []):
        atomic_write_json(body, harvest_dir / "raw_files" / f"page_{i:06d}.json")

    # Per-record documents (Chemotion), for backends whose listing endpoint
    # returns identifiers only and whose metadata arrives one record at a time.
    # Same reasoning as above: raw_pages/ means "listing responses", so these
    # get their own sequence. The harvester batches them; one file per record
    # would put thousands of tiny files in the tree.
    for i, body in enumerate(getattr(harvester, "raw_record_pages", []) or []):
        atomic_write_json(body, harvest_dir / "raw_records" / f"page_{i:06d}.json")

    df = _build_manifest(rows, log)
    atomic_write_parquet(df, harvest_dir / "manifest.parquet")
    log.info(f"[{repo['name']}] Wrote manifest: {len(df)} unique records")

    atomic_write_json(
        {
            "harvested_at": utc_now_iso(),
            "repository_name": repo["name"],
            "repository_type": repo["type"],
            "source_url": repo["base_url"],
            "subtree": repo.get("subtree"),
            "community": repo.get("community"),
            "element_type": repo.get("element_type"),
            "pages_fetched": len(raw_pages),
            # items_fetched counts everything the API returned; items_kept counts
            # what survived the resource_types filter (see resource_type_filter).
            "items_fetched": filter_stats["kept"] + filter_stats["dropped"],
            "items_kept": len(rows),
            "resource_type_filter": filter_stats,
            "content_class": harvester.content_stats,
            # Backend-specific, absent for backends that do not report them.
            # `enrichment` counts records whose per-record metadata fetch failed
            # (they are still evaluated); `pagination` exposes the gap between
            # rows returned and distinct identifiers, which is the only visible
            # symptom of an unstably ordered listing endpoint.
            "enrichment": getattr(harvester, "enrichment_stats", None),
            "pagination": getattr(harvester, "pagination_stats", None),
            # `registration` records identifiers the repository listed but never
            # registered with the DOI registry. They are excluded because F-UJI
            # scores an unresolvable DOI at its ~13% floor and reports success.
            "registration": getattr(harvester, "registration_stats", None),
            "unique_datasets": len(df),
            "label": cfg["run"]["label"],
        },
        harvest_dir / "harvest_meta.json",
    )
    return df


def _build_manifest(rows: list[dict], log) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    # Nullable int, so "no files" (0) stays distinct from "couldn't look" (NA)
    # instead of both becoming float NaN in parquet.
    df["n_files"] = pd.to_numeric(df["n_files"], errors="coerce").astype("Int64")

    missing = df["pid"].isna().sum()
    if missing:
        log.warning(f"{missing} records have no pid and will be unusable downstream")

    before = len(df)
    df = df.drop_duplicates(subset=["pid"]).reset_index(drop=True)
    if len(df) < before:
        log.info(f"Dropped {before - len(df)} duplicate records (same pid)")
    return df


if __name__ == "__main__":
    _cfg = load_config()
    for _repo in select_repos(_cfg):
        harvest(_repo, _cfg)
