"""Pipeline orchestrator.

Run from the project root:
    python -m pipeline                              # all stages, all enabled repositories
    python -m pipeline --stages harvest             # just harvest, all enabled repositories
    python -m pipeline --repo nfdi4cat_zenodo       # all stages, one repository
    python -m pipeline --list-repos                 # list configured repositories and exit

Stages run in order; within each stage every selected repository is processed.
If a stage fails for a repository, that repository is skipped for the remaining
stages while the others continue.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from pipeline.common import (
    atomic_write_json,
    load_config,
    provenance,
    redact_config,
    select_repos,
    utc_now_iso,
)

STAGES = ["harvest", "evaluate", "parse", "analyze"]

_STAGE_FUNCS = {
    "harvest": ("pipeline.stage1_harvest", "harvest"),
    "evaluate": ("pipeline.stage2_evaluate", "evaluate_all"),
    "parse": ("pipeline.stage3_parse", "parse_all"),
    "analyze": ("pipeline.stage4_analyze", "analyze"),
}


def _run_stage(stage: str, repo: dict, cfg: dict) -> None:
    module_name, func_name = _STAGE_FUNCS[stage]
    module = __import__(module_name, fromlist=[func_name])
    getattr(module, func_name)(repo, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="F-UJI pipeline orchestrator")
    ap.add_argument(
        "--stages", nargs="+", default=STAGES, choices=STAGES,
        help=f"Stages to run, in order. Default: {STAGES}",
    )
    ap.add_argument(
        "--repo", default=None,
        help="Run only this repository (by name). Default: all enabled repositories.",
    )
    ap.add_argument(
        "--list-repos", action="store_true",
        help="List configured repositories and exit.",
    )
    args = ap.parse_args()

    cfg = load_config()

    if args.list_repos:
        for r in cfg.get("repositories", []):
            scope = r.get("subtree") or r.get("community") or r.get("element_type") or ""
            scope = f" [{scope}]" if scope else ""
            flag = "" if r.get("enabled", True) else "  (disabled)"
            print(f"{r['name']:24s} {r['type']:10s} {r['base_url']}{scope}{flag}")
        return

    repos = select_repos(cfg, only=args.repo)
    started = utc_now_iso()
    fname_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"=== Pipeline run started {started} ===")
    print(f"Stages      : {args.stages}")
    print(f"Repositories: {[r['name'] for r in repos]}")
    print(f"Label       : {cfg['run']['label']}")
    print()

    results: list[dict] = []
    failed_repos: set[str] = set()
    for stage in args.stages:
        for repo in repos:
            name = repo["name"]
            if name in failed_repos:
                print(f"Skipping {stage} for {name} (an earlier stage failed)")
                continue
            s_start = utc_now_iso()
            print(f"Stage: {stage} | repo: {name} ({s_start})")
            ok, err = True, None
            try:
                _run_stage(stage, repo, cfg)
            except Exception as e:
                ok = False
                err = f"{type(e).__name__}: {e}"
                failed_repos.add(name)
                print(f"!! Stage {stage} for {name} failed: {err}")
            results.append(
                {"stage": stage, "repo": name, "started_at": s_start,
                 "finished_at": utc_now_iso(), "ok": ok, "error": err}
            )

    # Cross-repository comparison runs once, after the per-repo analyze stage.
    if "analyze" in args.stages:
        c_start = utc_now_iso()
        print(f"Stage: analyze | cross-repo comparison ({c_start})")
        ok, err = True, None
        try:
            from pipeline.stage4_analyze import compare
            compare(cfg)
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {e}"
            print(f"!! Cross-repo comparison failed: {err}")
        results.append(
            {"stage": "compare", "repo": "(all)", "started_at": c_start,
             "finished_at": utc_now_iso(), "ok": ok, "error": err}
        )

    manifest = {
        "started_at": started,
        "finished_at": utc_now_iso(),
        "stages_run": args.stages,
        "repos_run": [r["name"] for r in repos],
        "failed_repos": sorted(failed_repos),
        "results": results,
        "provenance": provenance(cfg, repos),
        "config_snapshot": redact_config(cfg),
    }
    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest, runs_dir / f"run_{fname_ts}.json")
    if failed_repos:
        print(f"!! Repositories with failures: {sorted(failed_repos)}")
    print(f"=== Pipeline run finished {manifest['finished_at']} ===")


if __name__ == "__main__":
    main()
