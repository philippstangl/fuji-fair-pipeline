"""Stage 4 Analysis: per-repository FAIR analysis + a cross-repository comparison."""
from __future__ import annotations

from pipeline.common import load_config, repo_paths, select_repos, setup_logging


def analyze(repo: dict, cfg: dict | None = None) -> None:
    # Imported inside the function: duckdb/matplotlib are analysis-only extras,
    # mirroring the lazy per-stage imports in pipeline/__main__.py.
    from pipeline.analysis import analyze_with_plots, decompose_effort

    if cfg is None:
        cfg = load_config()
    log = setup_logging(f"stage4_analyze_{repo['name']}", cfg["paths"]["logs_dir"])
    data_dir = str(cfg["paths"]["data_dir"])

    summary = repo_paths(cfg, repo)["parsed"] / "summary.parquet"
    if not summary.exists():
        log.warning(f"[{repo['name']}] no {summary}; run stages 1-3 first. Skipping analysis.")
        return

    for module in (analyze_with_plots, decompose_effort):
        log.info(f"[{repo['name']}] running: {module.__name__}")
        module.run(repo["name"], data_dir)


def compare(cfg: dict | None = None) -> None:
    """Cross-repository comparison; reads every repo that has parsed output."""
    from pipeline.analysis import compare_repos

    if cfg is None:
        cfg = load_config()
    log = setup_logging("stage4_compare", cfg["paths"]["logs_dir"])
    log.info(f"running: {compare_repos.__name__}")
    compare_repos.run(cfg)


if __name__ == "__main__":
    _cfg = load_config()
    for _repo in select_repos(_cfg):
        analyze(_repo, _cfg)
    compare(_cfg)
