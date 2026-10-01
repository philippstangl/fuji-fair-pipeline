"""Stage 2 — Evaluate each record (by pid) with F-UJI, per repository.

Outputs (under data/<repo name>/02_evaluations/):
  raw/<safe_pid>.json    immutable raw F-UJI response per dataset
  status.parquet         (pid, status, attempts, last_error, fuji_version, evaluated_at)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from pipeline.common import (
    atomic_write_json,
    atomic_write_parquet,
    fuji_settings,
    load_config,
    repo_paths,
    safe_filename,
    select_repos,
    setup_logging,
    utc_now_iso,
)
from pipeline.content import filter_manifest_by_content_class


STATUS_COLS = [
    "pid", "status", "attempts", "last_error", "fuji_version", "evaluated_at",
]

# What F-UJI puts in `resolved_url` when it could not fetch the landing page.
# This string is the only signal that a result is unusable.
UNRESOLVED_URL = "not defined"


def is_unresolved(response: dict) -> bool:
    return isinstance(response, dict) and response.get("resolved_url") == UNRESOLVED_URL


class _UnresolvedGuard:
    """Trip when the landing-page host stops usefully answering."""
    def __init__(
        self,
        consecutive: int = 10,
        rate: float = 0.8,
        window: int = 20,
    ) -> None:
        self._consecutive = int(consecutive or 0)
        self._rate = float(rate or 0.0)
        self._window = int(window or 0)
        self._lock = threading.Lock()
        self._streak = 0
        self._recent: deque[bool] = deque(maxlen=self._window or 1)
        self._tripped = False
        self._reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self._consecutive > 0 or (self._rate > 0 and self._window > 0)

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self._tripped

    @property
    def reason(self) -> str | None:
        """Which rule fired, phrased for the abort message. None until tripped."""
        with self._lock:
            return self._reason

    def record(self, unresolved: bool) -> None:
        with self._lock:
            if self._consecutive > 0:
                if unresolved:
                    self._streak += 1
                    if self._streak >= self._consecutive and not self._tripped:
                        self._tripped = True
                        self._reason = (
                            f"{self._streak} consecutive evaluations came back with "
                            f"resolved_url='{UNRESOLVED_URL}', so the landing-page host "
                            f"is almost certainly down"
                        )
                else:
                    self._streak = 0

            if self._rate > 0 and self._window > 0:
                self._recent.append(bool(unresolved))
                # Only judge a full window: a rate over the first two or three
                # completions is noise, and would abort healthy runs that happen
                # to start badly.
                if len(self._recent) == self._window:
                    n_bad = sum(self._recent)
                    frac = n_bad / self._window
                    if frac >= self._rate and not self._tripped:
                        self._tripped = True
                        self._reason = (
                            f"{n_bad} of the last {self._window} evaluations came back "
                            f"with resolved_url='{UNRESOLVED_URL}' ({frac:.0%}), so the "
                            f"landing-page host is not answering usefully even though it "
                            f"answers intermittently"
                        )


class _RateLimiter:
    """Enforce a minimum interval between dispatches, shared across threads.

    Set min_interval_s = 0 to disable.
    """

    def __init__(self, min_interval_s: float) -> None:
        self._min = max(0.0, float(min_interval_s))
        self._lock = threading.Lock()
        self._next_allowed = 0.0  # monotonic timestamp

    def wait(self) -> None:
        if self._min == 0.0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_allowed - now
            if sleep_for > 0:
                # Holding the lock during sleep is intentional
                time.sleep(sleep_for)
                now = time.monotonic()
            self._next_allowed = now + self._min


def evaluate_all(repo: dict, cfg: dict | None = None) -> None:
    if cfg is None:
        cfg = load_config()
    log = setup_logging(f"stage2_evaluate_{repo['name']}", cfg["paths"]["logs_dir"])

    fuji = fuji_settings(cfg, repo)
    auth = HTTPBasicAuth(fuji["username"], fuji["password"])
    endpoint = fuji["base_url"].rstrip("/") + fuji["endpoint"]
    timeout = fuji["request_timeout_s"]
    concurrency = fuji["concurrency"]
    retry_cfg = fuji["retry"]
    payload_defaults = fuji["payload_defaults"]
    polite_delay = float(fuji.get("polite_delay_s", 0.0))
    rate_limiter = _RateLimiter(polite_delay)
    # Defaults to on; set the thresholds to 0 in config.yaml to disable
    guard = _UnresolvedGuard(
        consecutive=fuji.get("abort_after_consecutive_unresolved", 10),
        rate=fuji.get("abort_unresolved_rate", 0.8),
        window=fuji.get("abort_unresolved_window", 20),
    )

    run_cfg = cfg["run"]
    skip_completed = run_cfg["skip_completed"]
    retry_failed = run_cfg["retry_failed"]
    checkpoint_every = run_cfg["status_checkpoint_every"]

    paths = repo_paths(cfg, repo)
    eval_dir = paths["evaluations"]
    raw_dir = eval_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    status_path = eval_dir / "status.parquet"

    manifest_path = paths["harvest"] / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run stage 1 for {repo['name']} first."
        )
    manifest = pd.read_parquet(manifest_path)
    log.info(f"[{repo['name']}] Loaded manifest with {len(manifest)} datasets")

    manifest, out_of_scope = filter_manifest_by_content_class(
        manifest, repo.get("content_classes"), log, repo["name"]
    )

    fuji_version = _get_fuji_version(fuji, auth, log)
    log.info(f"F-UJI version: {fuji_version}")
    log.info(
        f"Concurrency: {concurrency}, polite delay between requests: {polite_delay}s"
    )
    
    prior_status = _load_status(status_path)
    prior_index = {row["pid"]: row for _, row in prior_status.iterrows()}

    # Build worklist
    todo: list[str] = []
    for _, row in manifest.iterrows():
        doi = row["pid"]
        if not doi or pd.isna(doi):
            continue
        prior = prior_index.get(doi)
        raw_path = raw_dir / f"{safe_filename(doi)}.json"
        if prior is not None:
            if prior["status"] == "success" and raw_path.exists() and skip_completed:
                continue
            if prior["status"] == "failed" and not retry_failed:
                continue
        todo.append(doi)

    n_total = len(manifest)
    n_skip = n_total - len(todo)
    log.info(
        f"Worklist: {len(todo)} to evaluate, {n_skip} skipped (already done or no DOI), "
        f"{sum(out_of_scope.values())} out of content scope"
    )

    completed: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _evaluate_one,
                doi,
                endpoint,
                auth,
                timeout,
                retry_cfg,
                payload_defaults,
                raw_dir,
                fuji_version,
                log,
                rate_limiter,
                guard,
            ): doi
            for doi in todo
        }
        n_abandoned = 0
        for i, fut in enumerate(as_completed(futures), 1):
            doi = futures[fut]
            try:
                result = fut.result()
                if result is None:
                    n_abandoned += 1
                    continue
                completed.append(result)
            except Exception as e:
                log.error(f"Unhandled exception for {doi}: {e!r}")
                completed.append(
                    {
                        "pid": doi,
                        "status": "error",
                        "attempts": retry_cfg["max_attempts"],
                        "last_error": f"{type(e).__name__}: {e}",
                        "fuji_version": fuji_version,
                        "evaluated_at": utc_now_iso(),
                    }
                )
            if i % checkpoint_every == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                eta_s = (len(todo) - i) / rate if rate else 0
                log.info(
                    f"Progress {i}/{len(todo)} "
                    f"({rate:.2f}/s, ETA {eta_s/60:.1f} min). Checkpointing status."
                )
                _save_status(prior_status, completed, status_path)

    _save_status(prior_status, completed, status_path)

    if guard.tripped:
        raise RuntimeError(
            f"[{repo['name']}] Aborted: {guard.reason}. {len(completed)} evaluated, "
            f"{n_abandoned} left for the next run."
        )

    log.info("Stage 2 complete.")


def _evaluate_one(
    doi, endpoint, auth, timeout, retry_cfg, payload_defaults,
    raw_dir, fuji_version, log, rate_limiter, guard,
) -> dict | None:
    # Checked before the rate limiter so a tripped guard drains the remaining
    # queue immediately instead of politely sleeping through every one of them
    if guard.tripped:
        return None

    payload = {"object_identifier": doi, **payload_defaults}
    backoff = retry_cfg["backoff_initial_s"]
    last_exc: Exception | None = None
    for attempt in range(1, retry_cfg["max_attempts"] + 1):
        try:
            rate_limiter.wait()
            r = requests.post(endpoint, json=payload, auth=auth, timeout=timeout)
            r.raise_for_status()
            body = r.json()
            atomic_write_json(body, raw_dir / f"{safe_filename(doi)}.json")
            unresolved = is_unresolved(body)
            if unresolved:
                log.warning(
                    f"{doi}: F-UJI could not resolve the landing page "
                    f"(resolved_url='{UNRESOLVED_URL}'); the score for this record is "
                    "not meaningful"
                )
            guard.record(unresolved)
            return {
                "pid": doi,
                "status": "success",
                "attempts": attempt,
                "last_error": None,
                "fuji_version": fuji_version,
                "evaluated_at": utc_now_iso(),
            }
        except Exception as e:
            last_exc = e
            log.warning(
                f"{doi}: attempt {attempt}/{retry_cfg['max_attempts']} "
                f"failed: {type(e).__name__}: {e}"
            )
            if attempt < retry_cfg["max_attempts"]:
                time.sleep(backoff)
                backoff *= retry_cfg["backoff_multiplier"]
    return {
        "pid": doi,
        "status": "failed",
        "attempts": retry_cfg["max_attempts"],
        "last_error": f"{type(last_exc).__name__}: {last_exc}",
        "fuji_version": fuji_version,
        "evaluated_at": utc_now_iso(),
    }


def _get_fuji_version(fuji_cfg, auth, log) -> str:
    url = fuji_cfg["base_url"].rstrip("/") + fuji_cfg["version_endpoint"]
    try:
        r = requests.get(url, auth=auth, timeout=10)
        if r.ok:
            data = r.json()
            info = data.get("info")
            nested = info.get("version") if isinstance(info, dict) else None
            version = data.get("version") or data.get("api_version") or nested
            if version:
                return str(version)
            log.warning(f"No version field in the response from {url}; recording 'unknown'")
        else:
            log.warning(
                f"F-UJI version endpoint {url} returned HTTP {r.status_code}; "
                "recording 'unknown'. Check fuji.version_endpoint in config.yaml."
            )
    except Exception as e:
        log.warning(f"Could not fetch F-UJI version from {url}: {e}")
    return "unknown"


def _load_status(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        if "pid" not in df.columns and "global_id" in df.columns:
            df = df.rename(columns={"global_id": "pid"})
        return df
    return pd.DataFrame(columns=STATUS_COLS)


def _save_status(prior: pd.DataFrame, new_rows: list[dict], path: Path) -> None:
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows, columns=STATUS_COLS)
    if len(prior) == 0:
        combined = new_df
    else:
        # New rows replace prior rows with the same pid
        keep = prior[~prior["pid"].isin(new_df["pid"])]
        combined = pd.concat([keep, new_df], ignore_index=True)
    atomic_write_parquet(combined, path)


if __name__ == "__main__":
    _cfg = load_config()
    for _repo in select_repos(_cfg):
        evaluate_all(_repo, _cfg)
