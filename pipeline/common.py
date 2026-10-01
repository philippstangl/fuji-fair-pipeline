"""Cross-cutting utilities used by every stage."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from pipeline.content import CONTENT_CLASSES, normalize_class_set

_ENV_OVERRIDES = {
    "FUJI_BASE_URL": ("fuji", "base_url"),
    "FUJI_USERNAME": ("fuji", "username"),
    "FUJI_PASSWORD": ("fuji", "password"),
    "FUJI_CONCURRENCY": ("fuji", "concurrency"),
    "RUN_LABEL": ("run", "label"),
}

_INT_KEYS = {("fuji", "concurrency")}


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    if path is None:
        path = os.environ.get("ENV_FILE", ".env")
    path = Path(path)
    if not path.exists():
        return {}

    applied: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config, then apply env-var overrides for selected keys."""
    load_dotenv()

    if path is None:
        path = os.environ.get("CONFIG_PATH", "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for env_key, (section, key) in _ENV_OVERRIDES.items():
        if env_key in os.environ:
            val: Any = os.environ[env_key]
            if (section, key) in _INT_KEYS:
                val = int(val)
            cfg.setdefault(section, {})[key] = val

    _normalize_repositories(cfg)
    return cfg


_STAGE_DIRS = {
    "harvest": "01_harvest",
    "evaluations": "02_evaluations",
    "parsed": "03_parsed",
    "analysis": "04_analysis",
}

_VALID_REPO_TYPES = {"chemotion", "dataverse", "zenodo"}

# Chemotion's own element enum, mirrored
ELEMENT_TYPES = ("Sample", "Reaction", "Container", "Collection")

# A repository name doubles as a directory name
_SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _slug_ok(name: Any) -> bool:
    return isinstance(name, str) and _SLUG_RE.fullmatch(name) is not None


def _merge_defaults(repo: dict, defaults: dict) -> None:
    """Fill keys absent from `repo` using `defaults` (repo values always win)."""
    for key, val in defaults.items():
        if key == "retry":
            merged = dict(val or {})
            merged.update(repo.get("retry") or {})
            repo["retry"] = merged
        else:
            repo.setdefault(key, val)


def _normalize_repositories(cfg: dict) -> None:
    """Apply `defaults` to each repository and validate the list, in place.

    Raises ValueError on a malformed config.
    """
    repos = cfg.get("repositories")
    if repos is None:
        return  # tooling that only needs e.g. fuji/run config
    if not isinstance(repos, list) or not repos:
        raise ValueError("config 'repositories' must be a non-empty list")

    defaults = cfg.get("defaults") or {}
    reserved = (cfg.get("paths") or {}).get("comparison_subdir", "_comparison")
    seen: set[str] = set()
    for i, repo in enumerate(repos):
        name = repo.get("name")
        if not _slug_ok(name):
            raise ValueError(
                f"repositories[{i}].name {name!r} is not a valid slug "
                "(letters, digits, '-' or '_'; no path separators or spaces)"
            )
        if name == reserved:
            raise ValueError(f"repository name {name!r} is reserved for comparison output")
        if name in seen:
            raise ValueError(f"duplicate repository name {name!r}")
        seen.add(name)

        rtype = repo.get("type")
        if rtype not in _VALID_REPO_TYPES:
            raise ValueError(
                f"repositories[{i}] ({name}): type must be one of "
                f"{sorted(_VALID_REPO_TYPES)}, got {rtype!r}"
            )
        if not repo.get("base_url"):
            raise ValueError(f"repositories[{i}] ({name}): base_url is required")
        if rtype == "zenodo" and not repo.get("community"):
            raise ValueError(
                f"repositories[{i}] ({name}): zenodo repositories require a 'community'"
            )
        if rtype == "chemotion":
            element_type = repo.get("element_type")
            if element_type not in ELEMENT_TYPES:
                raise ValueError(
                    f"repositories[{i}] ({name}): chemotion repositories require an "
                    f"'element_type' of {list(ELEMENT_TYPES)}, got {element_type!r}"
                )
        _merge_defaults(repo, defaults)

        rtypes = repo.get("resource_types")
        if rtypes is not None and not isinstance(rtypes, (list, str)):
            raise ValueError(
                f"repositories[{i}] ({name}): resource_types must be a list of type "
                f"names (or null to keep everything), got {type(rtypes).__name__}"
            )

        cclasses = repo.get("content_classes")
        if cclasses is not None and not isinstance(cclasses, (list, str)):
            raise ValueError(
                f"repositories[{i}] ({name}): content_classes must be a list of class "
                f"names (or null to keep everything), got {type(cclasses).__name__}"
            )
        # Unlike resource_types, whose values are backend-specific, 
        # this is a closed vocabulary
        bad = (normalize_class_set(cclasses) or set()) - CONTENT_CLASSES
        if bad:
            raise ValueError(
                f"repositories[{i}] ({name}): unknown content_classes {sorted(bad)}; "
                f"valid values are {sorted(CONTENT_CLASSES)}"
            )

        # Must be a real bool: the string "false" is truthy
        rcc = repo.get("require_content_classification")
        if rcc is not None and not isinstance(rcc, bool):
            raise ValueError(
                f"repositories[{i}] ({name}): require_content_classification must be "
                f"true, false, or absent (use the backend default), got {rcc!r}"
            )

        rfuji = repo.get("fuji")
        if rfuji is not None:
            if not isinstance(rfuji, dict):
                raise ValueError(
                    f"repositories[{i}] ({name}): fuji must be a mapping of overrides "
                    f"for the global fuji block, got {type(rfuji).__name__}"
                )
            known = set(cfg.get("fuji") or {})
            unknown = set(rfuji) - known
            if unknown:
                raise ValueError(
                    f"repositories[{i}] ({name}): unknown fuji override(s) "
                    f"{sorted(unknown)}; valid keys are {sorted(known)}"
                )

        _resolve_credentials(repo, i)


# Credential fields, and the config key naming the env var each is read from
_CREDENTIAL_FIELDS = {
    "api_key": "api_key_env",        # Dataverse (X-Dataverse-key)
    "access_token": "access_token_env",  # Zenodo (Authorization: Bearer)
}


def _resolve_credentials(repo: dict, i: int) -> None:
    """Fill repo[field] from the env var named by repo[<field>_env], if set."""
    for field, env_field in _CREDENTIAL_FIELDS.items():
        var = repo.get(env_field)
        if var is None:
            continue
        if not isinstance(var, str) or not var.strip():
            raise ValueError(
                f"repositories[{i}] ({repo.get('name')}): {env_field} must be the NAME "
                f"of an environment variable, got {var!r}"
            )
        value = os.environ.get(var.strip())
        if value:
            repo[field] = value


def repo_paths(cfg: dict, repo: dict) -> dict[str, Path]:
    """Per-repository directories, repository-first: {root, harvest, evaluations,
    parsed, analysis}. Directories are not created here — callers mkdir as needed."""
    root = Path(cfg["paths"]["data_dir"]) / repo["name"]
    paths: dict[str, Path] = {"root": root}
    for key, sub in _STAGE_DIRS.items():
        paths[key] = root / sub
    return paths


def comparison_dir(cfg: dict) -> Path:
    """Top-level directory for cross-repository comparison output."""
    sub = cfg["paths"].get("comparison_subdir", "_comparison")
    return Path(cfg["paths"]["data_dir"]) / sub


# Sub-blocks of `fuji` that merge key-by-key when a repository overrides them.
# Everything else is replaced wholesale, matching how `defaults:` merges into a
# repository (see _merge_defaults).
_FUJI_MERGE_KEYS = ("retry", "payload_defaults")


def fuji_settings(cfg: dict, repo: dict | None = None) -> dict:
    base = dict(cfg.get("fuji") or {})
    override = (repo or {}).get("fuji") or {}
    for key, val in override.items():
        if key in _FUJI_MERGE_KEYS and isinstance(val, dict):
            merged = dict(base.get(key) or {})
            merged.update(val)
            base[key] = merged
        else:
            base[key] = val
    return base


def select_repos(cfg: dict, only: str | None = None) -> list[dict]:
    """Repositories to act on. Default: every entry with `enabled`."""
    all_repos = cfg.get("repositories") or []
    if only is not None:
        match = [r for r in all_repos if r.get("name") == only]
        if not match:
            known = ", ".join(r.get("name", "?") for r in all_repos) or "(none)"
            raise ValueError(f"repository {only!r} not found. Known repositories: {known}")
        return match
    return [r for r in all_repos if r.get("enabled", True)]


def setup_logging(stage_name: str, log_dir: str | Path = "logs") -> logging.Logger:
    """Set up file + console logging for a stage. One log file per stage per run."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{stage_name}_{ts}.log"

    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
    # Avoid duplicate handlers if re-imported in the same process.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger(stage_name)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_SECRET_KEY_MARKERS = ("password", "token", "secret", "api_key", "apikey")
REDACTED = "***redacted***"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_secret_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    if lowered.endswith("_env"):
        return False
    return any(m in lowered for m in _SECRET_KEY_MARKERS)


def redact_config(obj: Any) -> Any:
    """Deep-copy config with credential values masked."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _is_secret_key(k) and isinstance(v, str) and v:
                out[k] = REDACTED
            else:
                out[k] = redact_config(v)
        return out
    if isinstance(obj, list):
        return [redact_config(v) for v in obj]
    return obj


def git_commit(root: str | Path | None = None) -> dict | None:
    """Commit the code is running from, or None outside a git checkout."""
    import subprocess

    root = Path(root) if root else _REPO_ROOT
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=10,
        )
        if rev.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            capture_output=True, text=True, timeout=10,
        )
        return {
            "commit": rev.stdout.strip(),
            "short": rev.stdout.strip()[:7],
            # A dirty tree means the committed hash does not fully describe the
            # code that produced the output
            "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        }
    except (OSError, subprocess.SubprocessError):
        return None


def snapshot_provenance(cfg: dict, repos: list[dict] | None = None) -> list[dict]:
    """Per-repository snapshot facts: harvest date, evaluation window, F-UJI version."""
    import pandas as pd

    out: list[dict] = []
    for repo in repos if repos is not None else (cfg.get("repositories") or []):
        paths = repo_paths(cfg, repo)
        entry: dict[str, Any] = {"repository_name": repo["name"]}

        meta_path = paths["harvest"] / "harvest_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                entry["harvested_at"] = meta.get("harvested_at")
                entry["items_kept"] = meta.get("items_kept", meta.get("unique_datasets"))
                entry["label"] = meta.get("label")
            except (OSError, ValueError):
                pass

        status_path = paths["evaluations"] / "status.parquet"
        if status_path.exists():
            try:
                s = pd.read_parquet(status_path)
                versions = sorted({str(v) for v in s.get("fuji_version", []) if str(v)})
                entry["fuji_versions"] = versions
                entry["n_evaluated"] = int(len(s))
                if "evaluated_at" in s.columns and len(s):
                    entry["evaluated_from"] = str(s["evaluated_at"].min())
                    entry["evaluated_to"] = str(s["evaluated_at"].max())
            except (OSError, ValueError):
                pass
        out.append(entry)
    return out


def provenance(cfg: dict, repos: list[dict] | None = None) -> dict:
    """Everything needed to attribute an output to the code and data that made it."""
    return {
        "generated_at": utc_now_iso(),
        "git": git_commit(),
        "run_label": (cfg.get("run") or {}).get("label"),
        "snapshots": snapshot_provenance(cfg, repos),
    }


def atomic_write_json(obj: Any, path: str | Path) -> None:
    """Write JSON via a temp file + rename so a crash mid-write can't corrupt the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)
    tmp.replace(path)


def atomic_write_parquet(df, path: str | Path) -> None:
    """Same idea for parquet: write to .tmp then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def safe_filename(doi: str) -> str:
    """Turn a DOI into a filesystem-safe key."""
    s = doi
    for prefix in ("doi:", "https://doi.org/", "http://doi.org/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.replace("/", "__").replace(":", "_").replace(" ", "_")


def retry_call(
    fn: Callable,
    *args,
    max_attempts: int,
    backoff_initial_s: float,
    backoff_multiplier: float,
    logger: logging.Logger | None = None,
    label: str = "",
    **kwargs,
):
    """Call `fn` with exponential backoff. Re-raises the last exception on exhaustion."""
    backoff = backoff_initial_s
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if logger:
                logger.warning(
                    f"{label} attempt {attempt}/{max_attempts} failed: "
                    f"{type(e).__name__}: {e}"
                )
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff *= backoff_multiplier
    assert last_exc is not None
    raise last_exc
