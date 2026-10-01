"""Harvester abstraction shared by every repository architecture type."""
from __future__ import annotations

import logging
from collections import Counter

import requests

from pipeline.common import retry_call
from pipeline.content import CONTENT_CLASSES, classify_files, normalize_class_set


MANIFEST_COLUMNS = [
    "pid",
    "global_id",
    "repository_name",
    "repository_type",
    "name",
    "url",
    "type",
    "published_at",
    "description",
    "name_of_dataverse",
    "identifier_of_dataverse",
    "concept_pid",
    "version_pid",
    "publication_statuses",
    "citation",
    "content_class",
    "content_formats",
    "n_files",
]


def _get_json(url: str, params: dict, timeout: int, headers: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()


def _normalize_types(value) -> set[str] | None:
    """Config `resource_types` -> a lowercase set, or None meaning "keep all"."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    types = {str(v).strip().lower() for v in value if str(v).strip()}
    return types or None


class BaseHarvester:
    """Fetch + normalize the records of a single configured repository."""

    CONTENT_CLASSIFICATION_REQUIRED = False

    def __init__(self, repo: dict, log: logging.Logger) -> None:
        self.repo = repo
        self.log = log
        self.base = repo["base_url"].rstrip("/")
        self.per_page = int(repo.get("per_page", 100))
        self.timeout = repo.get("request_timeout_s", 60)
        self.polite_delay = float(repo.get("polite_delay_s", 0.0))
        self.retry_cfg = repo.get("retry") or {}
        self.resource_types = _normalize_types(repo.get("resource_types"))
        # Both populated by harvest(); stage 1 records them in harvest_meta.json.
        self.filter_stats: dict = {
            "resource_types": sorted(self.resource_types) if self.resource_types else None,
            "kept": 0,
            "dropped": 0,
            "dropped_by_type": {},
        }
        self.content_stats: dict = {
            "source": "not_implemented",
            "counts": {},
            "files_seen": 0,
            "records_joined": 0,
            "records_without_file_info": 0,
        }

    def harvest(self) -> tuple[list[dict], list[dict]]:
        """Return (raw_pages, manifest_rows), type-filtered and content-labeled."""
        raw_pages, rows = self._harvest()
        rows = self._filter_by_type(rows)
        self._label_content(rows)
        return raw_pages, rows

    def _harvest(self) -> tuple[list[dict], list[dict]]:
        """Backend-specific fetch + normalize. Implemented by subclasses."""
        raise NotImplementedError


    def _content_files(self, rows: list[dict]) -> dict[str, list] | None:
        """pid -> [(mime, filename)] for the given rows."""
        return None

    def _classification_scope(self) -> set[str] | None:
        """The configured content_classes, or None when nothing is filtered out."""
        scope = normalize_class_set(self.repo.get("content_classes"))
        if scope is None or CONTENT_CLASSES <= scope:
            # Either keep everything, or a scope naming every class
            return None
        return scope

    def _classification_failure_is_fatal(self) -> bool:
        """Whether an unavailable file listing should stop the harvest."""
        override = self.repo.get("require_content_classification")
        required = (
            self.CONTENT_CLASSIFICATION_REQUIRED if override is None else bool(override)
        )
        return required and self._classification_scope() is not None

    def _label_content(self, rows: list[dict]) -> None:
        source = "unavailable"
        by_pid: dict[str, list] | None = None
        try:
            by_pid = self._content_files(rows)
            if by_pid is None:
                source = "not_implemented"
            else:
                source = f"{self.repo['type']}_files"
        except Exception as e:
            if self._classification_failure_is_fatal():
                scope = self._classification_scope() or set()
                raise RuntimeError(
                    f"[{self.repo['name']}] Refusing to harvest: the file listing this "
                    f"repository needs to classify payloads is unavailable "
                    f"({type(e).__name__}: {e}). Every one of the {len(rows)} records "
                    f"would be labeled 'unknown', which is inside the configured "
                    f"content_classes {sorted(scope)} "
                ) from e
            self.log.warning(
                f"Could not determine file payloads ({type(e).__name__}: {e}); "
                "labeling every record 'unknown' (still in scope by default)."
            )

        counts: Counter = Counter()
        files_seen = joined = no_info = 0
        for row in rows:
            files = by_pid.get(row.get("pid")) if by_pid is not None else None
            if files is None:
                no_info += 1
            else:
                joined += 1
                files_seen += len(files)
            cls, formats, n = classify_files(files)
            row["content_class"] = cls
            row["content_formats"] = ",".join(formats) if formats else None
            row["n_files"] = n
            counts[cls] += 1

        self.content_stats.update(
            source=source,
            counts=dict(counts),
            files_seen=files_seen,
            records_joined=joined,
            records_without_file_info=no_info,
        )
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        self.log.info(
            f"content classification ({source}): {breakdown} "
            f"({files_seen} files, {joined}/{len(rows)} records with file info)"
        )

    def _filter_by_type(self, rows: list[dict]) -> list[dict]:
        """Keep only rows whose normalized `type` is whitelisted."""
        if not self.resource_types:
            self.filter_stats["kept"] = len(rows)
            return rows

        kept: list[dict] = []
        dropped: Counter = Counter()
        for row in rows:
            rtype = row.get("type")
            key = str(rtype).strip().lower() if rtype is not None else ""
            if key in self.resource_types:
                kept.append(row)
            else:
                dropped[key or "(no type)"] += 1

        self.filter_stats.update(
            kept=len(kept), dropped=sum(dropped.values()), dropped_by_type=dict(dropped)
        )
        if dropped:
            breakdown = ", ".join(f"{t}={n}" for t, n in dropped.most_common())
            self.log.info(
                f"resource_types filter {sorted(self.resource_types)}: kept {len(kept)}/"
                f"{len(rows)}, dropped {sum(dropped.values())} ({breakdown})"
            )
        else:
            self.log.info(
                f"resource_types filter {sorted(self.resource_types)}: kept all {len(kept)} records"
            )
        return kept

    # helpers for subclasses

    def _fetch(self, url: str, params: dict, label: str, headers: dict | None = None) -> dict:
        """GET JSON with the repository's retry/backoff policy."""
        return retry_call(
            _get_json,
            url,
            params,
            self.timeout,
            headers,
            max_attempts=self.retry_cfg.get("max_attempts", 5),
            backoff_initial_s=self.retry_cfg.get("backoff_initial_s", 2),
            backoff_multiplier=self.retry_cfg.get("backoff_multiplier", 2),
            logger=self.log,
            label=label,
        )

    def _new_row(self, **kwargs) -> dict:
        """Build a manifest row with every column present (None if unset) and the
        repository identity stamped on it. `global_id` mirrors `pid` by default."""
        row = {col: None for col in MANIFEST_COLUMNS}
        row["repository_name"] = self.repo["name"]
        row["repository_type"] = self.repo["type"]
        row.update(kwargs)
        if row.get("global_id") is None:
            row["global_id"] = row.get("pid")
        return row
