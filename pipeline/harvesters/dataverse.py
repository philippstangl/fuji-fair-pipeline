"""Dataverse Search API harvester.
Uses the native Dataverse Search API (`/api/search?q=*&type=dataset`).
"""
from __future__ import annotations

import time
from collections import defaultdict

from pipeline.harvesters.base import BaseHarvester

# Use of Dataverse maximum rather than the per-repo `per_page` maximum.
# Reusing per_page=100 here would turn 3 requests into 24 for no benefit.
_FILE_INDEX_PAGE_SIZE = 1000


class DataverseHarvester(BaseHarvester):

    CONTENT_CLASSIFICATION_REQUIRED = True

    def _auth_headers(self) -> dict:
        key = (self.repo.get("api_key") or "").strip()
        return {"X-Dataverse-key": key} if key else {}

    def _harvest(self) -> tuple[list[dict], list[dict]]:
        url = f"{self.base}/api/search"
        subtree = self.repo.get("subtree")
        if subtree:
            self.log.info(f"Scoping Dataverse search to subtree={subtree!r}")

        headers = self._auth_headers()
        self.log.info(
            "Using an API key for Dataverse requests"
            if headers
            else "No Dataverse API key configured"
        )

        # Start a re-harvest clean
        self._file_counts: dict[str, int] = {}

        raw_pages: list[dict] = []
        rows: list[dict] = []
        start = 0
        total: int | None = None

        while True:
            params = {"q": "*", "type": "dataset", "per_page": self.per_page, "start": start}
            if subtree:
                params["subtree"] = subtree
            body = self._fetch(url, params, label=f"search start={start}", headers=headers)
            raw_pages.append(body)

            data = body.get("data", {}) or {}
            if total is None:
                total = data.get("total_count", 0)
                self.log.info(f"Total datasets reported by API: {total}")

            items = data.get("items", []) or []
            rows.extend(self._normalize_item(it) for it in items)
            self.log.info(f"Fetched {len(rows)}/{total} (page start={start}, size={len(items)})")

            if not items:
                self.log.warning("Received empty page; stopping pagination.")
                break
            start += self.per_page
            if start >= total:
                break
            time.sleep(self.polite_delay)

        return raw_pages, rows

    def _content_files(self, rows: list[dict]) -> dict[str, list]:
        """pid -> [(mime, filename)] via the whole-instance file index.
        Deliberately all-or-nothing: if any page fails, the exception propagates
        to `_label_content`, which labels everything `unknown`.
        """
        url = f"{self.base}/api/search"
        subtree = self.repo.get("subtree")
        headers = self._auth_headers()
        by_pid: dict[str, list] = defaultdict(list)
        self.raw_file_pages = []
        start, total = 0, None

        while True:
            params = {
                "q": "*", "type": "file",
                "per_page": _FILE_INDEX_PAGE_SIZE, "start": start,
            }
            if subtree:
                params["subtree"] = subtree
            body = self._fetch(
                url, params, label=f"file index start={start}", headers=headers
            )
            self.raw_file_pages.append(body)

            data = body.get("data", {}) or {}
            if total is None:
                total = data.get("total_count", 0)
                self.log.info(f"File index reports {total} files")

            items = data.get("items", []) or []
            if not items:
                break
            for it in items:
                by_pid[it.get("dataset_persistent_id")].append(
                    (it.get("file_content_type"), it.get("name"))
                )
            start += _FILE_INDEX_PAGE_SIZE
            if total is not None and start >= total:
                break
            time.sleep(self.polite_delay)

        out: dict[str, list] = {}
        joined = 0
        mismatched: list[str] = []
        for row in rows:
            pid = row.get("pid")
            files = by_pid.get(pid)
            if files is None:
                out[pid] = []
                continue
            joined += 1
            # Integrity check: the search item's own fileCount must agree with what
            # the anonymous file index returned. A shortfall means files are
            # hidden from us (restricted/embargoed)
            expected = self._file_counts.get(pid)
            if expected is not None and expected != len(files):
                mismatched.append(pid)
                out[pid] = None
            else:
                out[pid] = files

        self.log.info(f"File index joined {joined}/{len(rows)} harvested records")
        if mismatched:
            self.log.warning(
                f"{len(mismatched)} records where fileCount disagrees with the file "
                f"index (files may be restricted); labeling them 'unknown': "
                f"{mismatched[:5]}{' ...' if len(mismatched) > 5 else ''}"
            )
        return out

    def _normalize_item(self, it: dict) -> dict:
        self._file_counts[it.get("global_id")] = it.get("fileCount")
        return self._new_row(
            pid=it.get("global_id"),
            name=it.get("name"),
            url=it.get("url"),
            type=it.get("type"),
            published_at=it.get("published_at"),
            description=it.get("description"),
            identifier_of_dataverse=it.get("identifier_of_dataverse"),
            name_of_dataverse=it.get("name_of_dataverse"),
            publication_statuses=",".join(it.get("publicationStatuses", []) or []),
            citation=it.get("citation"),
        )
