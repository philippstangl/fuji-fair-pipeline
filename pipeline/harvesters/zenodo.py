"""Zenodo (InvenioRDM) REST API harvester."""
from __future__ import annotations

import time

from pipeline.harvesters.base import BaseHarvester

_ANON_MAX_SIZE = 25
_AUTH_MAX_SIZE = 100


def _total(hits: dict) -> int | None:
    """Zenodo reports hits.total as a bare int (Invenio) or {value, relation}
    (Elasticsearch-style). Normalize to an int."""
    total = hits.get("total")
    if isinstance(total, dict):
        return total.get("value")
    return total


class ZenodoHarvester(BaseHarvester):
    def _harvest(self) -> tuple[list[dict], list[dict]]:
        community = self.repo["community"]
        url = f"{self.base}/api/communities/{community}/records"
        use_concept = bool(self.repo.get("use_concept_doi", False))

        token = self.repo.get("access_token") or None
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        max_size = _AUTH_MAX_SIZE if token else _ANON_MAX_SIZE
        size = min(self.per_page, max_size)
        if self.per_page > max_size:
            hint = "" if token else " (set access_token to raise the limit to 100)"
            self.log.info(f"Capping Zenodo page size {self.per_page} -> {size}{hint}")

        raw_pages: list[dict] = []
        rows: list[dict] = []
        page = 1
        total: int | None = None

        # NOTE: keyed on the row pid, not rec["doi"], with use_concept_doi the
        # pid is the concept DOI, and this must agree with what stage 2 looks up.
        # (With use_concept_doi, sibling versions share a pid and the last one
        # harvested wins here, matching the manifest's drop_duplicates.)
        self._files_by_pid: dict[str, list] = {}

        while True:
            params = {"size": size, "page": page}
            body = self._fetch(url, params, label=f"records page={page}", headers=headers)
            raw_pages.append(body)

            hits = body.get("hits", {}) or {}
            if total is None:
                total = _total(hits)
                self.log.info(f"Total records reported by API: {total}")

            records = hits.get("hits", []) or []
            for rec in records:
                row = self._normalize_record(rec, use_concept)
                # Zenodo gives filenames with extensions but no MIME type.
                self._files_by_pid[row["pid"]] = [
                    (None, f.get("key"))
                    for f in (rec.get("files") or [])
                    if isinstance(f, dict) and f.get("key")
                ]
                rows.append(row)
            self.log.info(f"Fetched {len(rows)}/{total} (page={page}, size={len(records)})")

            if not records:
                break
            if total is not None and len(rows) >= total:
                break
            page += 1
            time.sleep(self.polite_delay)

        return raw_pages, rows

    def _content_files(self, rows: list[dict]) -> dict[str, list]:
        """pid -> [(None, filename)]; already collected during _harvest()."""
        return {row["pid"]: self._files_by_pid.get(row["pid"], []) for row in rows}

    def _normalize_record(self, rec: dict, use_concept: bool) -> dict:
        meta = rec.get("metadata", {}) or {}
        links = rec.get("links", {}) or {}
        version_doi = rec.get("doi")
        concept_doi = rec.get("conceptdoi")
        # Fall back to the other DOI if the preferred one is missing.
        pid = (concept_doi or version_doi) if use_concept else (version_doi or concept_doi)

        resource_type = meta.get("resource_type") or {}
        type_str = resource_type.get("type") if isinstance(resource_type, dict) else resource_type

        return self._new_row(
            pid=pid,
            name=meta.get("title"),
            url=links.get("self_html") or links.get("html") or links.get("doi"),
            type=type_str,
            published_at=meta.get("publication_date"),
            description=meta.get("description"),
            concept_pid=concept_doi,
            version_pid=version_doi,
            publication_statuses=meta.get("access_right"),
        )
