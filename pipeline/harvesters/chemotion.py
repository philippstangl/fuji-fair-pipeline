"""Chemotion Repository harvester.

Chemotion publishes one DOI per element, and `element_type` selects which kind
this repository entry harvests. The valid values are 
`pipeline.common.ELEMENT_TYPES`, which is where they are validated:

    Container   a single analysis, e.g. one 13C NMR spectrum   schema.org Dataset
    Sample      a published molecule and its analyzes          schema.org Study
    Reaction    a reaction and its analyzes                    schema.org Study
    Collection  a bundle of the above                          no JSON-LD at all

Harvesting is two steps, because no single endpoint returns the data and its metadata.
"""
from __future__ import annotations

import time
from urllib.parse import parse_qs, quote, urlsplit

from pipeline.harvesters.base import BaseHarvester

_MAX_PAGE_SIZE = 1000

# Every Chemotion DOI exists under this prefix
_DEFAULT_DOI_PREFIX = "10.14272"

# Raw JSON-LD documents are batched this many to a file
_RAW_RECORDS_PER_FILE = 100

DATACITE_API = "https://api.datacite.org/dois"
_DATACITE_PAGE_SIZE = 1000

# Num of unregistered identifiers to name in harvest_meta.json
_MAX_RECORDED_UNREGISTERED = 1000

_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)


def normalize_doi(value) -> str | None:
    """Strip any resolver prefix from a DOI to get `10.x/suffix`.

    Chemotion is inconsistent about this:
    >>> normalize_doi("https://doi.org/10.14272/ABC-1")
    '10.14272/ABC-1'
    >>> normalize_doi("10.14272/reaction/SA-FUHFF")
    '10.14272/reaction/SA-FUHFF'
    >>> normalize_doi("  doi:10.14272/ABC-1  ")
    '10.14272/ABC-1'
    >>> normalize_doi(None) is None
    True
    """
    if not isinstance(value, str):
        return None
    doi = value.strip()
    for prefix in _DOI_URL_PREFIXES:
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi or None


def doi_suffix_from_url(url: str) -> str | None:
    """The `inchikey` query parameter of an enumeration URL."""
    values = parse_qs(urlsplit(url).query).get("inchikey") or []
    suffix = values[0].strip() if values else ""
    return suffix or None


class ChemotionHarvester(BaseHarvester):
    def _harvest(self) -> tuple[list[dict], list[dict]]:
        element_type = self.repo["element_type"]
        prefix = str(self.repo.get("doi_prefix") or _DEFAULT_DOI_PREFIX).strip("/")

        # Start re-harvest clean
        self.raw_record_pages: list[dict] = []
        self.enrichment_stats: dict = {
            "requested": 0, "enriched": 0, "from_registry": 0, "failed": 0,
            "type_mismatch": 0,
        }

        raw_pages, suffixes = self._enumerate(element_type)
        suffixes = self._drop_unregistered(suffixes, prefix)
        rows = self._enrich(suffixes, element_type, prefix)
        return raw_pages, rows


    def _enumerate(self, element_type: str) -> tuple[list[dict], list[str]]:
        url = f"{self.base}/api/v1/public/metadata/publications"
        size = min(self.per_page, _MAX_PAGE_SIZE)
        if self.per_page > _MAX_PAGE_SIZE:
            self.log.info(f"Capping Chemotion page size {self.per_page} -> {size}")

        raw_pages: list[dict] = []
        suffixes: list[str] = []
        offset = 0

        while True:
            params = {"type": element_type, "offset": offset, "limit": size}
            body = self._fetch(url, params, label=f"publications offset={offset}")
            raw_pages.append(body)

            urls = body.get("publications", []) or []
            for u in urls:
                suffix = doi_suffix_from_url(u)
                if suffix:
                    suffixes.append(suffix)
            self.log.info(
                f"Enumerated {len(suffixes)} {element_type} DOIs "
                f"(offset={offset}, size={len(urls)})"
            )

            if len(urls) < size:
                break
            offset += size
            time.sleep(self.polite_delay)

        unique = list(dict.fromkeys(suffixes))
        if len(unique) < len(suffixes):
            self.log.warning(
                f"Enumeration returned {len(suffixes)} rows but only {len(unique)} distinct DOIs"
            )
        self.pagination_stats = {"rows_returned": len(suffixes), "distinct": len(unique)}
        return raw_pages, unique


    def _registered_dois(self, prefix: str) -> set[str]:
        dois: set[str] = set()
        cursor = "1"
        pages = 0
        while True:
            params = {
                "prefix": prefix,
                "page[cursor]": cursor,
                "page[size]": _DATACITE_PAGE_SIZE,
                "fields[dois]": "doi",
            }
            body = self._fetch(
                DATACITE_API, params, label=f"datacite prefix={prefix} page={pages + 1}"
            )
            batch = [d.get("id") for d in (body.get("data") or []) if d.get("id")]
            dois.update(d.lower() for d in batch)
            pages += 1
            nxt = (body.get("links") or {}).get("next")
            if not batch or not nxt:
                break
            cursor_values = parse_qs(urlsplit(nxt).query).get("page[cursor]") or []
            if not cursor_values:
                break
            cursor = cursor_values[0]
        self.log.info(f"DataCite holds {len(dois)} DOIs under prefix {prefix} ({pages} pages)")
        return dois

    def _drop_unregistered(self, suffixes: list[str], prefix: str) -> list[str]:
        """Remove identifiers DataCite has never heard of."""
        self.registration_stats = {
            "checked": False, "registered": None, "unregistered": 0, "unregistered_pids": [],
        }
        if not self.repo.get("verify_doi_registration", True):
            self.log.info("verify_doi_registration is false. Skipping the DataCite check.")
            return suffixes

        try:
            registered = self._registered_dois(prefix)
        except Exception as e:
            raise RuntimeError(
                f"[{self.repo['name']}] Refusing to harvest: could not check which "
                f"DOIs are registered with DataCite ({type(e).__name__}: {e}). "
            ) from e

        kept, missing = [], []
        for suffix in suffixes:
            (kept if f"{prefix}/{suffix}".lower() in registered else missing).append(suffix)

        self.registration_stats = {
            "checked": True,
            "registered": len(kept),
            "unregistered": len(missing),
            "unregistered_pids": [f"{prefix}/{s}" for s in missing[:_MAX_RECORDED_UNREGISTERED]],
        }
        if missing:
            self.log.warning(
                f"{len(missing)}/{len(suffixes)} enumerated DOIs are not registered "
                f"with DataCite and do not resolve; excluding them."
            )
        else:
            self.log.info(f"All {len(suffixes)} enumerated DOIs are registered with DataCite")
        return kept


    def _enrich(self, suffixes: list[str], element_type: str, prefix: str) -> list[dict]:
        url = f"{self.base}/api/v1/public/metadata/download_json"
        total = len(suffixes)
        self.log.info(
            f"Fetching JSON-LD for {total} records (~{total * 2.2 / 3600:.1f}h at the "
            f"~2.2s/record measured for reactions)"
        )

        rows: list[dict] = []
        batch: list[dict] = []
        for i, suffix in enumerate(suffixes, start=1):
            self.enrichment_stats["requested"] += 1
            fallback_pid = f"{prefix}/{suffix}"
            try:
                doc = self._fetch(
                    url, {"inchikey": suffix}, label=f"jsonld {suffix}"
                )
            except Exception as e:
                rows.append(self._fill_gap(
                    fallback_pid, element_type,
                    f"JSON-LD unavailable ({type(e).__name__}: {e})",
                ))
                continue

            if not isinstance(doc, dict) or not doc:
                rows.append(self._fill_gap(
                    fallback_pid, element_type,
                    "Chemotion serves no RDF for this element",
                ))
                continue

            self.enrichment_stats["enriched"] += 1
            batch.append(doc)
            if len(batch) >= _RAW_RECORDS_PER_FILE:
                self.raw_record_pages.append({"records": batch})
                batch = []
            rows.append(self._normalize_record(doc, fallback_pid, element_type))

            if i % 250 == 0 or i == total:
                self.log.info(f"Enriched {i}/{total} records")
            time.sleep(self.polite_delay)

        if batch:
            self.raw_record_pages.append({"records": batch})

        stats = self.enrichment_stats
        if stats["from_registry"]:
            self.log.info(
                f"{stats['from_registry']}/{total} records took their metadata from "
                f"DataCite rather than from Chemotion. Recorded in harvest_meta.json"
            )
        if stats["failed"]:
            self.log.warning(
                f"{stats['failed']}/{total} records could not be enriched from either "
                f"source and carry only a pid and type. They are still evaluated."
            )
        if stats["type_mismatch"]:
            self.log.warning(
                f"{stats['type_mismatch']} records whose JSON-LD additionalType "
                f"disagrees with element_type={element_type!r}"
            )
        return rows

    def _fill_gap(self, pid: str, element_type: str, why: str) -> dict:
        """Build a row for a record Chemotion served no metadata for."""
        if self.repo.get("datacite_metadata_fallback", True):
            try:
                attrs = self._datacite_record(pid)
            except Exception as e:
                attrs = None
                self.log.debug(f"DataCite lookup failed for {pid}: {type(e).__name__}: {e}")
            if attrs:
                self.enrichment_stats["from_registry"] += 1
                return self._normalize_datacite(attrs, pid, element_type)

        self.enrichment_stats["failed"] += 1
        self.log.warning(f"No metadata for {pid} ({why}); emitting a metadata-less row")
        return self._degraded_row(pid, element_type)

    def _degraded_row(self, pid: str, element_type: str) -> dict:
        """A row for a record whose metadata could not be fetched from anywhere."""
        return self._new_row(
            pid=pid, type=element_type.lower(), url=f"https://doi.org/{pid}"
        )

    def _datacite_record(self, pid: str) -> dict | None:
        """DataCite's `attributes` for one DOI, or None if it holds nothing."""
        body = self._fetch(
            f"{DATACITE_API}/{quote(pid, safe='')}", {}, label=f"datacite doi={pid}"
        )
        return ((body or {}).get("data") or {}).get("attributes") or None

    def _normalize_datacite(self, attrs: dict, pid: str, element_type: str) -> dict:
        """Map a DataCite record onto the same manifest columns as the JSON-LD."""
        def _first(key: str, field: str):
            items = attrs.get(key) or []
            return items[0].get(field) if items and isinstance(items[0], dict) else None

        return self._new_row(
            pid=pid,
            name=_first("titles", "title"),
            url=attrs.get("url") or f"https://doi.org/{pid}",
            type=element_type.lower(),
            published_at=attrs.get("registered"),
            description=_first("descriptions", "description"),
            publication_statuses=_first("rightsList", "rightsIdentifier"),
        )

    def _content_files(self, rows: list[dict]) -> dict[str, list] | None:
        return None

    def _normalize_record(self, doc: dict, fallback_pid: str, element_type: str) -> dict:
        pid = normalize_doi(doc.get("@id")) or fallback_pid

        # `additionalType` is Chemotion's own element name "Reaction"
        additional = doc.get("additionalType")
        if isinstance(additional, str) and additional.lower() != element_type.lower():
            self.enrichment_stats["type_mismatch"] += 1

        license = doc.get("license")
        rights = license.get("rightsIdentifier") if isinstance(license, dict) else None

        return self._new_row(
            pid=pid,
            name=doc.get("name"),
            url=doc.get("url") or f"https://doi.org/{pid}",
            type=element_type.lower(),
            published_at=doc.get("datePublished"),
            description=doc.get("description"),
            publication_statuses=rights,
        )
