"""Classify a record by what its attached files actually are.

Why this exists:
`resource_types` (see BaseHarvester) filters on repository-declared types, 
but Dataverse labels every deposit as a dataset regardless of content. 
Since `kindOfData` is effectively unused on Repo4Cat, classification falls back to the payload: 
records containing only PDF, PPTX, or DOCX files are treated as non-data publications. 
This rule matched all 110 Repo4Cat records with no false positives.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, Sequence

CONTENT_CLASSES = frozenset({"data", "mixed", "document_only", "unknown"})
DOCUMENT_FORMATS = frozenset({"pdf", "pptx", "ppt", "docx", "doc", "odp", "odt", "rtf"})

_DOCUMENT_MIMES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "application/vnd.oasis.opendocument.presentation": "odp",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/rtf": "rtf",
    "text/rtf": "rtf",
}


def format_token(mime: str | None = None, name: str | None = None) -> str | None:
    """Normalize a MIME type and/or a filename to one lowercase format token.

    Returns a DOCUMENT_FORMATS member when either signal says "document",
    otherwise the bare MIME, otherwise the bare extension, otherwise None.
    """
    # MIME parameters are real in the wild ("text/plain; charset=US-ASCII").
    mime_t = ""
    if mime:
        mime_t = str(mime).split(";", 1)[0].strip().lower()

    # Last path segment, then last dot-suffix
    ext = ""
    if name:
        last = str(name).strip().rsplit("/", 1)[-1]
        if "." in last:
            ext = last.rsplit(".", 1)[-1].strip().lower()

    if mime_t in _DOCUMENT_MIMES:
        return _DOCUMENT_MIMES[mime_t]
    if ext in DOCUMENT_FORMATS:
        return ext
    return mime_t or ext or None


def classify_files(
    files: Sequence[tuple[str | None, str | None]] | None,
) -> tuple[str, list[str], int | None]:
    if files is None:
        return "unknown", [], None
    if not files:
        return "unknown", [], 0

    tokens = [format_token(mime, name) for mime, name in files]
    known = [t for t in tokens if t]
    n_doc = sum(1 for t in known if t in DOCUMENT_FORMATS)
    formats = sorted(set(known))

    if n_doc == 0:
        return "data", formats, len(files)
    # document_only requires every file to be identified AND a document
    if n_doc == len(tokens):
        return "document_only", formats, len(files)
    return "mixed", formats, len(files)


def files_from_fuji_response(resp: dict) -> list[tuple[str | None, str | None]] | None:
    """Recover a record's file list from a stored F-UJI evaluation.

    Returns None if no source carried `object_content_identifier` at all.
    """
    best: list | None = None
    for src in resp.get("harvested_metadata") or []:
        md = src.get("metadata") if isinstance(src, dict) else None
        if not isinstance(md, dict):
            continue
        items = md.get("object_content_identifier")
        if not isinstance(items, list):
            continue
        if best is None or len(items) > len(best):
            best = items
    if best is None:
        return None
    return [
        (it.get("type"), it.get("url")) if isinstance(it, dict) else (None, None)
        for it in best
    ]


def normalize_class_set(value) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    classes = {str(v).strip().lower() for v in value if str(v).strip()}
    return classes or None


def filter_manifest_by_content_class(
    manifest,
    classes: Iterable[str] | None,
    log: logging.Logger,
    where: str = "",
):
    """Restrict a manifest to the in-scope content classes.

    Returns (in_scope_manifest, {class: n_skipped}).
    """
    scope = normalize_class_set(classes)
    if scope is None:
        return manifest, {}

    tag = f"[{where}] " if where else ""
    if "content_class" not in manifest.columns:
        log.warning(
            f"{tag}manifest has no 'content_class' column (harvested before content "
            f"classification existed); treating all {len(manifest)} records as in "
            "scope. Re-run stage 1 to label them."
        )
        return manifest, {}

    col = manifest["content_class"].fillna("unknown").astype(str).str.strip().str.lower()
    keep = col.isin(scope)
    skipped = dict(Counter(col[~keep]))
    if skipped:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
        log.info(
            f"{tag}content_classes {sorted(scope)}: {int(keep.sum())}/{len(manifest)} "
            f"records in scope ({int((~keep).sum())} skipped: {breakdown})"
        )
    else:
        log.info(
            f"{tag}content_classes {sorted(scope)}: all {len(manifest)} records in scope"
        )
    return manifest[keep].reset_index(drop=True), skipped
