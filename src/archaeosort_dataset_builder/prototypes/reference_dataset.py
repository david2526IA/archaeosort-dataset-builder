"""Automatically source a small reference-image set for categories the
Fase G classifier was never trained on (bone/wood artefacts, and their
natural counterparts), so a prototype classifier (see prototypes.py) can
be built without hand-curating a dataset.

Sources (both public, no API key required):
  - The Metropolitan Museum of Art Open Access API (worked/artefact
    categories -- searched by keyword, filtered to public-domain images):
    https://metmuseum.github.io/
  - Wikimedia Commons (natural-material categories, via category
    membership + imageinfo): https://commons.wikimedia.org/wiki/Commons:API

This is deliberately NOT a rigorous, deduplicated, human-reviewed
dataset -- it's a fast way to get a handful (~10-30) of real, varied
photos per category for a *few-shot prototype* (see prototypes.py),
where a handful of diverse examples is the whole input, not thousands of
carefully balanced ones. `natural_rock`/`stone_artefact` reuse the
existing local dataset instead of downloading anything.

Network access is required (this queries live public APIs) -- run this
from the project's real environment, not an offline/sandboxed one. Each
network call is wrapped so one failed image doesn't abort the whole
category; a category that ends up with too few images raises
ReferenceDatasetError so the gap is visible rather than silently
producing a near-empty, unreliable prototype.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Every per-image network call below is wrapped in one of these -- a
# single failed search/fetch/download should skip that item, not abort
# the whole category (public APIs occasionally 404, rate-limit, or
# return malformed JSON for one record). Narrowed to the actual expected
# failure modes (HTTP/connection errors, and malformed-response KeyError
# lookups) rather than a blanket `except Exception`, and logged at debug
# level so a category that ends up short on images is diagnosable.
_FETCH_EXCEPTIONS = (OSError, ValueError, KeyError, TypeError)

MET_SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ArchaeoSort-DatasetBuilder/1.0 (research prototype; https://github.com)"
MIN_IMAGES_PER_CLASS = 5


class ReferenceDatasetError(Exception):
    """Raised when a category ends up with too few usable reference images."""


@dataclass
class CategorySource:
    """How to source reference images for one open-vocabulary category."""

    kind: str  # "local" | "met" | "commons"
    # "local": a class-folder under the existing dataset to sample from.
    local_dir: Path | None = None
    # "met": Met Museum keyword searches to try, in order, until enough
    # images are found (broader museum object-type filters, not a single
    # exact medium string, since Met's controlled vocabulary for "medium"
    # is inconsistent across departments).
    met_queries: list[str] = field(default_factory=list)
    # "commons": Wikimedia Commons category names to pull from.
    commons_categories: list[str] = field(default_factory=list)


# Default sourcing plan for the two new material pairs the user asked
# for. natural_rock/stone_artefact are intentionally absent here -- the
# CLI wires those up from the existing local dataset directly.
DEFAULT_SOURCES: dict[str, CategorySource] = {
    "bone_artefact": CategorySource(
        kind="met",
        met_queries=["carved bone", "bone comb", "bone tool", "bone needle", "bone amulet"],
    ),
    "natural_bone": CategorySource(
        kind="commons",
        commons_categories=["Category:Animal_bones", "Category:Skeletons_of_mammals"],
    ),
    "wood_artefact": CategorySource(
        kind="met",
        met_queries=["wood carving", "wooden sculpture", "carved wooden figure", "wooden mask"],
    ),
    "natural_wood": CategorySource(
        kind="commons",
        commons_categories=["Category:Driftwood", "Category:Tree_bark", "Category:Firewood"],
    ),
}


def _http_get_json(url: str, params: dict[str, Any], timeout: float = 15.0) -> dict[str, Any]:
    import requests

    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _download_bytes(url: str, timeout: float = 20.0) -> bytes:
    import requests

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.content


def _fetch_met_images(queries: list[str], limit: int) -> list[tuple[bytes, str]]:
    """Try each query in turn, collecting (image_bytes, source_url) pairs
    from Met Museum objects with a public-domain small image, until
    `limit` is reached or all queries are exhausted."""
    results: list[tuple[bytes, str]] = []
    seen_object_ids: set[int] = set()

    for query in queries:
        if len(results) >= limit:
            break
        try:
            search = _http_get_json(MET_SEARCH_URL, {"q": query, "hasImages": "true"})
        except _FETCH_EXCEPTIONS as exc:
            logger.debug("Met search failed for query %r: %s", query, exc)
            continue

        object_ids = search.get("objectIDs") or []
        random.shuffle(object_ids)

        for object_id in object_ids:
            if len(results) >= limit:
                break
            if object_id in seen_object_ids:
                continue
            seen_object_ids.add(object_id)

            try:
                obj = _http_get_json(MET_OBJECT_URL.format(object_id=object_id), {})
            except _FETCH_EXCEPTIONS as exc:
                logger.debug("Met object %s fetch failed: %s", object_id, exc)
                continue

            if not obj.get("isPublicDomain"):
                continue
            image_url = obj.get("primaryImageSmall") or obj.get("primaryImage")
            if not image_url:
                continue

            try:
                image_bytes = _download_bytes(image_url)
            except _FETCH_EXCEPTIONS as exc:
                logger.debug("Download failed for %s: %s", image_url, exc)
                continue

            results.append((image_bytes, image_url))

    return results


def _fetch_commons_images(categories: list[str], limit: int) -> list[tuple[bytes, str]]:
    """Pull member files from each Commons category, resolve their
    thumbnail URLs, and download them, until `limit` is reached."""
    results: list[tuple[bytes, str]] = []

    for category in categories:
        if len(results) >= limit:
            break
        try:
            members = _http_get_json(
                COMMONS_API_URL,
                {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": category,
                    "cmtype": "file",
                    "cmlimit": "50",
                    "format": "json",
                },
            )
        except _FETCH_EXCEPTIONS as exc:
            logger.debug("Commons categorymembers failed for %r: %s", category, exc)
            continue

        titles = [m["title"] for m in members.get("query", {}).get("categorymembers", [])]
        random.shuffle(titles)

        # Commons' imageinfo endpoint accepts multiple titles per call
        # (pipe-separated), so batch instead of one request per file.
        for batch_start in range(0, len(titles), 20):
            if len(results) >= limit:
                break
            batch = titles[batch_start : batch_start + 20]
            try:
                info = _http_get_json(
                    COMMONS_API_URL,
                    {
                        "action": "query",
                        "titles": "|".join(batch),
                        "prop": "imageinfo",
                        "iiprop": "url",
                        "iiurlwidth": "512",
                        "format": "json",
                    },
                )
            except _FETCH_EXCEPTIONS as exc:
                logger.debug("Commons imageinfo batch failed: %s", exc)
                continue

            pages = info.get("query", {}).get("pages", {})
            for page in pages.values():
                if len(results) >= limit:
                    break
                imageinfo = page.get("imageinfo") or []
                if not imageinfo:
                    continue
                image_url = imageinfo[0].get("thumburl") or imageinfo[0].get("url")
                if not image_url:
                    continue
                try:
                    image_bytes = _download_bytes(image_url)
                except _FETCH_EXCEPTIONS as exc:
                    logger.debug("Download failed for %s: %s", image_url, exc)
                    continue
                results.append((image_bytes, image_url))

    return results


def _sample_local_images(directory: Path, limit: int) -> list[Path]:
    directory = Path(directory)
    candidates = [
        p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()
    ]
    random.shuffle(candidates)
    return candidates[:limit]


def build_reference_dataset(
    output_dir: Path,
    sources: dict[str, CategorySource] | None = None,
    per_class: int = 20,
    force: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Populate `output_dir/<label>/` with up to `per_class` reference
    images per label, sourced per `sources` (defaults to DEFAULT_SOURCES).
    A label already holding >= MIN_IMAGES_PER_CLASS images is left alone
    unless force=True, so re-running is cheap and idempotent.

    Returns a manifest dict (also written to output_dir/manifest.json)
    recording each category's source URLs, for provenance/licensing.
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    sources = sources or DEFAULT_SOURCES
    manifest: dict[str, Any] = {"per_class_requested": per_class, "categories": {}}

    for label, source in sources.items():
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        existing = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png"))
        if existing and not force:
            manifest["categories"][label] = {"count": len(existing), "reused_existing": True}
            continue

        if source.kind == "local":
            if source.local_dir is None:
                raise ReferenceDatasetError(f"CategorySource for {label!r} is 'local' but local_dir is unset.")
            paths = _sample_local_images(source.local_dir, per_class)
            for i, path in enumerate(paths):
                (label_dir / f"{i:03d}{path.suffix.lower()}").write_bytes(path.read_bytes())
            manifest["categories"][label] = {
                "count": len(paths),
                "source": "local",
                "local_dir": str(source.local_dir),
            }
        elif source.kind == "met":
            fetched = _fetch_met_images(source.met_queries, per_class)
            urls = []
            for i, (image_bytes, url) in enumerate(fetched):
                (label_dir / f"{i:03d}.jpg").write_bytes(image_bytes)
                urls.append(url)
            manifest["categories"][label] = {"count": len(fetched), "source": "met_museum", "urls": urls}
        elif source.kind == "commons":
            fetched = _fetch_commons_images(source.commons_categories, per_class)
            urls = []
            for i, (image_bytes, url) in enumerate(fetched):
                (label_dir / f"{i:03d}.jpg").write_bytes(image_bytes)
                urls.append(url)
            manifest["categories"][label] = {"count": len(fetched), "source": "wikimedia_commons", "urls": urls}
        else:
            raise ReferenceDatasetError(f"Unknown CategorySource.kind {source.kind!r} for {label!r}.")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    too_few = {
        label: info["count"]
        for label, info in manifest["categories"].items()
        if info["count"] < MIN_IMAGES_PER_CLASS
    }
    if too_few:
        raise ReferenceDatasetError(
            f"Not enough reference images downloaded for: {too_few} "
            f"(need >= {MIN_IMAGES_PER_CLASS} each). Check network access, or re-run -- "
            "public APIs occasionally rate-limit or have a transient outage."
        )

    return manifest
