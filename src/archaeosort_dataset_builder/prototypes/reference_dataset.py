"""Automatically source a reference-image set for categories the Fase G
classifier was never trained on (bone/wood artefacts, and their natural
counterparts), so a real multi-class classifier (see classifier/baseline.py)
can be trained without hand-curating a dataset.

Sources (all public, no API key required):
  - The Metropolitan Museum of Art Open Access API (worked/artefact
    categories -- searched by keyword, filtered to public-domain images):
    https://metmuseum.github.io/
  - Wikimedia Commons (natural-material and some artefact categories, via
    category membership + imageinfo): https://commons.wikimedia.org/wiki/Commons:API
  - iNaturalist (natural-material photos -- animal bones, driftwood, bark
    -- via the observations search API, filtered to openly licensed
    photos): https://www.inaturalist.org/pages/api+reference
  - GBIF (Global Biodiversity Information Facility) occurrence search --
    a huge, free/keyless archive of specimen and observation photos
    contributed by museums and citizen scientists worldwide, used here as
    a fourth source for natural-material categories (skeletal/bone and
    driftwood/bark) to widen photographer/locale/species diversity beyond
    iNaturalist alone: https://www.gbif.org/developer/occurrence

v2 (robust-model pass): raised the realistic per-class yield by (a) adding
iNaturalist as a third source for natural-material categories, (b) letting
a category pull from more than one source at once (`kind="multi"`), and
(c) paginating Wikimedia Commons category membership instead of stopping
at the first 50 files. `natural_rock`/`stone_artefact` are NOT sourced
here -- the existing Auckland figshare dataset (~5,300 images, see
datasets/stone_figshare_cropped/) already covers those two classes far
better than anything scraped in a few hours, and is merged in separately
by scripts/build_robust_dataset.py.

v3 (broader-coverage pass): added GBIF as a fourth source and substantially
widened the keyword/category lists for all four categories (more Met
search terms, more Commons categories, more iNaturalist/GBIF queries) so
each class draws from a wider variety of museums, regions, and species/
object types rather than a handful of narrow searches -- the goal is more
genuine visual diversity per class, not just more images of the same kind
of photo.

This is deliberately NOT a rigorous, deduplicated, human-reviewed dataset
-- it is a fast way to get a real, licensed, varied set of photos per
category to train a genuine CNN classifier on. Network access is required
(this queries live public APIs) -- run it from a machine with real
internet access. Each network call is wrapped so one failed image doesn't
abort the whole category; a category that ends up with too few images
raises ReferenceDatasetError so the gap is visible rather than silently
producing a near-empty, unreliable class.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _heartbeat(source: str, count: int, limit: int) -> None:
    """Emit an INFO-level progress line every 25 fetched images, so a
    long-running download shows visible progress in whatever captures
    this logger (see scripts/build_robust_dataset.py)."""
    if count % 25 == 0 or count == limit:
        logger.info("[%s] %d/%d images fetched", source, count, limit)

# Every per-image network call below is wrapped in one of these -- a
# single failed search/fetch/download should skip that item, not abort
# the whole category (public APIs occasionally 404, rate-limit, or return
# malformed JSON for one record). Narrowed to the actual expected failure
# modes (HTTP/connection errors, and malformed-response KeyError lookups)
# rather than a blanket `except Exception`, and logged at debug level so a
# category that ends up short on images is diagnosable.
_FETCH_EXCEPTIONS = (OSError, ValueError, KeyError, TypeError)

MET_SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
INATURALIST_URL = "https://api.inaturalist.org/v1/observations"
GBIF_URL = "https://api.gbif.org/v1/occurrence/search"
USER_AGENT = "ArchaeoSort-DatasetBuilder/1.0 (research prototype; https://github.com)"
MIN_IMAGES_PER_CLASS = 5
# Politeness delay between network calls -- these are free public APIs
# with no key, so we self-throttle rather than risk a 429/ban mid-run.
_REQUEST_DELAY_SECONDS = 0.15


class ReferenceDatasetError(Exception):
    """Raised when a category ends up with too few usable reference images."""


@dataclass
class CategorySource:
    """How to source reference images for one open-vocabulary category."""

    kind: str = "multi"  # "local" | "met" | "commons" | "inaturalist" | "gbif" | "multi"
    # "local": a class-folder under an existing dataset to sample from.
    local_dir: Path | None = None
    # Met Museum keyword searches to try, in order, until enough images
    # are found (broader museum object-type filters, not a single exact
    # medium string, since Met's controlled vocabulary for "medium" is
    # inconsistent across departments).
    met_queries: list[str] = field(default_factory=list)
    # Wikimedia Commons category names to pull members from.
    commons_categories: list[str] = field(default_factory=list)
    # iNaturalist observation search terms (natural-material photos).
    inaturalist_queries: list[str] = field(default_factory=list)
    # GBIF occurrence search terms (natural-material specimen photos).
    gbif_queries: list[str] = field(default_factory=list)


# Default sourcing plan for the four open-vocabulary categories the user
# asked for beyond stone. natural_rock/stone_artefact are intentionally
# absent here -- those come from the existing Auckland figshare dataset,
# merged in by scripts/build_robust_dataset.py.
DEFAULT_SOURCES: dict[str, CategorySource] = {
    "bone_artefact": CategorySource(
        kind="multi",
        met_queries=[
            "carved bone",
            "bone comb",
            "bone tool",
            "bone needle",
            "bone amulet",
            "bone pendant",
            "bone flute",
            "bone awl",
            "bone knife handle",
            "engraved bone",
            "bone pin",
            "bone spoon",
            "bone fish hook",
            "bone harpoon",
            "bone whistle",
            "bone bead",
            "bone dice",
            "bone hairpin",
            "ivory carving",
            "antler tool",
            "horn carving",
            "scrimshaw",
        ],
        commons_categories=[
            "Category:Bone_tools",
            "Category:Objects_made_of_bone",
            "Category:Bone_carvings",
            "Category:Ivory_carvings",
            "Category:Scrimshaw",
            "Category:Antler_tools",
        ],
    ),
    "natural_bone": CategorySource(
        kind="multi",
        commons_categories=[
            "Category:Animal_bones",
            "Category:Skeletons_of_mammals",
            "Category:Skulls",
            "Category:Bones",
            "Category:Fish_skeletons",
            "Category:Bird_skeletons",
        ],
        inaturalist_queries=[
            "animal skeleton",
            "animal bone",
            "skull bone",
            "bone remains",
            "vertebra",
            "deer antler",
            "bird skull",
            "fish skeleton",
            "rib bone",
            "jaw bone",
            "whale bone",
        ],
        gbif_queries=[
            "skeleton",
            "skull",
            "bone",
            "vertebra",
            "antler",
        ],
    ),
    "wood_artefact": CategorySource(
        kind="multi",
        met_queries=[
            "wood carving",
            "wooden sculpture",
            "carved wooden figure",
            "wooden mask",
            "wooden bowl",
            "wood figure",
            "carved wooden box",
            "wooden spoon",
            "wooden comb",
            "wood netsuke",
            "wooden staff",
            "wooden panel carving",
            "wooden chest",
            "carved wooden door",
            "wooden totem",
            "wood inlay",
            "wooden furniture carved",
        ],
        commons_categories=[
            "Category:Wood_carvings",
            "Category:Wooden_sculptures",
            "Category:Wooden_masks",
            "Category:Wooden_furniture",
            "Category:Marquetry",
            "Category:Totem_poles",
        ],
    ),
    "natural_wood": CategorySource(
        kind="multi",
        commons_categories=[
            "Category:Driftwood",
            "Category:Tree_bark",
            "Category:Firewood",
            "Category:Dead_wood",
            "Category:Tree_stumps",
            "Category:Fallen_trees",
        ],
        inaturalist_queries=[
            "driftwood",
            "tree bark",
            "fallen log",
            "dead wood",
            "rotten log",
            "tree stump",
            "tree root",
            "decaying wood",
            "woodland floor log",
        ],
        gbif_queries=[
            "driftwood",
            "deadwood",
            "tree bark",
            "fallen tree",
        ],
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


def _fetch_met_images(
    queries: list[str], limit: int, exclude_urls: frozenset[str] = frozenset()
) -> list[tuple[bytes, str]]:
    """Try each query in turn, collecting (image_bytes, source_url) pairs
    from Met Museum objects with a public-domain small image, until
    `limit` is reached or all queries are exhausted. `exclude_urls` skips
    images already downloaded in a previous top-up run."""
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

            time.sleep(_REQUEST_DELAY_SECONDS)
            try:
                obj = _http_get_json(MET_OBJECT_URL.format(object_id=object_id), {})
            except _FETCH_EXCEPTIONS as exc:
                logger.debug("Met object %s fetch failed: %s", object_id, exc)
                continue

            if not obj.get("isPublicDomain"):
                continue
            image_url = obj.get("primaryImageSmall") or obj.get("primaryImage")
            if not image_url or image_url in exclude_urls:
                continue

            try:
                image_bytes = _download_bytes(image_url)
            except _FETCH_EXCEPTIONS as exc:
                logger.debug("Download failed for %s: %s", image_url, exc)
                continue

            results.append((image_bytes, image_url))
            _heartbeat("met", len(results), limit)

    return results


def _fetch_commons_category_titles(category: str, limit: int) -> list[str]:
    """Paginate Wikimedia Commons category membership (cmcontinue) instead
    of stopping at the first 50 files, up to `limit` titles."""
    titles: list[str] = []
    cmcontinue: str | None = None

    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "file",
            "cmlimit": "100",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        try:
            page = _http_get_json(COMMONS_API_URL, params)
        except _FETCH_EXCEPTIONS as exc:
            logger.debug("Commons categorymembers failed for %r: %s", category, exc)
            break

        members = page.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)

        cmcontinue = page.get("continue", {}).get("cmcontinue")
        if not cmcontinue or not members:
            break
        time.sleep(_REQUEST_DELAY_SECONDS)

    return titles[:limit]


def _fetch_commons_images(
    categories: list[str], limit: int, exclude_urls: frozenset[str] = frozenset()
) -> list[tuple[bytes, str]]:
    """Pull member files from each Commons category, resolve their
    thumbnail URLs, and download them, until `limit` is reached.
    `exclude_urls` skips images already downloaded in a previous top-up run."""
    results: list[tuple[bytes, str]] = []

    for category in categories:
        if len(results) >= limit:
            break
        # Ask for more titles than the remaining quota -- not every title
        # resolves to a usable imageinfo entry.
        titles = _fetch_commons_category_titles(category, max((limit - len(results)) * 3, 50))
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
                if not image_url or image_url in exclude_urls:
                    continue
                time.sleep(_REQUEST_DELAY_SECONDS)
                try:
                    image_bytes = _download_bytes(image_url)
                except _FETCH_EXCEPTIONS as exc:
                    logger.debug("Download failed for %s: %s", image_url, exc)
                    continue
                results.append((image_bytes, image_url))
                _heartbeat("commons", len(results), limit)

    return results


def _fetch_inaturalist_images(
    queries: list[str], limit: int, exclude_urls: frozenset[str] = frozenset()
) -> list[tuple[bytes, str]]:
    """Search iNaturalist observations for each query, taking one openly
    licensed photo per observation (for diversity across individuals /
    photographers), until `limit` is reached. `exclude_urls` skips images
    already downloaded in a previous top-up run."""
    results: list[tuple[bytes, str]] = []
    seen_photo_ids: set[int] = set()

    for query in queries:
        if len(results) >= limit:
            break
        for page in range(1, 6):
            if len(results) >= limit:
                break
            try:
                data = _http_get_json(
                    INATURALIST_URL,
                    {
                        "q": query,
                        "photos": "true",
                        "photo_license": "cc0,cc-by,cc-by-sa,pd",
                        "per_page": 50,
                        "page": page,
                        "order_by": "votes",
                    },
                )
            except _FETCH_EXCEPTIONS as exc:
                logger.debug("iNaturalist search failed for %r page %s: %s", query, page, exc)
                break

            observations = data.get("results") or []
            if not observations:
                break

            for obs in observations:
                if len(results) >= limit:
                    break
                for photo in obs.get("photos") or []:
                    photo_id = photo.get("id")
                    url = photo.get("url")
                    if not url or photo_id in seen_photo_ids:
                        continue
                    # iNaturalist thumbnail URLs default to the tiny
                    # "square" size; request a larger rendition.
                    image_url = url.replace("square", "medium")
                    if image_url in exclude_urls:
                        continue
                    time.sleep(_REQUEST_DELAY_SECONDS)
                    try:
                        image_bytes = _download_bytes(image_url)
                    except _FETCH_EXCEPTIONS as exc:
                        logger.debug("Download failed for %s: %s", image_url, exc)
                        continue
                    seen_photo_ids.add(photo_id)
                    results.append((image_bytes, image_url))
                    _heartbeat("inaturalist", len(results), limit)
                    break  # one photo per observation

            time.sleep(_REQUEST_DELAY_SECONDS)

    return results


def _fetch_gbif_images(
    queries: list[str], limit: int, exclude_urls: frozenset[str] = frozenset()
) -> list[tuple[bytes, str]]:
    """Search GBIF occurrence records with attached photos for each query,
    taking one image per occurrence, until `limit` is reached.
    `exclude_urls` skips images already downloaded in a previous top-up
    run. GBIF aggregates specimen/observation photos from museums and
    citizen-science projects worldwide -- useful as a fourth, independent
    source of natural-material photos (different contributor pool than
    iNaturalist alone)."""
    results: list[tuple[bytes, str]] = []
    seen_keys: set[int] = set()

    for query in queries:
        if len(results) >= limit:
            break
        for page_offset in range(0, 150, 50):
            if len(results) >= limit:
                break
            try:
                data = _http_get_json(
                    GBIF_URL,
                    {
                        "q": query,
                        "mediaType": "StillImage",
                        "limit": 50,
                        "offset": page_offset,
                    },
                )
            except _FETCH_EXCEPTIONS as exc:
                logger.debug("GBIF search failed for %r offset %s: %s", query, page_offset, exc)
                break

            occurrences = data.get("results") or []
            if not occurrences:
                break

            for occurrence in occurrences:
                if len(results) >= limit:
                    break
                key = occurrence.get("key")
                if key in seen_keys:
                    continue
                media_list = occurrence.get("media") or []
                image_url = next(
                    (m.get("identifier") for m in media_list if m.get("identifier")), None
                )
                if not image_url or image_url in exclude_urls:
                    continue
                seen_keys.add(key)
                time.sleep(_REQUEST_DELAY_SECONDS)
                try:
                    image_bytes = _download_bytes(image_url)
                except _FETCH_EXCEPTIONS as exc:
                    logger.debug("Download failed for %s: %s", image_url, exc)
                    continue
                results.append((image_bytes, image_url))
                _heartbeat("gbif", len(results), limit)

            time.sleep(_REQUEST_DELAY_SECONDS)

    return results


def _sample_local_images(directory: Path, limit: int) -> list[Path]:
    directory = Path(directory)
    candidates = [
        p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()
    ]
    random.shuffle(candidates)
    return candidates[:limit]


def _fetch_for_source(
    source: CategorySource, per_class: int, exclude_urls: frozenset[str] = frozenset()
) -> tuple[list[tuple[bytes, str]], str]:
    """Dispatch a CategorySource to its fetcher(s). "multi" pools every
    non-empty field on the source, in met -> commons -> inaturalist order,
    stopping early once per_class is reached. `exclude_urls` (already-
    downloaded image URLs from a previous top-up run) is passed through so
    a top-up fetches genuinely new images instead of re-downloading."""
    if source.kind == "met":
        return _fetch_met_images(source.met_queries, per_class, exclude_urls), "met_museum"
    if source.kind == "commons":
        return _fetch_commons_images(source.commons_categories, per_class, exclude_urls), "wikimedia_commons"
    if source.kind == "inaturalist":
        return _fetch_inaturalist_images(source.inaturalist_queries, per_class, exclude_urls), "inaturalist"
    if source.kind == "gbif":
        return _fetch_gbif_images(source.gbif_queries, per_class, exclude_urls), "gbif"
    if source.kind == "multi":
        fetched: list[tuple[bytes, str]] = []
        used: list[str] = []
        if source.met_queries:
            fetched.extend(_fetch_met_images(source.met_queries, per_class - len(fetched), exclude_urls))
            used.append("met_museum")
        if source.commons_categories and len(fetched) < per_class:
            fetched.extend(
                _fetch_commons_images(source.commons_categories, per_class - len(fetched), exclude_urls)
            )
            used.append("wikimedia_commons")
        if source.inaturalist_queries and len(fetched) < per_class:
            fetched.extend(
                _fetch_inaturalist_images(source.inaturalist_queries, per_class - len(fetched), exclude_urls)
            )
            used.append("inaturalist")
        if source.gbif_queries and len(fetched) < per_class:
            fetched.extend(
                _fetch_gbif_images(source.gbif_queries, per_class - len(fetched), exclude_urls)
            )
            used.append("gbif")
        return fetched, "+".join(used) if used else "multi"
    raise ReferenceDatasetError(f"Unknown CategorySource.kind {source.kind!r}.")


def build_reference_dataset(
    output_dir: Path,
    sources: dict[str, CategorySource] | None = None,
    per_class: int = 20,
    force: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Populate `output_dir/<label>/` with up to `per_class` reference
    images per label, sourced per `sources` (defaults to DEFAULT_SOURCES).

    A label already holding >= `per_class` images is left alone unless
    force=True. A label holding *fewer* than `per_class` images (e.g. a
    previous run used a smaller target, or was interrupted) is topped up
    with `per_class - existing` additional images instead of being
    re-downloaded from scratch -- URLs already recorded in a prior
    manifest.json are excluded so a top-up adds genuinely new images
    rather than re-fetching duplicates. force=True always starts a label
    over from zero.

    Returns a manifest dict (also written to output_dir/manifest.json)
    recording each category's source URLs, for provenance/licensing.
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    sources = sources or DEFAULT_SOURCES
    manifest: dict[str, Any] = {"per_class_requested": per_class, "categories": {}}

    previous_manifest_path = output_dir / "manifest.json"
    previous_urls_by_label: dict[str, list[str]] = {}
    if previous_manifest_path.is_file():
        try:
            previous_data = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
            for label, info in previous_data.get("categories", {}).items():
                previous_urls_by_label[label] = list(info.get("urls") or [])
        except (json.JSONDecodeError, OSError):
            pass

    for label, source in sources.items():
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png")))
        if force:
            existing = []

        if len(existing) >= per_class:
            logger.info("[%s] reusing %d existing images (already at target)", label, len(existing))
            manifest["categories"][label] = {
                "count": len(existing),
                "reused_existing": True,
                "urls": previous_urls_by_label.get(label, []),
            }
            continue

        remaining = per_class - len(existing)
        if existing:
            logger.info(
                "[%s] topping up: %d existing, fetching %d more (target %d)",
                label, len(existing), remaining, per_class,
            )
        else:
            logger.info("[%s] starting download (target %d images)", label, per_class)

        if source.kind == "local":
            if source.local_dir is None:
                raise ReferenceDatasetError(f"CategorySource for {label!r} is 'local' but local_dir is unset.")
            paths = _sample_local_images(source.local_dir, remaining)
            for offset, path in enumerate(paths):
                index = len(existing) + offset
                (label_dir / f"{index:03d}{path.suffix.lower()}").write_bytes(path.read_bytes())
            manifest["categories"][label] = {
                "count": len(existing) + len(paths),
                "source": "local",
                "local_dir": str(source.local_dir),
            }
            continue

        exclude_urls = frozenset(previous_urls_by_label.get(label, []))
        fetched, source_label = _fetch_for_source(source, remaining, exclude_urls)
        urls = list(previous_urls_by_label.get(label, []))
        for offset, (image_bytes, url) in enumerate(fetched):
            index = len(existing) + offset
            (label_dir / f"{index:03d}.jpg").write_bytes(image_bytes)
            urls.append(url)
        manifest["categories"][label] = {
            "count": len(existing) + len(fetched),
            "source": source_label,
            "urls": urls,
        }

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
