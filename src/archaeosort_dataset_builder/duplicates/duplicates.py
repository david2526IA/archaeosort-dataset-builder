import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.archaeosort_dataset_builder.config.settings import settings
from src.archaeosort_dataset_builder.duplicates.phash import phash
from src.archaeosort_dataset_builder.duplicates.sha256 import sha256
from src.archaeosort_dataset_builder.duplicates.size_index import size_index


def md5(file):

    h = hashlib.md5()

    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def duplicates(dataset=None):

    dataset = Path(dataset) if dataset else settings.dataset

    sizes = size_index(dataset)

    candidates = [files for files in sizes.values() if len(files) > 1]

    print(f"Candidate groups : {len(candidates)}")

    md5_hashes = defaultdict(list)
    sha_hashes = defaultdict(list)
    phash_hashes = defaultdict(list)

    candidate_files = sum(len(x) for x in candidates)

    print(f"Candidate images : {candidate_files}")

    for group in tqdm(candidates, desc="MD5"):
        for img in group:
            md5_hashes[md5(img)].append(img)

    md5_dup = {k: v for k, v in md5_hashes.items() if len(v) > 1}

    for group in tqdm(md5_dup.values(), desc="SHA256"):
        for img in group:
            sha_hashes[sha256(img)].append(img)

    sha_dup = {k: v for k, v in sha_hashes.items() if len(v) > 1}

    for group in tqdm(sha_dup.values(), desc="pHash"):
        for img in group:
            try:
                phash_hashes[phash(img)].append(img)

            except (OSError, ValueError):
                continue

    phash_dup = {k: v for k, v in phash_hashes.items() if len(v) > 1}

    report = {
        "images": sum(len(v) for v in sizes.values()),
        "candidate_groups": len(candidates),
        "candidate_images": candidate_files,
        "md5_groups": len(md5_dup),
        "sha256_groups": len(sha_dup),
        "phash_groups": len(phash_dup),
    }

    settings.reports.mkdir(parents=True, exist_ok=True)

    with open(settings.reports / "duplicates.json", "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("=" * 60)
    print("DUPLICATE REPORT")
    print("=" * 60)

    for k, v in report.items():
        print(f"{k:20} {v}")
