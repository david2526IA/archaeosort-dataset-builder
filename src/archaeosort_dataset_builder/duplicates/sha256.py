import hashlib
from pathlib import Path


def sha256(file: Path):

    h = hashlib.sha256()

    with file.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()
