from pathlib import Path

import imagehash
from PIL import Image


def phash(file: Path):

    with Image.open(file) as img:
        return str(imagehash.phash(img))
