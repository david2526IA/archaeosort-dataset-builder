import json

from src.archaeosort_dataset_builder.config.settings import settings


def quality():

    score = 100
    problems = []

    dup_file = settings.reports / "duplicates.json"

    if dup_file.exists():
        dup = json.loads(dup_file.read_text())

        md5 = dup.get("md5_groups", 0)
        phash = dup.get("phash_groups", 0)

        score -= min(md5, 10)
        score -= min(phash // 25, 10)

        problems.append(f"MD5 duplicate groups : {md5}")
        problems.append(f"Visual duplicate groups : {phash}")

    blur_file = settings.reports / "blur.json"

    if blur_file.exists():
        blur = json.loads(blur_file.read_text())

        blurry = blur["blurry"]

        score -= min(blurry // 20, 10)

        problems.append(f"Blurry images : {blurry}")

    brightness_file = settings.reports / "brightness.json"

    if brightness_file.exists():
        brightness = json.loads(brightness_file.read_text())

        total = brightness["dark"] + brightness["bright"]

        score -= min(total // 40, 10)

        problems.append(f"Dark/Bright images : {total}")

    resolution_file = settings.reports / "resolution.json"

    if resolution_file.exists():
        resolution = json.loads(resolution_file.read_text())

        tiny = resolution["tiny"]

        score -= min(tiny // 10, 10)

        problems.append(f"Tiny images : {tiny}")

    score = max(score, 0)

    print("=" * 60)
    print("DATASET QUALITY")
    print("=" * 60)
    print(f"Score : {score}/100")
    print()

    for p in problems:
        print("-", p)
