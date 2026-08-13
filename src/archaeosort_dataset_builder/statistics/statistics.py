from archaeosort_dataset_builder.config.settings import settings


def statistics():

    reports = settings.reports

    files = [
        "duplicates.json",
        "blur.json",
        "brightness.json",
        "contrast.json",
        "resolution.json",
        "aspect_ratio.json",
        "class_balance.json",
    ]

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    for f in files:
        path = reports / f

        if path.exists():
            print(f"[OK] {f}")

        else:
            print(f"[--] {f}")

