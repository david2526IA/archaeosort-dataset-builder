from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from archaeosort_dataset_builder.config.settings import settings


def build_outlier_contact_sheet(limit: int = 30):

    report_path = settings.reports / "outliers.json"

    data = json.loads(report_path.read_text(encoding="utf8"))

    cases = data["cases"][:limit]

    thumb_w = 240
    thumb_h = 180
    text_h = 65
    margin = 15
    columns = 4

    rows = (len(cases) + columns - 1) // columns

    width = columns * (thumb_w + margin) + margin
    height = rows * (thumb_h + text_h + margin) + margin

    canvas = Image.new("RGB", (width, height), "white")

    draw = ImageDraw.Draw(canvas)

    for i, case in enumerate(cases):
        row = i // columns
        col = i % columns

        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + text_h + margin)

        path = Path(case["path"])

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((thumb_w, thumb_h))

                px = x + (thumb_w - img.width) // 2
                py = y + (thumb_h - img.height) // 2

                canvas.paste(img, (px, py))

        except (OSError, ValueError, MemoryError) as exc:
            draw.text((x, y), f"ERROR\n{exc}", fill="black")

        text = f"#{i + 1} {path.name}\n{case['label']}\nSimilarity: {case['mean_similarity']:.4f}"

        draw.multiline_text((x, y + thumb_h + 3), text, fill="black", spacing=2)

    output_dir = settings.reports / "outlier_review"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / "outliers_top30.jpg"

    canvas.save(output, quality=95)

    print("=" * 60)
    print("OUTLIER REVIEW")
    print("=" * 60)
    print(f"Total outliers : {data['outliers']}")
    print(f"Displayed      : {len(cases)}")
    print(f"Output         : {output}")


if __name__ == "__main__":
    build_outlier_contact_sheet()
