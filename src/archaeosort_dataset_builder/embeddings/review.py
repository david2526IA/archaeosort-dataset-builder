from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from archaeosort_dataset_builder.config.settings import settings


def review():

    report_path = settings.reports / "semantic_audit.json"

    data = json.loads(report_path.read_text(encoding="utf8"))

    output_dir = settings.reports / "semantic_review"
    output_dir.mkdir(parents=True, exist_ok=True)

    for case_number, case in enumerate(data["cases"], start=1):
        neighbors = case["neighbors"]

        items = [
            {
                "path": case["path"],
                "label": "QUERY",
                "similarity": None,
            }
        ] + neighbors

        thumb_w = 260
        thumb_h = 190
        text_h = 65
        margin = 15

        columns = 4
        rows = (len(items) + columns - 1) // columns

        header_h = 120

        width = columns * (thumb_w + margin) + margin
        height = header_h + rows * (thumb_h + text_h + margin) + margin

        canvas = Image.new("RGB", (width, height), "white")

        draw = ImageDraw.Draw(canvas)

        header = (
            f"Current label: {case['current_label']}\n"
            f"Neighbor prediction: {case['neighbor_prediction']}\n"
            f"Agreement: {case['agreement']:.0%}"
        )

        draw.multiline_text((margin, 15), header, fill="black", spacing=5)

        for i, item in enumerate(items):
            row = i // columns
            col = i % columns

            x = margin + col * (thumb_w + margin)
            y = header_h + row * (thumb_h + text_h + margin)

            path = Path(item["path"])

            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((thumb_w, thumb_h))

                px = x + (thumb_w - img.width) // 2
                py = y + (thumb_h - img.height) // 2

                canvas.paste(img, (px, py))

            if i == 0:
                text = f"QUERY\n{case['current_label']}\n{path.name}"

            else:
                text = f"{item['label']}\nSimilarity: {item['similarity']:.4f}\n{path.name}"

            draw.multiline_text((x, y + thumb_h + 3), text, fill="black", spacing=2)

        output = output_dir / f"suspicious_{case_number:03d}_full.jpg"

        canvas.save(output, quality=95)

        print(output)


if __name__ == "__main__":
    review()
