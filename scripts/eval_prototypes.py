"""One-shot held-out evaluation of the DINOv2 prototype classifier on
natural_rock/stone_artefact (the two classes with abundant local data,
so we can hold out images that were NOT used to build the prototypes).
No held-out set exists yet for bone/wood (only 20 downloaded images per
category, all used as reference) -- that's a known gap, noted honestly.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")

from PIL import Image

from archaeosort_dataset_builder.prototypes.prototypes import classify_by_prototype, load_prototypes

random.seed(123)

bundle = load_prototypes(Path("outputs/prototypes/dinov2.npz"))
print(f"Loaded prototypes for: {bundle.labels}")

results = {"natural_rock": {"correct": 0, "total": 0, "confusions": []},
           "stone_artefact": {"correct": 0, "total": 0, "confusions": []}}

for true_label, folder in [
    ("natural_rock", "datasets/stone_figshare_cropped/v1/test/natural_rock"),
    ("stone_artefact", "datasets/stone_figshare_cropped/v1/test/stone_artefact"),
]:
    # test split was never used for the prototype references (those came
    # from train), so this is a genuinely held-out check.
    paths = sorted(Path(folder).iterdir())
    random.shuffle(paths)
    sample = paths[:15]
    for path in sample:
        image = Image.open(path).convert("RGB")
        result = classify_by_prototype(bundle, image)
        results[true_label]["total"] += 1
        if result["predicted_class"] == true_label:
            results[true_label]["correct"] += 1
        else:
            results[true_label]["confusions"].append((path.name, result["predicted_class"], round(result["confidence"], 3)))

for label, stats in results.items():
    acc = stats["correct"] / stats["total"] if stats["total"] else 0
    print(f"{label}: {stats['correct']}/{stats['total']} correct ({acc:.1%})")
    if stats["confusions"]:
        print(f"  misclassified as: {stats['confusions']}")

Path("outputs/prototypes/eval_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("Report saved: outputs/prototypes/eval_report.json")
