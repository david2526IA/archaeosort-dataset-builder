from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from archaeosort_dataset_builder.config.settings import settings
from archaeosort_dataset_builder.embeddings.model import get_model


def build_embeddings(
    dataset: Path | None = None,
    batch_size: int = 16,
):

    dataset = Path(dataset) if dataset else settings.dataset

    runtime = get_model()

    images = [p for p in dataset.rglob("*") if p.suffix.lower() in settings.image_extensions]

    vectors = []
    paths = []

    for start in tqdm(range(0, len(images), batch_size), desc="Embeddings"):
        batch_paths = images[start : start + batch_size]

        batch_images = []

        for path in batch_paths:
            with Image.open(path) as image:
                batch_images.append(image.convert("RGB").copy())

        inputs = runtime.processor(
            images=batch_images,
            return_tensors="pt",
        )

        inputs = {key: value.to(runtime.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = runtime.model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0, :]

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1,
            )

        vectors.append(embeddings.cpu().numpy().astype("float32"))

        paths.extend(str(path.resolve()) for path in batch_paths)

    matrix = np.concatenate(vectors, axis=0)

    output_dir = settings.outputs / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "dinov2_embeddings.npy", matrix)

    with open(
        output_dir / "dinov2_paths.json",
        "w",
        encoding="utf8",
    ) as f:
        json.dump(
            paths,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("DINOv2 EMBEDDINGS")
    print("=" * 60)
    print(f"Images     : {len(paths)}")
    print(f"Shape      : {matrix.shape}")
    print(f"Batch size : {batch_size}")
    print(f"Device     : {runtime.device}")
    print(f"Output     : {output_dir.resolve()}")


if __name__ == "__main__":
    build_embeddings()
