from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from archaeosort_dataset_builder.embeddings.model import get_model


def encode_image(image_path: Path) -> np.ndarray:

    runtime = get_model()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        inputs = runtime.processor(
            images=image,
            return_tensors="pt",
        )

    inputs = {key: value.to(runtime.device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = runtime.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :]

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1,
        )

    return embedding.squeeze(0).cpu().numpy().astype("float32")
