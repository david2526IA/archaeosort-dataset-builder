from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-small"


class DinoV2Model:
    def __init__(self) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Loading: {MODEL_NAME}")

        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)

        self.model.eval()

        print("[OK] DINOv2 loaded")


_model_instance: DinoV2Model | None = None


def get_model() -> DinoV2Model:

    global _model_instance

    if _model_instance is None:
        _model_instance = DinoV2Model()

    return _model_instance
