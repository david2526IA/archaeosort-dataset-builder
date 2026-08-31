"""Baseline image classifier: ResNet18 transfer learning over a
train/val/test dataset produced by ``archaeosort build``.

Fase G. Reuses the training recipe already validated in the legacy
``archaeosort-ai`` checkout (ResNet18, image_size 224, batch_size 16,
lr 0.001, seed 42 — see ``_legacy_archaeosort-ai/configs/train_resnet18.yaml``)
but changes two things on purpose, based on what that legacy run showed:

1. ``freeze_backbone`` now defaults to ``False``. The legacy run froze the
   backbone and overfit anyway (train accuracy climbed to 92%+ while
   validation accuracy plateaued around 76-78% and validation loss trended
   up from epoch 4) — freezing didn't prevent overfitting there, it just
   limited how well the model could fit the (different, smaller) legacy
   dataset in the first place.
2. Early stopping (``patience``) is now built in, keeping the model whose
   validation accuracy was best rather than whatever the last epoch
   produced.

This module has no CLI-level side effects — see ``cli.py``'s
``train-baseline`` command for the command-line entry point, which does the
lazy heavy imports (torch/torchvision/scikit-learn) so basic CLI commands
stay fast to start, matching the rest of this codebase's convention.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_SPLITS = ("train", "val", "test")


@dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 15
    learning_rate: float = 1e-3
    freeze_backbone: bool = False
    patience: int = 3
    seed: int = 42
    num_workers: int = 0
    device: str | None = None
    # Robust-model pass: the 6-class dataset (natural_rock/stone_artefact
    # from the large figshare dataset vs. the scraped bone/wood classes)
    # is heavily imbalanced (stone_artefact ~2700 train images vs. ~175
    # for each bone/wood class). Plain CrossEntropyLoss would let the
    # majority classes dominate the gradient; class_weighted=True scales
    # each class's loss contribution by compute_class_weights() so a
    # mistake on a rare class counts as much as one on a common class.
    class_weighted: bool = False
    # Backbone architecture. "resnet18" (default, matches the original
    # legacy recipe) plus stronger/newer torchvision options for when more
    # capacity is worth the extra training time: "resnet34", "resnet50",
    # "convnext_tiny", "efficientnet_v2_s".
    backbone: str = "resnet18"
    # Label smoothing for nn.CrossEntropyLoss (0.0 = off). A small value
    # (e.g. 0.05-0.1) is standard modern practice for classifiers trained
    # on modest per-class sample counts -- it discourages the model from
    # becoming overconfident on the majority classes, which pairs well
    # with class_weighted for an imbalanced dataset like this one.
    label_smoothing: float = 0.0
    # Automatic mixed precision (torch.cuda.amp): trains in float16 where
    # safe, float32 where numerically required. Meaningfully faster on a
    # CUDA GPU with no accuracy cost in practice; a no-op (silently
    # ignored) when running on CPU.
    mixed_precision: bool = False
    # Data augmentation strength for the training transform. "standard"
    # keeps the original recipe (RandomResizedCrop + horizontal flip
    # only). "strong" adds RandAugment + RandomErasing, which is current
    # standard practice for improving generalization on a classifier
    # trained from a modest number of images per class -- exactly the
    # bone/wood classes this robust model struggles with.
    augmentation: str = "standard"


class TrainingError(ValueError):
    """Raised when the training configuration or input dataset is invalid."""


def compute_class_weights(targets: list[int], num_classes: int) -> list[float]:
    """Inverse-frequency weights for ``nn.CrossEntropyLoss(weight=...)``:
    ``weight_i = total / (num_classes * count_i)``, so a class with fewer
    training images contributes proportionally more to the loss instead of
    being drowned out by a majority class. A class absent from `targets`
    (count 0) gets weight 0.0 -- in practice this shouldn't happen, since
    build_dataloaders() already requires train/val/test to share the same
    classes. No torch import here on purpose, so this is testable without
    the heavy dependency (see build_dataloaders/train_baseline for where
    the torch-dependent glue lives)."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    counts = [0] * num_classes
    for target in targets:
        counts[target] += 1

    total = len(targets)
    return [(total / (num_classes * count)) if count > 0 else 0.0 for count in counts]


def validate_split_layout(data_dir: Path) -> Path:
    """Check that ``data_dir`` has train/val/test subdirectories (the layout
    produced by ``archaeosort build``). Returns the resolved path."""
    resolved = Path(data_dir).expanduser().resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved}")

    missing = [split for split in REQUIRED_SPLITS if not (resolved / split).is_dir()]
    if missing:
        raise TrainingError(
            f"Missing split directories under {resolved}: {', '.join(missing)}. "
            "Expected the output of 'archaeosort build' (train/val/test folders)."
        )

    return resolved


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_transforms(image_size: int, train: bool, augmentation: str = "standard"):
    from torchvision import transforms

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if train:
        steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if augmentation == "strong":
            # RandAugment: a small, automatically-searched policy of
            # standard photometric/geometric ops (rotate, shear, contrast,
            # posterize, ...) applied with random magnitude each call --
            # current standard practice for improving generalization when
            # training on a modest number of images per class.
            steps.append(transforms.RandAugment())
        steps += [
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
        if augmentation == "strong":
            # RandomErasing runs on the tensor (after ToTensor), randomly
            # blanking a small rectangular patch -- teaches the model not
            # to rely on any single localized region of the object.
            steps.append(transforms.RandomErasing(p=0.25))
        return transforms.Compose(steps)

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )


def build_dataloaders(
    data_dir: Path,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    augmentation: str = "standard",
) -> tuple[dict[str, Any], list[str]]:
    """Build train/val/test DataLoaders from a build()-shaped dataset.

    Raises TrainingError if the three splits don't share the exact same
    classes in the same order (ImageFolder assigns label indices by sorted
    class-folder name, so a class missing from one split would silently
    misalign labels if this weren't checked).
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets

    resolved = validate_split_layout(data_dir)

    train_ds = datasets.ImageFolder(
        resolved / "train", transform=_build_transforms(image_size, train=True, augmentation=augmentation)
    )
    val_ds = datasets.ImageFolder(
        resolved / "val", transform=_build_transforms(image_size, train=False)
    )
    test_ds = datasets.ImageFolder(
        resolved / "test", transform=_build_transforms(image_size, train=False)
    )

    if not (train_ds.classes == val_ds.classes == test_ds.classes):
        raise TrainingError(
            "train/val/test must expose the exact same classes in the same "
            f"order: train={train_ds.classes} val={val_ds.classes} test={test_ds.classes}"
        )

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    return loaders, train_ds.classes


_BACKBONE_FACTORIES = {
    # name -> (constructor kwarg name for the pretrained weights, weights
    # enum, name of the final classification layer to replace)
    "resnet18": ("resnet18", "ResNet18_Weights", "fc"),
    "resnet34": ("resnet34", "ResNet34_Weights", "fc"),
    "resnet50": ("resnet50", "ResNet50_Weights", "fc"),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", "classifier"),
    "efficientnet_v2_s": ("efficientnet_v2_s", "EfficientNet_V2_S_Weights", "classifier"),
}


def build_model(num_classes: int, freeze_backbone: bool, backbone: str = "resnet18"):
    """Build a pretrained-backbone classifier with its final layer replaced
    for `num_classes`. Defaults to resnet18 (the original recipe); also
    supports resnet34/resnet50 (more capacity, same architecture family)
    and convnext_tiny/efficientnet_v2_s (modern architectures that
    generally outperform ResNet18 on small-to-medium datasets, at the cost
    of more parameters and slower training)."""
    from torch import nn
    from torchvision import models

    if backbone not in _BACKBONE_FACTORIES:
        raise TrainingError(
            f"Unknown backbone {backbone!r}. Choose one of: {sorted(_BACKBONE_FACTORIES)}."
        )
    constructor_name, weights_enum_name, head_attr = _BACKBONE_FACTORIES[backbone]
    constructor = getattr(models, constructor_name)
    weights_enum = getattr(models, weights_enum_name)
    model = constructor(weights=weights_enum.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    head = getattr(model, head_attr)
    if isinstance(head, nn.Sequential):
        # convnext_tiny / efficientnet_v2_s expose `classifier` as a small
        # Sequential ending in the Linear layer -- replace just that layer.
        in_features = head[-1].in_features
        head[-1] = nn.Linear(in_features, num_classes)
    else:
        setattr(model, head_attr, nn.Linear(head.in_features, num_classes))
    return model


def _run_epoch(
    model, loader, criterion, optimizer, device, train: bool, scaler=None
) -> tuple[float, float]:
    import torch

    model.train() if train else model.eval()
    use_amp = scaler is not None and train

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def _evaluate_with_predictions(model, loader, criterion, device):
    import torch

    model.eval()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    accuracy = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / total
    return total_loss / total, accuracy, y_true, y_pred


def train_baseline(config: TrainConfig) -> dict[str, Any]:
    """Train a ResNet18 baseline classifier and evaluate it on the test split.

    Writes ``best_model.pt``, ``training_history.json`` and
    ``evaluation.json`` into ``config.output_dir``, and returns their paths
    plus the in-memory history/evaluation for callers (CLI, tests) that
    don't want to re-read the files from disk.
    """
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    from torch import nn, optim

    set_seed(config.seed)

    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    loaders, classes = build_dataloaders(
        config.data_dir,
        config.image_size,
        config.batch_size,
        config.num_workers,
        augmentation=config.augmentation,
    )
    model = build_model(len(classes), config.freeze_backbone, config.backbone).to(device)

    class_weights_used: list[float] | None = None
    loss_weight = None
    if config.class_weighted:
        train_targets = loaders["train"].dataset.targets
        class_weights_used = compute_class_weights(train_targets, len(classes))
        loss_weight = torch.tensor(class_weights_used, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=config.label_smoothing)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.learning_rate)
    # AMP only helps on CUDA; silently skip it on CPU rather than erroring,
    # so the same config works on either device.
    scaler = torch.cuda.amp.GradScaler() if (config.mixed_precision and device.type == "cuda") else None

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model, loaders["train"], criterion, optimizer, device, train=True, scaler=scaler
        )
        val_loss, val_accuracy = _run_epoch(
            model, loaders["val"], criterion, optimizer, device, train=False
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
            }
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_accuracy, y_true, y_pred = _evaluate_with_predictions(
        model, loaders["test"], criterion, device
    )

    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred).tolist()

    model_path = output_dir / "best_model.pt"
    torch.save(model.state_dict(), model_path)

    evaluation = {
        "classes": classes,
        "device": str(device),
        "freeze_backbone": config.freeze_backbone,
        "backbone": config.backbone,
        "class_weighted": config.class_weighted,
        "class_weights": (
            dict(zip(classes, class_weights_used, strict=True)) if class_weights_used else None
        ),
        "label_smoothing": config.label_smoothing,
        "mixed_precision": scaler is not None,
        "augmentation": config.augmentation,
        "best_validation_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "stopped_early": stopped_early,
        "epochs_run": len(history),
        "epochs_configured": config.epochs,
        "patience": config.patience,
        "seed": config.seed,
    }

    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps({"classes": classes, "history": history}, indent=2), encoding="utf-8"
    )

    evaluation_path = output_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    return {
        "model_path": model_path,
        "history_path": history_path,
        "evaluation_path": evaluation_path,
        "history": history,
        "evaluation": evaluation,
    }
