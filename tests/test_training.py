"""Smoke tests for the training pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

from PIL import Image

from covid_xray import TrainingConfig, train_model
from covid_xray.training import evaluate_model


def _create_dummy_dataset(root: Path, samples_per_class: int = 10) -> None:
    """Populate a dummy dataset for smoke testing the training pipeline.

    Args:
        root: Root directory where the synthetic dataset will be created.
        samples_per_class: Number of images generated for each class label.
    """

    for class_name in ("no", "yes"):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_class):
            pixels = os.urandom(256 * 256)
            Image.frombytes("L", (256, 256), pixels).save(class_dir / f"sample_{index}.png")


def test_train_model_smoke(tmp_path: Path) -> None:
    """Train the model for a single epoch and ensure outputs are generated.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """

    dataset_root = tmp_path / "dataset"
    _create_dummy_dataset(dataset_root)

    output_dir = tmp_path / "outputs"
    best_model_path = tmp_path / "best_model.pth"
    config = TrainingConfig(
        dataset_root=dataset_root,
        batch_size=4,
        num_workers=0,
        seed=0,
        epochs=1,
        patience=1,
        learning_rate=1e-4,
        output_dir=output_dir,
        best_model_path=best_model_path,
    )

    model, device, dataloaders = train_model(config)
    assert best_model_path.exists()

    _, _, test_loader = dataloaders
    loss, accuracy = evaluate_model(model, test_loader, device, output_dir)
    assert loss >= 0.0
    assert 0.0 <= accuracy <= 1.0
