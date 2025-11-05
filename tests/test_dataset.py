"""Tests for dataset utilities."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image
from torchvision import transforms

from covid_xray.data import CTDataset, validate_dataset_structure


def _create_dummy_dataset(root: Path, samples_per_class: int = 2) -> None:
    """Populate a dummy dataset with grayscale PNG images.

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


def test_validate_dataset_structure(tmp_path: Path) -> None:
    """Ensure dataset validation catches missing class directories.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    assert not validate_dataset_structure(dataset_root)
    _create_dummy_dataset(dataset_root)
    assert validate_dataset_structure(dataset_root)


def test_ctdataset_loading(tmp_path: Path) -> None:
    """Validate that CTDataset loads images and applies transforms.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """

    dataset_root = tmp_path / "dataset"
    _create_dummy_dataset(dataset_root)
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
    )
    dataset = CTDataset(dataset_root, transform=transform)
    assert len(dataset) == 4
    sample, label = dataset[0]
    assert sample.shape[0] == 1
    assert label in (0, 1)
