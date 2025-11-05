"""Tests for inference utilities."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

import torch
from PIL import Image

from covid_xray.inference import CLASS_NAMES, build_inference_transform, load_trained_model, predict_image
from covid_xray.model import RadiographyCNN


def test_load_and_predict(tmp_path: Path) -> None:
    """Ensure a saved model can be reloaded and used for inference.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """

    weights_path = tmp_path / "model.pth"
    model = RadiographyCNN()
    torch.save(model.state_dict(), weights_path)

    loaded_model, device = load_trained_model(weights_path, device=torch.device("cpu"))
    transform = build_inference_transform()

    pixels = os.urandom(256 * 256 * 3)
    image = Image.frombytes("RGB", (256, 256), pixels)

    label, confidence, probabilities = predict_image(
        image_input=image,
        model=loaded_model,
        device=device,
        transform=transform,
    )

    assert label in CLASS_NAMES
    assert 0.0 <= confidence <= 1.0
    assert probabilities.shape[0] == len(CLASS_NAMES)
