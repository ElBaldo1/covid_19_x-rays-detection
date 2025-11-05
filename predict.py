"""Command-line interface for running COVID-19 chest X-ray inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import torch

from covid_xray.inference import (
    build_inference_transform,
    load_trained_model,
    predict_image,
)

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inference script.

    Returns:
        Parsed command-line arguments namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run inference on chest X-ray images using a pretrained RadiographyCNN "
            "model."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Image files or directories containing images for prediction.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("best_model.pth"),
        help="Path to the trained model weights file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (e.g. 'cpu', 'cuda').",
    )
    return parser.parse_args()


def discover_images(paths: Iterable[Path]) -> List[Path]:
    """Discover image files under the provided paths.

    Args:
        paths: Iterable of files or directories supplied by the user.

    Returns:
        Sorted list of valid image file paths.
    """

    discovered: List[Path] = []
    for path in paths:
        if path.is_dir():
            for extension in SUPPORTED_EXTENSIONS:
                discovered.extend(sorted(path.glob(f"**/*{extension}")))
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            discovered.append(path)
        else:
            LOGGER.warning("Skipping unsupported path: %s", path)
    return discovered


def main() -> None:
    """Entry point for the inference script."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()

    device = None
    if args.device:
        device = torch.device(args.device)

    model, device = load_trained_model(args.weights, device=device)
    transform = build_inference_transform()

    images = discover_images(args.paths)
    if not images:
        LOGGER.error("No valid images found under the provided paths.")
        raise SystemExit(1)

    LOGGER.info("Running inference on %d image(s).", len(images))
    for image_path in images:
        label, confidence, probabilities = predict_image(
            image_input=image_path,
            model=model,
            device=device,
            transform=transform,
        )
        LOGGER.info(
            "%s -> %s (confidence: %.2f%% | probs: %s)",
            image_path.name,
            label,
            confidence * 100,
            ", ".join(f"{value:.3f}" for value in probabilities.tolist()),
        )


if __name__ == "__main__":
    main()
