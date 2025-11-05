"""Inference helpers for the COVID-19 chest X-ray classification project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from .model import RadiographyCNN
from .training import get_device

LOGGER = logging.getLogger(__name__)
CLASS_NAMES = ("COVID-negative", "COVID-positive")
ImageInput = Union[str, Path, Image.Image]


def build_inference_transform() -> transforms.Compose:
    """Return the default transform pipeline used during inference.

    Returns:
        Compose: Torchvision transform matching the training preprocessing.
    """

    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def load_trained_model(
    weights_path: str | Path,
    device: Optional[torch.device] = None,
) -> Tuple[RadiographyCNN, torch.device]:
    """Load a trained :class:`RadiographyCNN` from disk.

    Args:
        weights_path: Location of the ``.pth`` file containing model weights.
        device: Optional device override used for mapping the weights.

    Returns:
        Tuple with the instantiated model and the device used for inference.

    Raises:
        FileNotFoundError: If ``weights_path`` does not exist.
    """

    resolved_path = Path(weights_path)
    if not resolved_path.exists():
        msg = f"Model weights not found at {resolved_path}"
        LOGGER.error(msg)
        raise FileNotFoundError(msg)

    if device is None:
        device = get_device()
    model = RadiographyCNN().to(device)
    state_dict = torch.load(resolved_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    LOGGER.info("Loaded model weights from %s", resolved_path.resolve())
    return model, device


def _load_image(image_input: ImageInput) -> Image.Image:
    """Load an image from a path or return an existing :class:`Image.Image`.

    Args:
        image_input: Path to an image file or a Pillow image instance.

    Returns:
        ``Image.Image`` representation of the provided input.

    Raises:
        FileNotFoundError: If ``image_input`` is a path that cannot be located.
    """

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    resolved_path = Path(image_input)
    if not resolved_path.exists():
        msg = f"Image not found at {resolved_path}"
        LOGGER.error(msg)
        raise FileNotFoundError(msg)
    return Image.open(resolved_path).convert("RGB")


def predict_image(
    image_input: ImageInput,
    model: nn.Module,
    device: torch.device,
    transform: Optional[transforms.Compose] = None,
) -> Tuple[str, float, torch.Tensor]:
    """Run inference on a single image and return the prediction.

    Args:
        image_input: Path to an image or ``Image.Image`` instance for inference.
        model: Neural network used to compute predictions.
        device: Device on which the model operates.
        transform: Optional preprocessing transform to apply before inference.

    Returns:
        Tuple containing the predicted label, confidence score, and raw
        probabilities tensor on the CPU.
    """

    transform = transform or build_inference_transform()
    image = _load_image(image_input)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    predicted_index = int(torch.argmax(probabilities).item())
    label = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index].item())
    return label, confidence, probabilities.cpu()
