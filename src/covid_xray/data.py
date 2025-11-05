"""Dataset utilities for the COVID-19 chest X-ray classification project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

LOGGER = logging.getLogger(__name__)
EXPECTED_CLASS_SUBDIRS = ("no", "yes")


def validate_dataset_structure(dataset_root: str | Path) -> bool:
    """Validate that the dataset directory follows the expected structure.

    Args:
        dataset_root: Path to the root directory of the dataset.

    Returns:
        ``True`` if the expected directory structure is present, ``False`` otherwise.
    """

    root_path = Path(dataset_root)
    if not root_path.exists():
        LOGGER.error("Dataset directory %s does not exist.", root_path)
        return False

    missing_subdirs = [
        subdir for subdir in EXPECTED_CLASS_SUBDIRS if not (root_path / subdir).is_dir()
    ]
    if missing_subdirs:
        LOGGER.error(
            "Dataset directory %s is missing required sub-directories: %s",
            root_path,
            ", ".join(sorted(missing_subdirs)),
        )
        return False
    return True


class CTDataset(Dataset):
    """Wrapper around :class:`torchvision.datasets.ImageFolder` for CT scans.

    Args:
        root_dir: Path to the dataset root directory.
        transform: Optional callable transform to apply to the images.

    Raises:
        FileNotFoundError: If ``root_dir`` does not exist or the expected
            sub-directories are missing.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        root_path = Path(root_dir)
        if not validate_dataset_structure(root_path):
            raise FileNotFoundError(
                "Dataset not found or missing required sub-directories. "
                "Expected 'no/' and 'yes/' folders inside the dataset root."
            )
        self.data = ImageFolder(root=str(root_path), transform=transform)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Total number of images in the dataset.
        """

        return len(self.data)

    def __getitem__(self, index: int):
        """Return a single sample from the dataset.

        Args:
            index: Position of the desired sample.

        Returns:
            Tuple of image tensor and label index.
        """

        return self.data[index]
