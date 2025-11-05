"""Training utilities for the COVID-19 chest X-ray classification project."""

from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms

from .data import CTDataset
from .model import RadiographyCNN

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - fallback when NumPy is unavailable
    np = None

LOGGER = logging.getLogger(__name__)
CLASS_MAPPING = {0: "No Covid", 1: "Yes Covid"}


@dataclass(slots=True)
class TrainingConfig:
    """Configuration container used for training runs."""

    dataset_root: Path
    batch_size: int = 8
    num_workers: int = 2
    seed: int = 1
    epochs: int = 20
    patience: int = 3
    learning_rate: float = 1e-4
    output_dir: Path = Path("outputs")
    best_model_path: Path = Path("best_model.pth")


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Deterministic seed used across libraries.
    """

    import random

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)


def build_default_transform() -> transforms.Compose:
    """Return the default transform pipeline used during training.

    Returns:
        Compose: Torchvision transform replicating the original training pipeline.
    """

    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def create_datasets(config: TrainingConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """Create the training, validation, and test dataset splits.

    Args:
        config: Training configuration describing dataset and random seed.

    Returns:
        Tuple containing the training, validation, and test datasets.
    """

    transform = build_default_transform()
    dataset = CTDataset(config.dataset_root, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    remainder = len(dataset) - train_size - val_size
    return random_split(
        dataset,
        [train_size, val_size, remainder],
        generator=torch.Generator().manual_seed(config.seed),
    )


def create_dataloaders(
    config: TrainingConfig,
    datasets: Tuple[Dataset, Dataset, Dataset],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for the training, validation, and test splits.

    Args:
        config: Training configuration describing data loader parameters.
        datasets: Pre-split dataset tuple returned by :func:`create_datasets`.

    Returns:
        Tuple containing the training, validation, and test data loaders.
    """

    train_dataset, val_dataset, test_dataset = datasets
    labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    label_counts = Counter(labels)
    weights = [1.0 / label_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def get_device() -> torch.device:
    """Return the preferred device for training.

    Returns:
        torch.device: Available MPS, CUDA, or CPU device.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    """Run a single training or evaluation epoch.

    Args:
        model: Model under training or evaluation.
        dataloader: Data loader supplying mini-batches.
        criterion: Loss criterion used for optimization.
        optimizer: Optional optimizer; if provided, the epoch performs training.
        device: Target device for computation.

    Returns:
        Tuple containing the mean loss and accuracy for the epoch.
    """

    is_training = optimizer is not None
    model.train(mode=is_training)
    if is_training:
        optimizer.zero_grad()

    total_loss = 0.0
    correct_predictions = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        outputs = model(features)
        loss = criterion(outputs, targets)
        if is_training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        correct_predictions += (outputs.argmax(1) == targets).sum().item()

    return total_loss / len(dataloader), correct_predictions / len(dataloader.dataset)


def train_model(
    config: TrainingConfig,
) -> Tuple[nn.Module, torch.device, Tuple[DataLoader, DataLoader, DataLoader]]:
    """Train the :class:`RadiographyCNN` model using the provided configuration.

    Args:
        config: Training configuration with dataset, optimizer, and runtime details.

    Returns:
        Tuple containing the trained model, selected device, and data loaders.
    """

    set_global_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    LOGGER.info("Using device: %s", device)

    datasets = create_datasets(config)
    dataloaders = create_dataloaders(config, datasets)
    train_loader, val_loader, test_loader = dataloaders

    model = RadiographyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    csv_path = config.output_dir / "training_log.csv"
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        train_losses: list[float] = []
        val_losses: list[float] = []
        train_accuracies: list[float] = []
        val_accuracies: list[float] = []

        for epoch in range(1, config.epochs + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
            scheduler.step(val_loss)
            learning_rate = optimizer.param_groups[0]["lr"]

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.4f}",
                    f"{train_acc:.4f}",
                    f"{val_loss:.4f}",
                    f"{val_acc:.4f}",
                    f"{learning_rate:.6f}",
                ]
            )

            LOGGER.info(
                "Epoch %d: train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), config.best_model_path)
                LOGGER.info(
                    "New best model saved to %s", config.best_model_path.resolve()
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    LOGGER.info("Early stopping triggered.")
                    break

    plot_training_curves(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        config.output_dir,
    )

    if config.best_model_path.exists():
        model.load_state_dict(torch.load(config.best_model_path, map_location=device))

    return model, device, dataloaders


def plot_training_curves(
    train_losses: Iterable[float],
    val_losses: Iterable[float],
    train_accuracies: Iterable[float],
    val_accuracies: Iterable[float],
    output_dir: Path,
) -> None:
    """Plot training and validation curves and save them to disk.

    Args:
        train_losses: Sequence of training loss values.
        val_losses: Sequence of validation loss values.
        train_accuracies: Sequence of training accuracy values.
        val_accuracies: Sequence of validation accuracy values.
        output_dir: Directory where the plot will be written.
    """

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    LOGGER.info("Training curves saved to %s", output_path.resolve())


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> Tuple[float, float]:
    """Evaluate the model and export metrics to disk.

    Args:
        model: Trained model to evaluate.
        dataloader: Data loader providing evaluation data.
        device: Device on which evaluation is performed.
        output_dir: Directory for saving evaluation artifacts.

    Returns:
        Tuple with the evaluation loss and accuracy.
    """

    criterion = nn.CrossEntropyLoss()
    loss, accuracy = run_epoch(model, dataloader, criterion, None, device)

    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs.to(device))
            predictions.extend(outputs.argmax(1).cpu().numpy().tolist())
            labels.extend(targets.numpy().tolist())

    report = classification_report(
        labels,
        predictions,
        target_names=list(CLASS_MAPPING.values()),
        digits=4,
    )
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report)
    LOGGER.info("Classification report saved to %s", report_path.resolve())

    confusion = confusion_matrix(labels, predictions)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(CLASS_MAPPING.values()),
        yticklabels=list(CLASS_MAPPING.values()),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = output_dir / "cm.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    LOGGER.info("Confusion matrix saved to %s", cm_path.resolve())

    return loss, accuracy
