"""Train and evaluate the COVID-19 chest X-ray classification model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from covid_xray import (
    TrainingConfig,
    evaluate_model,
    set_global_seed,
    train_model,
)
from covid_xray.model import RadiographyCNN
from covid_xray.training import create_dataloaders, create_datasets, get_device

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Returns:
        Parsed command-line arguments namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train the RadiographyCNN model on the COVID-19 Radiography Database "
            "and export metrics for reproducibility."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("COVID-Data-Radiography"),
        help=(
            "Path to the dataset root directory. The folder must contain 'no/' "
            "and 'yes/' sub-directories."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where training artifacts will be saved.",
    )
    parser.add_argument(
        "--best-model",
        type=Path,
        default=Path("best_model.pth"),
        help="Path where the best-performing model weights are stored.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Mini-batch size used during training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes for the data loaders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs without improvement before early stopping.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain the model even if pretrained weights already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the training script."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    set_global_seed(args.seed)

    config = TrainingConfig(
        dataset_root=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        best_model_path=args.best_model,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.force_retrain and config.best_model_path.exists():
        LOGGER.info(
            "Found existing model weights at %s. Skipping training.",
            config.best_model_path.resolve(),
        )
        device = get_device()
        model = RadiographyCNN().to(device)
        state_dict = torch.load(config.best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        datasets = create_datasets(config)
        dataloaders = create_dataloaders(config, datasets)
        _, _, test_loader = dataloaders
    else:
        model, device, dataloaders = train_model(config)
        _, _, test_loader = dataloaders

    test_loss, test_accuracy = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        output_dir=config.output_dir,
    )
    LOGGER.info("Test loss: %.4f", test_loss)
    LOGGER.info("Test accuracy: %.4f", test_accuracy)


if __name__ == "__main__":
    main()
