"""Model architectures used for chest X-ray classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RadiographyCNN(nn.Module):
    """Convolutional neural network for binary chest X-ray classification."""

    def __init__(self) -> None:
        """Initialize convolutional, pooling, and fully connected layers."""

        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 26 * 26, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            inputs: Batch of grayscale chest X-ray tensors with shape ``(N, 1, 224, 224)``.

        Returns:
            Logits for the two output classes.
        """

        x = self.pool(F.relu(self.bn1(self.conv1(inputs))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
