from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class ModelConfig:
    """Configuration shared by PyTorch training and inference."""

    image_size: Tuple[int, int] = (160, 160)
    sequence_length: int = 40
    num_classes: int = 2
    learning_rate: float = 5e-4
    lstm_hidden_size: int = 40
    temporal_dense_size: int = 160
    classifier_hidden_size: int = 512
    dropout: float = 0.3


class ViolenceDetectionModel(nn.Module):
    """
    PyTorch implementation of the original repo's VGG19 + LSTM idea.

    Input shape:
    - `(batch_size, sequence_length, 3, height, width)`

    Output shape:
    - `(batch_size, 2)` logits for `non-violence` and `violence`
    """

    def __init__(self, config: ModelConfig, pretrained_backbone: bool = False) -> None:
        super().__init__()
        self.config = config

        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = models.vgg19(weights=weights)
        self.feature_extractor = backbone.features

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.image_size[0], config.image_size[1])
            feature_dim = int(torch.flatten(self.feature_extractor(dummy), start_dim=1).shape[1])

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=config.lstm_hidden_size,
            batch_first=True,
        )
        self.frame_dense = nn.Linear(config.lstm_hidden_size, config.temporal_dense_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.classifier_hidden = nn.Linear(
            config.temporal_dense_size,
            config.classifier_hidden_size,
        )
        self.classifier = nn.Linear(config.classifier_hidden_size, config.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = inputs.shape

        frames = inputs.reshape(batch_size * sequence_length, channels, height, width)
        features = self.feature_extractor(frames)
        features = torch.flatten(features, start_dim=1)
        features = features.reshape(batch_size, sequence_length, -1)

        lstm_output, _ = self.lstm(features)
        temporal_features = self.relu(self.frame_dense(lstm_output))
        pooled_features = temporal_features.mean(dim=1)
        hidden = self.relu(self.classifier_hidden(pooled_features))
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        return logits


def build_vgg19_lstm_model(
    config: ModelConfig,
    pretrained_backbone: bool = False,
) -> ViolenceDetectionModel:
    """Build the shared PyTorch model architecture."""

    return ViolenceDetectionModel(config, pretrained_backbone=pretrained_backbone)


def preprocess_frame(frame_bgr: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Resize and normalize an OpenCV BGR frame for model inference."""

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, image_size)
    return frame_resized.astype(np.float32) / 255.0


def resolve_device(requested_device: str | None = None) -> torch.device:
    """Resolve the target torch device."""

    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def serialize_config(config: ModelConfig) -> dict[str, Any]:
    """Convert the config dataclass to a checkpoint-friendly dictionary."""

    return asdict(config)


def deserialize_config(data: dict[str, Any] | None) -> ModelConfig:
    """Rebuild a model config from checkpoint metadata."""

    if not data:
        return ModelConfig()

    config_data = dict(data)
    if "image_size" in config_data:
        config_data["image_size"] = tuple(config_data["image_size"])
    return ModelConfig(**config_data)


def load_violence_model(
    model_path: str | Path,
    config: ModelConfig | None = None,
    device: str | torch.device | None = None,
) -> ViolenceDetectionModel:
    """
    Load a trained PyTorch violence model.

    Supported checkpoint formats:
    - raw state dict saved with `torch.save(model.state_dict(), path)`
    - checkpoint dictionary containing `model_state_dict`
    """

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    torch_device = resolve_device(str(device) if device is not None else None)
    checkpoint = torch.load(model_path, map_location=torch_device)

    checkpoint_config = None
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_config = checkpoint.get("config")

    final_config = config or deserialize_config(checkpoint_config)
    model = build_vgg19_lstm_model(final_config, pretrained_backbone=False)
    model.load_state_dict(state_dict)
    model.to(torch_device)
    model.eval()
    return model


def predict_violence_score(model: nn.Module, clip_frames: np.ndarray) -> float:
    """
    Predict the violence probability for a single clip tensor.

    `clip_frames` must be shaped `(sequence_length, height, width, 3)`.
    """

    device = next(model.parameters()).device
    clip_tensor = torch.from_numpy(clip_frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(clip_tensor)
        probabilities = torch.softmax(logits, dim=1)

    return float(probabilities[0, 1].item())
