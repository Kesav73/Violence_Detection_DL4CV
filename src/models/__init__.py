"""Model definitions for violence detection."""

from .violence_detection import (
    ModelConfig,
    ViolenceDetectionModel,
    build_vgg19_lstm_model,
    deserialize_config,
    load_violence_model,
    predict_violence_score,
    preprocess_frame,
    resolve_device,
    serialize_config,
)

__all__ = [
    "ModelConfig",
    "ViolenceDetectionModel",
    "build_vgg19_lstm_model",
    "load_violence_model",
    "predict_violence_score",
    "preprocess_frame",
    "resolve_device",
    "serialize_config",
    "deserialize_config",
]
