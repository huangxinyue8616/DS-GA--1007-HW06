"""Small inference wrapper shared by the CLI and web application."""

from pathlib import Path
from typing import Any


def load_yolo(weights: str):
    """Load an Ultralytics YOLO model, reporting a useful dependency error."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    return YOLO(weights)


def validate_inference_options(confidence: float, image_size: int) -> None:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0.")
    if image_size <= 0:
        raise ValueError("Image size must be a positive integer.")


class FlangeDetector:
    """Run defect detection with a trained YOLO weights file."""

    def __init__(self, weights: str | Path):
        weights_path = Path(weights).expanduser()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        self.model = load_yolo(str(weights_path))

    def predict(
        self,
        source: Any,
        confidence: float = 0.25,
        image_size: int = 640,
        **options: Any,
    ):
        validate_inference_options(confidence, image_size)
        return self.model.predict(
            source=source,
            conf=confidence,
            imgsz=image_size,
            verbose=False,
            **options,
        )
