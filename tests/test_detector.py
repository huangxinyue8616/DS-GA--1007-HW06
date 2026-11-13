from pathlib import Path

import pytest

from flange_inspector.detector import FlangeDetector, validate_inference_options


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_rejects_confidence_outside_unit_interval(confidence):
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        validate_inference_options(confidence, 640)


def test_rejects_non_positive_image_size():
    with pytest.raises(ValueError, match="positive integer"):
        validate_inference_options(0.25, 0)


def test_missing_weights_fail_before_loading_model(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        FlangeDetector(tmp_path / "missing.pt")
