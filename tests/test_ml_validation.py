"""
ML-specific tests for model correctness, prediction validity,
output shape/type, and model artifact integrity.

Tests cover:
- Model loads with correct architecture
- Predictions produce valid bounding box coordinates
- Confidence scores are in [0, 1]
- Class IDs are within valid COCO range
- Output image dimensions are preserved (no accidental resize)
- Model weights file integrity
- Device placement correctness

References: detector.py L1–L112
"""

import pytest
import numpy as np
from PIL import Image
import torch
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detector import CarDetector


# ===================================================================
# TEST GROUP: Model architecture validation
# ===================================================================

class TestModelArchitecture:
    """Validate the loaded model's architecture properties."""

    def test_model_task_is_detect(self, detector_nano):
        """detector.py L32: Loaded model should be a detection model."""
        assert detector_nano.model.task == "detect"

    def test_model_has_80_classes(self, detector_nano):
        """detector.py L42: COCO pretrained should have exactly 80 classes."""
        assert len(detector_nano.model_names) == 80

    def test_model_device_placement(self, detector_nano):
        """detector.py L33-34: Model should be on the declared device."""
        device = detector_nano.device
        # The model's parameters should be on the correct device
        model_device = str(next(detector_nano.model.model.parameters()).device)
        if device == 'cuda':
            assert 'cuda' in model_device
        else:
            assert 'cpu' in model_device


# ===================================================================
# TEST GROUP: Prediction validity (bounding boxes)
# ===================================================================

class TestPredictionValidity:
    """Validate that raw model predictions are geometrically valid."""

    def test_bounding_box_format_xyxy(self, detector_nano, sample_car_image):
        """
        detector.py L81: box.xyxy[0] → (x1, y1, x2, y2).
        Validate x2 > x1 and y2 > y1 for all detections.
        """
        results = detector_nano.model.predict(
            source=sample_car_image, conf=0.1, verbose=False
        )
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            assert x2 > x1, f"Invalid box: x2 ({x2}) <= x1 ({x1})"
            assert y2 > y1, f"Invalid box: y2 ({y2}) <= y1 ({y1})"

    def test_bounding_boxes_within_image_bounds(self, detector_nano, sample_car_image):
        """
        detector.py L81: Bounding box coords should be within image dimensions.
        """
        w, h = sample_car_image.size  # PIL (width, height)
        results = detector_nano.model.predict(
            source=sample_car_image, conf=0.1, verbose=False
        )
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            assert x1 >= 0 and y1 >= 0, f"Negative coords: ({x1}, {y1})"
            # Allow 1px tolerance for rounding
            assert x2 <= w + 1, f"x2 ({x2}) exceeds width ({w})"
            assert y2 <= h + 1, f"y2 ({y2}) exceeds height ({h})"

    def test_confidence_scores_in_valid_range(self, detector_nano, sample_car_image):
        """
        detector.py L84: box.conf[0] should be in (0.0, 1.0].
        """
        results = detector_nano.model.predict(
            source=sample_car_image, conf=0.1, verbose=False
        )
        result = results[0]
        for box in result.boxes:
            conf = float(box.conf[0])
            assert 0.0 < conf <= 1.0, f"Confidence {conf} out of range"

    def test_class_ids_in_coco_range(self, detector_nano, sample_car_image):
        """
        detector.py L85: int(box.cls[0]) should be in [0, 79] for COCO.
        """
        results = detector_nano.model.predict(
            source=sample_car_image, conf=0.1, verbose=False
        )
        result = results[0]
        for box in result.boxes:
            cls_id = int(box.cls[0])
            assert 0 <= cls_id <= 79, f"Class ID {cls_id} out of COCO range [0, 79]"

    def test_class_ids_have_valid_names(self, detector_nano, sample_car_image):
        """
        detector.py L86: self.model.names[cls_id] must return a string.
        """
        results = detector_nano.model.predict(
            source=sample_car_image, conf=0.1, verbose=False
        )
        result = results[0]
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = detector_nano.model.names[cls_id]
            assert isinstance(name, str) and len(name) > 0


# ===================================================================
# TEST GROUP: Output shape/type preservation
# ===================================================================

class TestOutputShapePreservation:
    """Ensure detect() preserves input dimensions in output."""

    @pytest.mark.parametrize("width,height", [
        (640, 480),
        (1920, 1080),
        (320, 240),
        (1, 1),
    ])
    def test_output_size_matches_input_various_sizes(self, detector_nano, width, height):
        """
        detector.py L70, L103: Output PIL image size must match input size.
        YOLO internally resizes for inference but orig_img retains original dims.
        """
        arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        result_img, _ = detector_nano.detect(img, conf_threshold=0.5)
        assert result_img.size == (width, height)

    def test_output_numpy_shape_preserved(self, detector_nano, bgr_numpy_image):
        """
        detector.py L70: When numpy array input, output image matches dims.
        """
        h, w = bgr_numpy_image.shape[:2]
        result_img, _ = detector_nano.detect(bgr_numpy_image, conf_threshold=0.5)
        assert result_img.size == (w, h)


# ===================================================================
# TEST GROUP: Model weight file integrity
# ===================================================================

class TestModelWeightIntegrity:
    """Validate model weight files on disk."""

    @pytest.mark.parametrize("model_file", ["yolo26n.pt", "yolo26m.pt", "yolo26l.pt"])
    def test_model_file_exists(self, model_file):
        """Validate that each expected .pt file exists."""
        path = os.path.join(PROJECT_ROOT, model_file)
        assert os.path.isfile(path), f"Model file {model_file} not found"

    @pytest.mark.parametrize("model_file", ["yolo26n.pt", "yolo26m.pt", "yolo26l.pt"])
    def test_model_file_not_empty(self, model_file):
        """Model .pt files should have non-zero size."""
        path = os.path.join(PROJECT_ROOT, model_file)
        size = os.path.getsize(path)
        assert size > 0, f"Model file {model_file} is empty (0 bytes)"

    @pytest.mark.parametrize("model_file", ["yolo26n.pt", "yolo26m.pt", "yolo26l.pt"])
    def test_model_file_is_valid_torch(self, model_file):
        """Model .pt files should be loadable by torch (not corrupted)."""
        path = os.path.join(PROJECT_ROOT, model_file)
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            assert data is not None
        except Exception as e:
            pytest.fail(f"Failed to torch.load {model_file}: {e}")

    def test_missing_small_model(self):
        """
        Validate that removed unsupported small model file is not required.
        """
        path = os.path.join(PROJECT_ROOT, "yolo26s.pt")
        assert not os.path.isfile(path)


# ===================================================================
# TEST GROUP: Device handling
# ===================================================================

class TestDeviceHandling:
    """Validate correct CPU/GPU device handling."""

    def test_detect_runs_on_declared_device(self, detector_nano, pil_image):
        """
        detector.py L63: device is passed to model.predict().
        Should not raise device mismatch errors.
        """
        # This will raise if there's a device mismatch
        result_img, count = detector_nano.detect(pil_image, conf_threshold=0.5)
        assert isinstance(result_img, Image.Image)

    def test_cpu_fallback_works(self):
        """
        detector.py L23-27: If CUDA is not available, should fall back to CPU.
        This test always passes on CPU machines; on GPU machines it validates
        that the model CAN be loaded (doesn't test CPU fallback on GPU hardware).
        """
        detector = CarDetector(model_name="yolo26n.pt")
        assert detector.device in ('cuda', 'cpu')
        # Inference should work regardless of device
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result_img, _ = detector.detect(Image.fromarray(arr), conf_threshold=0.5)
        assert isinstance(result_img, Image.Image)
