"""
Integration tests for the inference pipeline.

Tests cover:
- End-to-end: image input → CarDetector → annotated PIL output
- End-to-end: numpy (video frame) input → CarDetector → annotated PIL output
- Model switching: loading different model sizes produces valid results
- Pipeline consistency: same input produces consistent output across calls
- app.py module-level validation: model_options dict, class mapping logic

References: app.py L1–L215, detector.py L1–L112
"""

import pytest
import numpy as np
from PIL import Image
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detector import CarDetector


# ===================================================================
# TEST GROUP: End-to-end inference pipeline
# ===================================================================

class TestInferencePipelineE2E:
    """End-to-end inference: input → detect → output."""

    def test_pil_image_e2e(self, detector_nano, sample_car_image):
        """
        Full pipeline: PIL Image (Car.png) → detect() → annotated PIL + count.
        Covers: detector.py L59-L103
        """
        result_img, count = detector_nano.detect(
            sample_car_image, conf_threshold=0.3, classes=None
        )
        assert isinstance(result_img, Image.Image)
        assert isinstance(count, int)
        assert count >= 0
        assert result_img.mode == "RGB"
        assert result_img.size == sample_car_image.size

    def test_numpy_frame_e2e(self, detector_nano):
        """
        Full pipeline: numpy BGR frame (simulating cv2 video frame) → detect() → output.
        Covers: detector.py L59-L103, app.py L143-L145
        """
        # Simulate a video frame (BGR, as cv2.VideoCapture produces)
        frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        result_img, count = detector_nano.detect(frame, conf_threshold=0.5, classes=[2, 5, 7])
        assert isinstance(result_img, Image.Image)
        assert isinstance(count, int)
        assert result_img.size == (1280, 720)

    def test_detect_with_real_image_finds_objects(self, detector_nano, sample_car_image):
        """
        Car.png should produce at least 1 detection at low confidence.
        Validates the model actually performs inference, not just returning zeros.
        """
        _, count = detector_nano.detect(sample_car_image, conf_threshold=0.1, classes=None)
        assert count >= 1, "Expected at least 1 detection on Car.png at conf=0.1"


# ===================================================================
# TEST GROUP: Model switching (app.py L24–L38)
# ===================================================================

class TestModelSwitching:
    """Tests that different model sizes produce valid results."""

    def test_nano_and_medium_both_produce_output(self, detector_nano, detector_medium, pil_image):
        """
        app.py L24-29: Both Nano and Medium model variants should return
        valid (PIL.Image, int) tuples.
        """
        r1, c1 = detector_nano.detect(pil_image, conf_threshold=0.5)
        r2, c2 = detector_medium.detect(pil_image, conf_threshold=0.5)
        assert isinstance(r1, Image.Image)
        assert isinstance(r2, Image.Image)
        assert isinstance(c1, int) and c1 >= 0
        assert isinstance(c2, int) and c2 >= 0

    def test_available_model_files_exist(self):
        """
        Validate configured model files exist on disk.
        """
        expected_files = {
            'yolo26n.pt': True,
            'yolo26m.pt': True,
            'yolo26l.pt': True,
        }
        for filename, should_exist in expected_files.items():
            path = os.path.join(PROJECT_ROOT, filename)
            exists = os.path.isfile(path)
            if should_exist:
                assert exists, f"Expected model file '{filename}' to exist"


    def test_model_options_dict_validity(self):
        """
        app.py L24-29: Validate the model_options dict from app.py.
        All referenced .pt files should exist on disk.
        """
        # Reproduced from app.py L24-29
        model_options = {
            'Nano (Fastest)': 'yolo26n.pt',
            'Medium (Balanced)': 'yolo26m.pt',
            'Large (Best Accuracy)': 'yolo26l.pt'
        }
        missing = []
        for label, filename in model_options.items():
            if not os.path.isfile(os.path.join(PROJECT_ROOT, filename)):
                missing.append(f"{label} -> {filename}")

        assert len(missing) == 0, f"Unexpected missing models: {missing}"


# ===================================================================
# TEST GROUP: Pipeline determinism
# ===================================================================

class TestPipelineDeterminism:
    """Tests that inference is deterministic for the same input."""

    def test_same_input_same_count(self, detector_nano, sample_car_image):
        """
        detector.py L59: Same image + same params should give same count.
        """
        _, count1 = detector_nano.detect(sample_car_image, conf_threshold=0.3, classes=None)
        _, count2 = detector_nano.detect(sample_car_image, conf_threshold=0.3, classes=None)
        assert count1 == count2

    def test_same_input_same_output_size(self, detector_nano, sample_car_image):
        """
        detector.py L70: Output image size should be deterministic.
        """
        img1, _ = detector_nano.detect(sample_car_image, conf_threshold=0.3)
        img2, _ = detector_nano.detect(sample_car_image, conf_threshold=0.3)
        assert img1.size == img2.size


# ===================================================================
# TEST GROUP: app.py class mapping logic (app.py L59–L83)
# ===================================================================

class TestClassMappingLogic:
    """Tests for the class name ↔ ID mapping used in app.py."""

    def test_name_to_id_reverse_mapping(self, detector_nano):
        """
        app.py L68: name_to_id = {v: k for k, v in class_names_dict.items()}
        Verify this reverse mapping is correct.
        """
        class_names_dict = detector_nano.model_names
        name_to_id = {v: k for k, v in class_names_dict.items()}

        # Every name should map back to its original ID
        for class_id, class_name in class_names_dict.items():
            assert name_to_id[class_name] == class_id

    def test_default_targets_exist_in_model(self, detector_nano):
        """
        app.py L72-73: default_targets = ['car', 'truck', 'bus', 'motorcycle']
        All should be present in COCO class names.
        """
        name_values = set(detector_nano.model_names.values())
        default_targets = ['car', 'truck', 'bus', 'motorcycle']
        for target in default_targets:
            assert target in name_values, f"Default target '{target}' not in model names"

    def test_selected_classes_maps_to_valid_ids(self, detector_nano):
        """
        app.py L82: selected_classes = [name_to_id[n] for n in selected_names]
        Validate that mapped IDs are valid COCO IDs (0-79).
        """
        class_names_dict = detector_nano.model_names
        name_to_id = {v: k for k, v in class_names_dict.items()}
        default_targets = ['car', 'truck', 'bus', 'motorcycle']
        selected_classes = [name_to_id[n] for n in default_targets]
        for cls_id in selected_classes:
            assert 0 <= cls_id <= 79, f"Class ID {cls_id} out of COCO range"
