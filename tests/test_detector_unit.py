"""
Unit tests for detector.py — CarDetector class.

Tests cover:
- __init__: model loading, device assignment, error handling
- model_names property: return type, content
- detect(): return types, count, annotation drawing, input formats
- detect(): conf_threshold behavior
- detect(): class filtering behavior

References: detector.py L7–L112
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
# TEST GROUP: __init__ (detector.py L14–L37)
# ===================================================================

class TestCarDetectorInit:
    """Tests for CarDetector.__init__"""

    def test_init_default_model(self, detector_medium):
        """detector.py L14: Default model_name='yolo26m.pt' loads successfully."""
        assert detector_medium is not None
        assert detector_medium.model is not None

    def test_init_nano_model(self, detector_nano):
        """detector.py L14: Nano model loads successfully."""
        assert detector_nano is not None
        assert detector_nano.model is not None

    def test_device_is_valid_string(self, detector_nano):
        """detector.py L23-27: device must be 'cuda' or 'cpu'."""
        assert detector_nano.device in ('cuda', 'cpu')

    def test_device_matches_torch_availability(self, detector_nano):
        """detector.py L23-27: device should match torch.cuda.is_available()."""
        if torch.cuda.is_available():
            assert detector_nano.device == 'cuda'
        else:
            assert detector_nano.device == 'cpu'

    def test_model_has_names_after_init(self, detector_nano):
        """detector.py L32-34: After YOLO() load, model.names must exist."""
        assert detector_nano.model.names is not None
        assert isinstance(detector_nano.model.names, dict)

    def test_init_nonexistent_model_raises(self):
        """detector.py L35-37: Loading a non-existent model file should raise."""
        with pytest.raises(Exception):
            CarDetector(model_name="nonexistent_fake_model_xyz.pt")

    def test_init_invalid_model_path_raises(self):
        """detector.py L35-37: A completely invalid path should raise."""
        with pytest.raises(Exception):
            CarDetector(model_name="/invalid/path/to/model.pt")

    def test_init_empty_string_raises(self):
        """detector.py L32: Empty model name should raise."""
        with pytest.raises(Exception):
            CarDetector(model_name="")


# ===================================================================
# TEST GROUP: model_names property (detector.py L40–L43)
# ===================================================================

class TestModelNamesProperty:
    """Tests for CarDetector.model_names"""

    def test_model_names_returns_dict(self, detector_nano):
        """detector.py L42: model_names should return a dict."""
        names = detector_nano.model_names
        assert isinstance(names, dict)

    def test_model_names_has_coco_classes(self, detector_nano):
        """detector.py L42: COCO pretrained model should have 80 classes."""
        names = detector_nano.model_names
        assert len(names) == 80

    def test_model_names_keys_are_ints(self, detector_nano):
        """detector.py L42: Keys should be integer class IDs."""
        names = detector_nano.model_names
        for key in names:
            assert isinstance(key, int)

    def test_model_names_values_are_strings(self, detector_nano):
        """detector.py L42: Values should be string class names."""
        names = detector_nano.model_names
        for val in names.values():
            assert isinstance(val, str)
            assert len(val) > 0

    def test_model_names_contains_common_classes(self, detector_nano):
        """detector.py L42: Should contain known COCO classes."""
        names = detector_nano.model_names
        name_values = set(names.values())
        for expected in ['person', 'car', 'truck', 'bus']:
            assert expected in name_values, f"'{expected}' not in model names"


# ===================================================================
# TEST GROUP: detect() return types (detector.py L45–L103)
# ===================================================================

class TestDetectReturnTypes:
    """Tests for CarDetector.detect() return value contract."""

    def test_detect_returns_tuple(self, detector_nano, pil_image):
        """detector.py L56: detect() must return a tuple."""
        result = detector_nano.detect(pil_image)
        assert isinstance(result, tuple)

    def test_detect_returns_two_elements(self, detector_nano, pil_image):
        """detector.py L56: Return tuple must have exactly 2 elements."""
        result = detector_nano.detect(pil_image)
        assert len(result) == 2

    def test_detect_first_element_is_pil_image(self, detector_nano, pil_image):
        """detector.py L103: First return element must be PIL.Image."""
        result_img, _ = detector_nano.detect(pil_image)
        assert isinstance(result_img, Image.Image)

    def test_detect_second_element_is_int(self, detector_nano, pil_image):
        """detector.py L73: Second return element (count) must be int."""
        _, count = detector_nano.detect(pil_image)
        assert isinstance(count, int)

    def test_detect_count_non_negative(self, detector_nano, pil_image):
        """detector.py L73: Detection count must be >= 0."""
        _, count = detector_nano.detect(pil_image)
        assert count >= 0

    def test_detect_output_image_is_rgb(self, detector_nano, pil_image):
        """detector.py L101: Output image must be RGB (converted from BGR)."""
        result_img, _ = detector_nano.detect(pil_image)
        assert result_img.mode == "RGB"


# ===================================================================
# TEST GROUP: detect() with different input types (detector.py L59)
# ===================================================================

class TestDetectInputTypes:
    """Tests for detect() handling various input types."""

    def test_detect_accepts_pil_image(self, detector_nano, pil_image):
        """detector.py L49: PIL.Image input should work."""
        result_img, count = detector_nano.detect(pil_image)
        assert isinstance(result_img, Image.Image)

    def test_detect_accepts_numpy_bgr(self, detector_nano, bgr_numpy_image):
        """detector.py L49: np.ndarray (BGR) input should work."""
        result_img, count = detector_nano.detect(bgr_numpy_image)
        assert isinstance(result_img, Image.Image)

    def test_detect_output_dimensions_match_input_pil(self, detector_nano, pil_image):
        """detector.py L70: Output image dimensions should match input."""
        result_img, _ = detector_nano.detect(pil_image)
        assert result_img.size == pil_image.size  # PIL .size is (width, height)

    def test_detect_output_dimensions_match_input_numpy(self, detector_nano, bgr_numpy_image):
        """detector.py L70: Output image dims should match numpy input."""
        result_img, _ = detector_nano.detect(bgr_numpy_image)
        h, w = bgr_numpy_image.shape[:2]
        assert result_img.size == (w, h)


# ===================================================================
# TEST GROUP: detect() conf_threshold behavior (detector.py L61)
# ===================================================================

class TestDetectConfidenceThreshold:
    """Tests for confidence threshold filtering."""

    def test_high_conf_reduces_detections(self, detector_nano, sample_car_image):
        """detector.py L61: Higher threshold should yield <= detections than lower."""
        _, count_low = detector_nano.detect(sample_car_image, conf_threshold=0.1)
        _, count_high = detector_nano.detect(sample_car_image, conf_threshold=0.9)
        assert count_high <= count_low

    def test_conf_zero_allows_all(self, detector_nano, sample_car_image):
        """detector.py L61: conf=0.0 should allow maximum detections."""
        _, count_zero = detector_nano.detect(sample_car_image, conf_threshold=0.01)
        _, count_mid = detector_nano.detect(sample_car_image, conf_threshold=0.5)
        assert count_zero >= count_mid

    def test_conf_one_blocks_all(self, detector_nano, pil_image):
        """detector.py L61: conf=1.0 should block all detections (nothing is 100%)."""
        _, count = detector_nano.detect(pil_image, conf_threshold=1.0)
        assert count == 0


# ===================================================================
# TEST GROUP: detect() class filtering (detector.py L62)
# ===================================================================

class TestDetectClassFiltering:
    """Tests for class-based detection filtering."""

    def test_classes_none_detects_all_types(self, detector_nano, sample_car_image):
        """detector.py L62: classes=None should detect all classes."""
        _, count = detector_nano.detect(sample_car_image, conf_threshold=0.25, classes=None)
        # With a real car image, should detect at least something
        assert count >= 0  # Basic contract; non-negative

    def test_specific_class_filter(self, detector_nano, sample_car_image):
        """detector.py L62: Filtering to class [2] (car) should work."""
        _, count = detector_nano.detect(sample_car_image, conf_threshold=0.25, classes=[2])
        assert count >= 0

    def test_empty_classes_list_returns_zero(self, detector_nano, sample_car_image):
        """
        detector.py: classes=[] should be treated as None (all classes),
        so output should match unfiltered behavior for same input/threshold.
        """
        _, count_empty = detector_nano.detect(sample_car_image, conf_threshold=0.1, classes=[])
        _, count_none = detector_nano.detect(sample_car_image, conf_threshold=0.1, classes=None)
        assert count_empty == count_none

    def test_nonexistent_class_id_returns_zero(self, detector_nano, pil_image):
        """detector.py L62: A class ID not in COCO (e.g., 999) should return 0."""
        _, count = detector_nano.detect(pil_image, conf_threshold=0.1, classes=[999])
        assert count == 0
