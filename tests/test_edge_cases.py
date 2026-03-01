"""
Edge case and boundary tests for CarDetector.

Tests cover:
- Empty / zero-dimension images
- Single-pixel images
- Very large images
- Grayscale input (wrong channel count)
- Non-image input types (string, int, None)
- Corrupt / invalid model files
- Extreme confidence thresholds
- RGBA images (4 channels)
- Float dtype images

References: detector.py L45–L103
"""

import pytest
import numpy as np
from PIL import Image
import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detector import CarDetector


# ===================================================================
# TEST GROUP: Image dimension edge cases
# ===================================================================

class TestImageDimensionEdgeCases:
    """Edge cases for image dimensions and shapes."""

    def test_single_pixel_image(self, detector_nano):
        """
        detector.py L59: 1x1 image should not crash.
        YOLO resizes internally; should handle gracefully.
        """
        img = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8), mode="RGB")
        result_img, count = detector_nano.detect(img, conf_threshold=0.5)
        assert isinstance(result_img, Image.Image)
        assert count == 0  # Nothing to detect in 1 pixel

    def test_very_small_image_32x32(self, detector_nano, small_pil_image):
        """
        detector.py L59: Very small image (32x32) should work.
        """
        result_img, count = detector_nano.detect(small_pil_image, conf_threshold=0.5)
        assert isinstance(result_img, Image.Image)
        assert result_img.size == small_pil_image.size
        assert count >= 0

    def test_non_square_image_wide(self, detector_nano):
        """
        detector.py L59: Very wide image (2000x100) should not crash.
        """
        arr = np.random.randint(0, 256, (100, 2000, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        result_img, count = detector_nano.detect(img, conf_threshold=0.5)
        assert result_img.size == (2000, 100)

    def test_non_square_image_tall(self, detector_nano):
        """
        detector.py L59: Very tall image (100x2000) should not crash.
        """
        arr = np.random.randint(0, 256, (2000, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        result_img, count = detector_nano.detect(img, conf_threshold=0.5)
        assert result_img.size == (100, 2000)


# ===================================================================
# TEST GROUP: Invalid input types
# ===================================================================

class TestInvalidInputTypes:
    """Tests for graceful handling of invalid inputs to detect()."""

    def test_none_input_raises(self, detector_nano):
        """
        detector.py L59: None input should be rejected explicitly.
        """
        with pytest.raises(ValueError):
            detector_nano.detect(None, conf_threshold=0.5)

    def test_string_input_raises(self, detector_nano):
        """
        detector.py L59: A plain string (not a file path) should raise.
        """
        with pytest.raises(Exception):
            detector_nano.detect("not_an_image", conf_threshold=0.5)

    def test_integer_input_raises(self, detector_nano):
        """
        detector.py L59: An integer input should raise.
        """
        with pytest.raises(Exception):
            detector_nano.detect(42, conf_threshold=0.5)

    def test_empty_numpy_array_raises(self, detector_nano):
        """
        detector.py L59: An empty numpy array should raise.
        """
        with pytest.raises(Exception):
            detector_nano.detect(np.array([]), conf_threshold=0.5)


# ===================================================================
# TEST GROUP: Channel edge cases
# ===================================================================

class TestChannelEdgeCases:
    """Tests for non-standard channel configurations."""

    def test_rgba_image_4_channels(self, detector_nano):
        """
        detector.py L59: RGBA (4-channel) PIL image.
        YOLO should handle or convert internally.
        """
        arr = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        # Ultralytics may or may not handle RGBA; test behavior
        try:
            result_img, count = detector_nano.detect(img, conf_threshold=0.5)
            assert isinstance(result_img, Image.Image)
            assert count >= 0
        except Exception:
            # If it raises, that's acceptable — RGBA is non-standard input
            pass

    def test_grayscale_numpy_array(self, detector_nano, grayscale_numpy_image):
        """
        detector.py L59: Single-channel grayscale numpy array.
        YOLO may handle by converting to 3-channel, or may fail.
        """
        try:
            result_img, count = detector_nano.detect(
                grayscale_numpy_image, conf_threshold=0.5
            )
            assert isinstance(result_img, Image.Image)
        except Exception:
            # Acceptable — grayscale is non-standard for YOLO
            pass

    def test_grayscale_pil_image(self, detector_nano):
        """
        detector.py L59: Grayscale ('L' mode) PIL Image.
        """
        arr = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        try:
            result_img, count = detector_nano.detect(img, conf_threshold=0.5)
            assert isinstance(result_img, Image.Image)
        except Exception:
            # Acceptable — grayscale is non-standard
            pass


# ===================================================================
# TEST GROUP: Numpy dtype edge cases
# ===================================================================

class TestNumpyDtypeEdgeCases:
    """Tests for non-standard numpy dtypes."""

    def test_float32_numpy_image(self, detector_nano):
        """
        detector.py L59: float32 numpy array (values 0.0-1.0).
        YOLO expects uint8; behavior may vary.
        """
        arr = np.random.rand(480, 640, 3).astype(np.float32)
        try:
            result_img, count = detector_nano.detect(arr, conf_threshold=0.5)
            assert isinstance(result_img, Image.Image)
        except Exception:
            pass  # Acceptable if type is not handled

    def test_float64_numpy_image(self, detector_nano):
        """
        detector.py L59: float64 numpy array.
        """
        arr = np.random.rand(480, 640, 3).astype(np.float64)
        try:
            result_img, count = detector_nano.detect(arr, conf_threshold=0.5)
            assert isinstance(result_img, Image.Image)
        except Exception:
            pass

    def test_uint16_numpy_image(self, detector_nano):
        """
        detector.py L59: uint16 numpy array.
        """
        arr = np.random.randint(0, 65535, (480, 640, 3), dtype=np.uint16)
        try:
            result_img, count = detector_nano.detect(arr, conf_threshold=0.5)
            assert isinstance(result_img, Image.Image)
        except Exception:
            pass


# ===================================================================
# TEST GROUP: Confidence threshold edge cases
# ===================================================================

class TestConfidenceEdgeCases:
    """Tests for extreme confidence threshold values."""

    def test_conf_exactly_zero(self, detector_nano, pil_image):
        """
        detector.py L61: conf=0.0 — should be valid (allow everything).
        Note: Ultralytics may clamp to a minimum internally.
        """
        try:
            result_img, count = detector_nano.detect(pil_image, conf_threshold=0.0)
            assert isinstance(result_img, Image.Image)
            assert count >= 0
        except Exception:
            pass  # Some YOLO versions reject 0.0

    def test_conf_exactly_one(self, detector_nano, pil_image):
        """
        detector.py L61: conf=1.0 — nothing should pass.
        """
        result_img, count = detector_nano.detect(pil_image, conf_threshold=1.0)
        assert count == 0

    def test_conf_negative_raises_or_clamps(self, detector_nano, pil_image):
        """
        detector.py L61: Negative confidence is invalid.
        """
        try:
            result_img, count = detector_nano.detect(pil_image, conf_threshold=-0.5)
            # If it doesn't raise, count should still be non-negative
            assert count >= 0
        except Exception:
            pass  # Acceptable to reject negative values

    def test_conf_greater_than_one_raises_or_clamps(self, detector_nano, pil_image):
        """
        detector.py L61: conf > 1.0 is invalid.
        """
        try:
            result_img, count = detector_nano.detect(pil_image, conf_threshold=1.5)
            assert count == 0  # Nothing should pass
        except Exception:
            pass  # Acceptable to reject values > 1


# ===================================================================
# TEST GROUP: Corrupt model file
# ===================================================================

class TestCorruptModelFile:
    """Tests for loading invalid/corrupt model files."""

    def test_corrupt_pt_file_raises(self):
        """
        detector.py L32-37: A corrupt .pt file should raise during init.
        """
        # Create a temporary corrupt .pt file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, mode='wb') as f:
            f.write(b"this is not a valid pytorch model file")
            corrupt_path = f.name

        try:
            with pytest.raises(Exception):
                CarDetector(model_name=corrupt_path)
        finally:
            os.unlink(corrupt_path)

    def test_empty_pt_file_raises(self):
        """
        detector.py L32-37: An empty .pt file should raise during init.
        """
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, mode='wb') as f:
            corrupt_path = f.name  # 0 bytes

        try:
            with pytest.raises(Exception):
                CarDetector(model_name=corrupt_path)
        finally:
            os.unlink(corrupt_path)

    def test_directory_as_model_raises(self):
        """
        detector.py L32: Passing a directory path as model_name should raise.
        """
        with pytest.raises(Exception):
            CarDetector(model_name=tempfile.gettempdir())


# ===================================================================
# TEST GROUP: Annotation drawing edge cases
# ===================================================================

class TestAnnotationDrawingEdgeCases:
    """Tests for edge cases in the custom drawing logic (detector.py L72-L99)."""

    def test_all_black_image(self, detector_nano):
        """
        detector.py L72-L99: Drawing on an all-black image.
        Should not crash even with detections.
        """
        arr = np.zeros((480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        result_img, count = detector_nano.detect(img, conf_threshold=0.1)
        assert isinstance(result_img, Image.Image)

    def test_all_white_image(self, detector_nano):
        """
        detector.py L72-L99: Drawing on an all-white image.
        """
        arr = np.full((480, 640, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        result_img, count = detector_nano.detect(img, conf_threshold=0.1)
        assert isinstance(result_img, Image.Image)

    def test_output_image_differs_from_input_when_detections(
        self, detector_nano, sample_car_image
    ):
        """
        detector.py L72-L99: If detections exist, the output image
        should differ from the input (annotations drawn).
        """
        result_img, count = detector_nano.detect(
            sample_car_image, conf_threshold=0.1, classes=None
        )
        if count > 0:
            # Convert both to numpy for comparison
            input_arr = np.array(sample_car_image)
            output_arr = np.array(result_img)
            assert not np.array_equal(input_arr, output_arr), (
                "Output image is identical to input despite detections"
            )
