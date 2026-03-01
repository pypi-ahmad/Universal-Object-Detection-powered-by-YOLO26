"""
Shared fixtures for the test suite.

Provides reusable detector instances and test images to avoid
repeated model loading across test modules.
"""

import pytest
import numpy as np
from PIL import Image
import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detector import CarDetector


# ---------------------------------------------------------------------------
# Model fixtures (session-scoped to load only once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def detector_nano():
    """Load the Nano model once for the entire test session."""
    return CarDetector(model_name="yolo26n.pt")


@pytest.fixture(scope="session")
def detector_medium():
    """Load the Medium model once for the entire test session."""
    return CarDetector(model_name="yolo26m.pt")


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rgb_numpy_image():
    """A synthetic 640x480 RGB numpy array (uint8)."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def bgr_numpy_image():
    """A synthetic 640x480 BGR numpy array (uint8), simulating cv2.imread output."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def pil_image():
    """A synthetic 640x480 RGB PIL Image."""
    arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def small_pil_image():
    """A very small 32x32 RGB PIL Image for boundary testing."""
    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def large_pil_image():
    """A large 4000x3000 RGB PIL Image for stress testing."""
    arr = np.random.randint(0, 256, (3000, 4000, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def grayscale_numpy_image():
    """A single-channel 640x480 grayscale numpy array."""
    return np.random.randint(0, 256, (480, 640), dtype=np.uint8)


@pytest.fixture
def car_png_path():
    """Path to the Car.png sample image in the repo root."""
    path = os.path.join(PROJECT_ROOT, "Car.png")
    if os.path.exists(path):
        return path
    pytest.skip("Car.png not found in project root")


@pytest.fixture
def sample_car_image(car_png_path):
    """Load Car.png as a PIL Image."""
    return Image.open(car_png_path).convert("RGB")
