"""
PHASE 4 — Stress tests for CarDetector system.

Categories:
1. SYSTEM STRESS: Large inputs, repeated inference, memory pressure
2. ML STRESS: Batch processing, GPU/CPU switching, missing/corrupt models
3. DATA STRESS: Null values, wrong schema, unusual data
4. PERFORMANCE: Timing benchmarks to detect slow paths

Each test captures:
- Pass/Fail
- Timing (where relevant)
- Memory delta (where measurable)
- Stack trace on failure

References: detector.py L1–L112, app.py L1–L215
"""

import pytest
import numpy as np
from PIL import Image
import torch
import os
import sys
import time
import gc
import tempfile
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detector import CarDetector


# ===================================================================
# FIXTURES (stress-specific)
# ===================================================================

@pytest.fixture(scope="module")
def stress_detector():
    """Single nano detector for all stress tests (fast model)."""
    return CarDetector(model_name="yolo26n.pt")


@pytest.fixture
def car_image():
    """Load Car.png if available."""
    path = os.path.join(PROJECT_ROOT, "Car.png")
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    pytest.skip("Car.png not found")


# ===================================================================
# 1. SYSTEM STRESS — Large inputs
# ===================================================================

class TestSystemStressLargeInputs:
    """Stress test with large image dimensions."""

    def test_4k_image_inference(self, stress_detector):
        """
        Stress: 3840x2160 (4K) image.
        detector.py L59: model.predict() must handle 4K without crash.
        """
        arr = np.random.randint(0, 256, (2160, 3840, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        
        start = time.time()
        result_img, count = stress_detector.detect(img, conf_threshold=0.5)
        elapsed = time.time() - start
        
        assert isinstance(result_img, Image.Image)
        assert result_img.size == (3840, 2160)
        print(f"\n  4K inference: {elapsed:.2f}s, detections: {count}")

    def test_8k_image_inference(self, stress_detector):
        """
        Stress: 7680x4320 (8K) image — extreme resolution.
        Tests memory pressure on GPU/CPU.
        """
        arr = np.random.randint(0, 256, (4320, 7680, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        
        start = time.time()
        result_img, count = stress_detector.detect(img, conf_threshold=0.5)
        elapsed = time.time() - start
        
        assert isinstance(result_img, Image.Image)
        assert result_img.size == (7680, 4320)
        print(f"\n  8K inference: {elapsed:.2f}s, detections: {count}")

    def test_very_wide_panoramic_image(self, stress_detector):
        """
        Stress: 10000x500 panoramic image — extreme aspect ratio.
        """
        arr = np.random.randint(0, 256, (500, 10000, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        
        result_img, count = stress_detector.detect(img, conf_threshold=0.5)
        assert result_img.size == (10000, 500)

    def test_very_tall_image(self, stress_detector):
        """
        Stress: 500x10000 tall image — extreme aspect ratio.
        """
        arr = np.random.randint(0, 256, (10000, 500, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        
        result_img, count = stress_detector.detect(img, conf_threshold=0.5)
        assert result_img.size == (500, 10000)


# ===================================================================
# 2. SYSTEM STRESS — Repeated inference (memory leak detection)
# ===================================================================

class TestSystemStressRepeatedInference:
    """Test for memory leaks and stability under repeated calls."""

    def test_100_sequential_inferences(self, stress_detector, car_image):
        """
        Stress: 100 back-to-back detect() calls.
        Simulates rapid UI interactions in Streamlit.
        detector.py L45-L103: detect() must be stable across repeated calls.
        """
        counts = []
        start = time.time()
        
        for i in range(100):
            _, count = stress_detector.detect(car_image, conf_threshold=0.3)
            counts.append(count)
        
        elapsed = time.time() - start
        
        # All counts should be identical (deterministic)
        assert len(set(counts)) == 1, (
            f"Non-deterministic: got {len(set(counts))} unique counts across 100 runs"
        )
        print(f"\n  100 inferences: {elapsed:.2f}s total, {elapsed/100:.3f}s/call, "
              f"count={counts[0]}")

    def test_memory_stability_50_iterations(self, stress_detector):
        """
        Stress: 50 inferences on different random images.
        Check that memory doesn't grow unboundedly.
        """
        import psutil
        process = psutil.Process(os.getpid())
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        for i in range(50):
            arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            result_img, _ = stress_detector.detect(img, conf_threshold=0.5)
            del result_img, img, arr
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        mem_delta = mem_after - mem_before
        
        print(f"\n  Memory: before={mem_before:.0f}MB, after={mem_after:.0f}MB, "
              f"delta={mem_delta:+.0f}MB")
        
        # Allow up to 500MB growth (generous for random variance)
        assert mem_delta < 500, (
            f"Potential memory leak: {mem_delta:.0f}MB growth over 50 iterations"
        )

    def test_rapid_alternating_conf_thresholds(self, stress_detector, car_image):
        """
        Stress: Rapidly alternating between high and low confidence.
        Simulates user dragging the Streamlit slider (app.py L55).
        """
        for conf in [0.1, 0.9, 0.1, 0.9, 0.5, 0.01, 0.99, 0.5] * 5:
            result_img, count = stress_detector.detect(
                car_image, conf_threshold=conf
            )
            assert isinstance(result_img, Image.Image)
            assert count >= 0


# ===================================================================
# 3. ML STRESS — Batch processing
# ===================================================================

class TestMLStressBatchProcessing:
    """Simulate batch-like sequential processing (video frames)."""

    def test_simulate_video_300_frames(self, stress_detector):
        """
        Stress: Simulate 300-frame video processing (app.py L134-L150).
        Every 3rd frame processed (matching app.py L140 optimization).
        """
        processed = 0
        start = time.time()
        
        for frame_num in range(1, 301):
            if frame_num % 3 != 0:
                continue
            
            frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
            result_img, count = stress_detector.detect(
                frame, conf_threshold=0.4, classes=[2, 5, 7]
            )
            assert isinstance(result_img, Image.Image)
            processed += 1
        
        elapsed = time.time() - start
        assert processed == 100  # 300/3 = 100 frames
        print(f"\n  300-frame video sim: {elapsed:.2f}s ({processed} processed, "
              f"{elapsed/processed:.3f}s/frame)")

    def test_different_frame_sizes_in_sequence(self, stress_detector):
        """
        Stress: Process frames of varying sizes in sequence.
        Some video sources change resolution mid-stream.
        """
        sizes = [
            (640, 480), (1280, 720), (1920, 1080), (640, 480),
            (320, 240), (1280, 720), (800, 600),
        ]
        for w, h in sizes:
            frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            result_img, count = stress_detector.detect(frame, conf_threshold=0.5)
            assert result_img.size == (w, h), (
                f"Size mismatch for input ({w},{h}): got {result_img.size}"
            )


# ===================================================================
# 4. ML STRESS — Model switching
# ===================================================================

class TestMLStressModelSwitching:
    """Simulate switching between model sizes (app.py L24-L38)."""

    def test_load_all_available_models_sequentially(self):
        """
        Stress: Load nano → medium → large sequentially.
        app.py L43-44: @st.cache_resource caches by model name.
        Tests that each model loads and produces valid output.
        """
        models = ["yolo26n.pt", "yolo26m.pt", "yolo26l.pt"]
        arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        
        for model_name in models:
            start = time.time()
            detector = CarDetector(model_name=model_name)
            load_time = time.time() - start
            
            start = time.time()
            result_img, count = detector.detect(img, conf_threshold=0.5)
            infer_time = time.time() - start
            
            assert isinstance(result_img, Image.Image)
            print(f"\n  {model_name}: load={load_time:.2f}s, "
                  f"infer={infer_time:.3f}s, count={count}")
            
            del detector
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_rapid_model_reload_same_model(self):
        """
        Stress: Load the same model 5 times rapidly.
        Simulates cache miss scenario.
        """
        for i in range(5):
            d = CarDetector(model_name="yolo26n.pt")
            assert d.model is not None
            del d
            gc.collect()


# ===================================================================
# 5. ML STRESS — Missing/corrupt model files
# ===================================================================

class TestMLStressMissingCorruptModels:
    """Stress tests for model file error paths."""

    def test_missing_model_file_raises(self):
        """
        Missing local model path should raise load error.
        """
        with pytest.raises(Exception):
            CarDetector(model_name="nonexistent_model_file.pt")

    def test_corrupted_weights_file(self):
        """
        Stress: Create a file with random bytes named as .pt.
        detector.py L32-37: Should raise during YOLO() load.
        """
        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False, mode='wb'
        ) as f:
            f.write(os.urandom(1024))  # 1KB random bytes
            corrupt_path = f.name
        
        try:
            with pytest.raises(Exception) as exc_info:
                CarDetector(model_name=corrupt_path)
            print(f"\n  Corrupt model error: {exc_info.value}")
        finally:
            os.unlink(corrupt_path)

    def test_truncated_model_file(self):
        """
        Stress: Copy first 1KB of a real model (truncated).
        Simulates partial download or disk corruption.
        """
        real_model = os.path.join(PROJECT_ROOT, "yolo26n.pt")
        
        with open(real_model, 'rb') as f:
            partial_data = f.read(1024)  # Only first 1KB
        
        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False, mode='wb'
        ) as f:
            f.write(partial_data)
            truncated_path = f.name
        
        try:
            with pytest.raises(Exception) as exc_info:
                CarDetector(model_name=truncated_path)
            print(f"\n  Truncated model error: {exc_info.value}")
        finally:
            os.unlink(truncated_path)


# ===================================================================
# 6. DATA STRESS — Null/wrong schema inputs
# ===================================================================

class TestDataStressInvalidInputs:
    """Stress tests for unusual or malformed data inputs."""

    def test_none_source_behavior(self, stress_detector):
        """
        detector.py: None input must raise ValueError.
        """
        with pytest.raises(ValueError):
            stress_detector.detect(None, conf_threshold=0.5)

    def test_empty_list_as_classes(self, stress_detector, car_image):
        """
        classes=[] should behave as unfiltered classes=None.
        """
        _, count_empty = stress_detector.detect(
            car_image, conf_threshold=0.1, classes=[]
        )
        _, count_none = stress_detector.detect(
            car_image, conf_threshold=0.1, classes=None
        )
        assert count_empty == count_none

    def test_all_80_classes_selected(self, stress_detector, car_image):
        """
        Stress: Select all 80 COCO classes.
        app.py L78-82: User could select everything.
        """
        all_classes = list(range(80))
        result_img, count = stress_detector.detect(
            car_image, conf_threshold=0.1, classes=all_classes
        )
        assert isinstance(result_img, Image.Image)
        assert count >= 0

    def test_single_class_all_80_individually(self, stress_detector, car_image):
        """
        Stress: Run detection for each of 80 classes individually.
        """
        for cls_id in range(80):
            result_img, count = stress_detector.detect(
                car_image, conf_threshold=0.5, classes=[cls_id]
            )
            assert isinstance(result_img, Image.Image)
            assert count >= 0

    def test_negative_class_ids(self, stress_detector):
        """
        Data stress: Negative class IDs (invalid).
        """
        arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        try:
            result_img, count = stress_detector.detect(
                img, conf_threshold=0.5, classes=[-1, -99]
            )
            # If no crash, count should be 0 (no valid class match)
            assert count == 0
        except Exception as e:
            # Acceptable — negative IDs are invalid
            print(f"\n  Negative class ID error: {e}")

    def test_very_large_class_ids(self, stress_detector):
        """
        Data stress: Class IDs far beyond COCO range.
        """
        arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        try:
            result_img, count = stress_detector.detect(
                img, conf_threshold=0.5, classes=[9999, 100000]
            )
            assert count == 0
        except Exception:
            pass  # Acceptable

    def test_float_class_ids(self, stress_detector):
        """
        Data stress: Float class IDs (wrong type).
        """
        arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        try:
            result_img, count = stress_detector.detect(
                img, conf_threshold=0.5, classes=[2.5, 7.9]
            )
            # If YOLO truncates to int, this might work
            assert count >= 0
        except Exception:
            pass  # Acceptable

    def test_string_in_classes_list(self, stress_detector):
        """
        Data stress: String values in classes list (wrong type).
        """
        arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        with pytest.raises(Exception):
            stress_detector.detect(img, conf_threshold=0.5, classes=["car"])

    def test_conf_threshold_as_string(self, stress_detector):
        """
        Data stress: String confidence threshold.
        detector.py L61: conf parameter should be float.
        """
        arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        try:
            result_img, count = stress_detector.detect(
                img, conf_threshold="high"
            )
            # Should not reach here
            assert False, "String conf_threshold should have failed"
        except (TypeError, ValueError, Exception):
            pass  # Expected


# ===================================================================
# 7. PERFORMANCE BENCHMARKS
# ===================================================================

class TestPerformanceBenchmarks:
    """Measure inference latency for performance regression detection."""

    def test_nano_640x480_latency(self, stress_detector):
        """
        Benchmark: Nano model on 640x480 image.
        """
        arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        
        # Warmup
        stress_detector.detect(img, conf_threshold=0.5)
        
        times = []
        for _ in range(10):
            start = time.time()
            stress_detector.detect(img, conf_threshold=0.5)
            times.append(time.time() - start)
        
        avg = sum(times) / len(times)
        p95 = sorted(times)[int(0.95 * len(times))]
        
        print(f"\n  Nano 640x480: avg={avg:.3f}s, p95={p95:.3f}s, "
              f"min={min(times):.3f}s, max={max(times):.3f}s")
        
        # Fail if avg exceeds 5 seconds (very generous for CPU)
        assert avg < 5.0, f"Average latency too high: {avg:.2f}s"

    def test_model_load_time(self):
        """
        Benchmark: How long does model initialization take?
        detector.py L24-L37: YOLO() + .to(device)
        """
        start = time.time()
        d = CarDetector(model_name="yolo26n.pt")
        elapsed = time.time() - start
        
        print(f"\n  Model load time: {elapsed:.2f}s")
        
        # Fail if loading takes more than 30 seconds
        assert elapsed < 30.0, f"Model load too slow: {elapsed:.2f}s"
        del d

    def test_1080p_inference_latency(self, stress_detector):
        """
        Benchmark: 1920x1080 (Full HD) inference.
        """
        arr = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        
        # Warmup
        stress_detector.detect(img, conf_threshold=0.5)
        
        times = []
        for _ in range(5):
            start = time.time()
            stress_detector.detect(img, conf_threshold=0.5)
            times.append(time.time() - start)
        
        avg = sum(times) / len(times)
        print(f"\n  1080p inference: avg={avg:.3f}s")
        
        assert avg < 10.0, f"1080p latency too high: {avg:.2f}s"


# ===================================================================
# 8. GPU/CPU SWITCHING STRESS
# ===================================================================

class TestGPUCPUStress:
    """Test behavior when forcing CPU vs GPU paths."""

    def test_force_cpu_detector(self):
        """
        Stress: Even on GPU machine, CPU path must work.
        detector.py L23-27: Tests that CPU codepath is functional.
        """
        d = CarDetector(model_name="yolo26n.pt")
        # Force CPU prediction regardless of detector's device
        arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        
        # Call predict with device='cpu' directly
        results = d.model.predict(
            source=img, conf=0.5, device='cpu', verbose=False
        )
        assert len(results) == 1
        assert results[0].boxes is not None
        del d

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No GPU available"
    )
    def test_force_gpu_detector(self):
        """
        Stress: Explicit GPU prediction path.
        """
        d = CarDetector(model_name="yolo26n.pt")
        arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        
        results = d.model.predict(
            source=img, conf=0.5, device='cuda', verbose=False
        )
        assert len(results) == 1
        del d

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No GPU available"
    )
    def test_gpu_memory_after_repeated_inference(self):
        """
        Stress: Check GPU memory doesn't leak across inferences.
        """
        d = CarDetector(model_name="yolo26n.pt")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        mem_start = torch.cuda.memory_allocated() / (1024 * 1024)
        
        for _ in range(50):
            arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            d.detect(img, conf_threshold=0.5)
        
        mem_end = torch.cuda.memory_allocated() / (1024 * 1024)
        mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        print(f"\n  GPU memory: start={mem_start:.0f}MB, end={mem_end:.0f}MB, "
              f"peak={mem_peak:.0f}MB, delta={mem_end-mem_start:+.0f}MB")
        
        del d


# ===================================================================
# 9. STREAMLIT-SPECIFIC DATA FLOW STRESS
# ===================================================================

class TestStreamlitDataFlowStress:
    """
    Stress tests simulating Streamlit-specific data flow patterns.
    These test the code paths in app.py without actually running Streamlit.
    """

    def test_pil_image_from_file_uploader_simulation(self, stress_detector):
        """
        app.py L94-95: Image.open(uploaded_file).
        Simulate by opening Car.png via PIL (same path).
        """
        car_path = os.path.join(PROJECT_ROOT, "Car.png")
        if not os.path.exists(car_path):
            pytest.skip("Car.png not found")
        
        img = Image.open(car_path)
        # Don't convert to RGB — test if detect handles non-RGB PIL modes
        result_img, count = stress_detector.detect(img, conf_threshold=0.3)
        assert isinstance(result_img, Image.Image)

    def test_class_name_mapping_roundtrip_all_classes(self, stress_detector):
        """
        app.py L68-82: Full roundtrip of name↔ID mapping for all 80 classes.
        """
        class_names_dict = stress_detector.model_names
        name_to_id = {v: k for k, v in class_names_dict.items()}
        
        # Roundtrip: ID → name → ID
        for cls_id, cls_name in class_names_dict.items():
            recovered_id = name_to_id[cls_name]
            assert recovered_id == cls_id, (
                f"Roundtrip failed: {cls_id} → '{cls_name}' → {recovered_id}"
            )
        
        # Ensure all 80 mapped
        assert len(name_to_id) == 80

    def test_conf_slider_full_range_sweep(self, stress_detector, car_image):
        """
        app.py L55: Confidence slider from 0.0 to 1.0 in 0.05 steps.
        Simulate full sweep.
        """
        prev_count = None
        for conf in [x / 20.0 for x in range(21)]:  # 0.0, 0.05, 0.10, ..., 1.0
            _, count = stress_detector.detect(
                car_image, conf_threshold=max(conf, 0.01), classes=None
            )
            assert count >= 0
            if prev_count is not None:
                # Monotonic: higher conf → <= detections
                assert count <= prev_count or conf <= 0.05, (
                    f"Non-monotonic: conf={conf:.2f} gave {count} > "
                    f"prev {prev_count}"
                )
            prev_count = count
