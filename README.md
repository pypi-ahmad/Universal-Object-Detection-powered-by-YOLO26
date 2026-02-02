# OmniDetect AI 🦅

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![YOLO26](https://img.shields.io/badge/YOLO-26-green)

## Universal Object Detection powered by YOLO26

**OmniDetect AI** is a state-of-the-art object detection system leveraging the **NMS-Free YOLO26 architecture**. Optimized for high-performance inference on NVIDIA RTX 4060 GPUs, it delivers zero-latency tracking and robust detection across diverse scenarios.

### ✨ Key Features

*   **🦅 Universal Detection**: Capable of detecting all **80 COCO classes**, including vehicles, pedestrians, animals, and everyday objects.
*   **⚡ Dynamic Model Switching**: Instantly swap between **Nano** (High Speed) and **Large** (High Accuracy) models without restarting.
*   **🎨 Custom Visualization**: Professional "Light Green" bounding boxes with high-contrast confidence scores for clear visibility.
*   **📹 Multi-Source Support**: Seamlessly analyze static images, video files, or live webcam streams.

### 🛠 Tech Stack

*   **Ultralytics YOLO26**: Next-gen object detection with End-to-End architecture.
*   **Streamlit**: Interactive, real-time web dashboard.
*   **OpenCV**: Advanced image processing.
*   **NVIDIA CUDA**: Accelerated GPU inference.

### 🚀 Getting Started

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YourUsername/OmniDetect-AI.git
    cd OmniDetect-AI
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

### � Deployment & Builds

#### 🐳 Docker (Recommended)
The most robust way to run OmniDetect AI is via Docker, ensuring all system dependencies (OpenCV/GL libraries) are correctly installed.

```bash
docker-compose up --build
```

#### 🪟 Windows Executable (.exe)
You can build a standalone executable for easy distribution on Windows machines.
1.  **Install PyInstaller**: `pip install pyinstaller`
2.  **Build**:
    ```bash
    pyinstaller --onefile --additional-hooks-dir=./hooks run.py
    ```
    *(Note: You may need to create a simple `run.py` wrapper script that calls `streamlit run app.py` via `subprocess` or `sys` for the executable logic).*

#### 📱 Android Support
For mobile use, we recommend deploying this application to a cloud server (e.g., Streamlit Cloud, HuggingFace Spaces) and accessing it via the mobile browser (`Chrome` or `Safari`).
*   **Why?** Native Android APKs require converting YOLO models to TFLite or ONNX, which may reduce the specific performance benefits of the dynamic PyTorch switching used here.


