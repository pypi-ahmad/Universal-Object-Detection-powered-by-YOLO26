import streamlit as st
import cv2
import tempfile
import os
import time
from PIL import Image
from detector import CarDetector

# Page Config
st.set_page_config(
    page_title="OmniDetect AI 🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("OmniDetect AI 🦅")
st.markdown("### Universal Object Detection powered by YOLO26")

# Sidebar Configuration
st.sidebar.header("Model Configuration")

# Model Size Dropdown
model_options = {
    'Nano (Fastest)': 'yolo26n.pt',
    'Medium (Balanced)': 'yolo26m.pt',
    'Large (Best Accuracy)': 'yolo26l.pt'
}

model_selection = st.sidebar.selectbox(
    "Model Size",
    options=list(model_options.keys()),
    index=2 # Default to Medium
)

selected_model_file = model_options[model_selection]

# Initialize Detector
@st.cache_resource
def get_detector(model_name):
    return CarDetector(model_name=model_name)

try:
    detector = get_detector(selected_model_file)
except Exception as e:
    st.error(f"Failed to load detector: {e}")
    st.stop()

# Performance Info
if detector.device == 'cuda':
    st.sidebar.info(f"Running on {detector.device_name}")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

# Class Selection
# Get class names from the model
if detector.model_names:
    class_names_dict = detector.model_names
else:
    # Fallback if model names are not available immediately (though they should be)
    class_names_dict = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck'} # minimal fallback

# Create a mapping from Name -> ID for the UI
name_to_id = {v: k for k, v in class_names_dict.items()}
sorted_names = sorted(name_to_id.keys())

# Default selection
default_targets = ['car', 'truck', 'bus', 'motorcycle']
default_selection = [name for name in default_targets if name in sorted_names]

selected_names = st.sidebar.multiselect(
    "Target Objects",
    options=sorted_names,
    default=default_selection
)

# Map selected names to IDs
if not selected_names:
    st.sidebar.info("No class filter selected. Detecting all classes.")
    selected_classes = None
else:
    selected_classes = [name_to_id[n] for n in selected_names]

# Tabs
tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Analytics", "🔴 Live Scout"])

# --- Tab 1: Image Analysis ---
with tab1:
    st.header("Image Analysis")
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg', 'webp'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Run detection
        # Ensure classes are passed
        result_img, count = detector.detect(image, conf_threshold, selected_classes)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader(f"Detected ({count})")
            st.image(result_img, use_container_width=True)
            
        st.metric(label="Objects Detected", value=count)

# --- Tab 2: Video Analytics ---
with tab2:
    st.header("Video Analytics")
    uploaded_video = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_video:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_video.read())
            video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error opening video file.")
        else:
            st_frame = st.empty()
            frame_count = 0
            
            # Stop button
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Optimization: Process every 3rd frame
                if frame_count % 3 != 0:
                    continue
                
                # Convert Frame to Image for display/detect (detect handles numpy BGR but returns PIL RGB)
                # To pass to detector: frame is BGR numpy array
                result_img, count = detector.detect(frame, conf_threshold, selected_classes)
                
                # Display
                st_frame.image(result_img, caption=f"Frame {frame_count} | Detected: {count}", use_container_width=True)
                
                # Small sleep to allow UI updates
                time.sleep(0.01)
                
            cap.release()
        
        # Cleanup
        try:
            os.unlink(video_path)
        except Exception as e:
            print(f"Error removing temp file: {e}")

# --- Tab 3: Live Scout (Webcam) ---
with tab3:
    st.header("Live Scout")

    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False
    if "camera_cap" not in st.session_state:
        st.session_state.camera_cap = None

    st.checkbox("Start Camera", key="run_camera")
    if st.button("Stop Camera"):
        st.session_state.run_camera = False

    if st.session_state.run_camera:
        st_frame_cam = st.empty()

        if st.session_state.camera_cap is None:
            st.session_state.camera_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        cap = st.session_state.camera_cap
        if not cap.isOpened():
            st.error("Cannot open webcam.")
            st.session_state.run_camera = False
        else:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from camera.")
                st.session_state.run_camera = False
            else:
                result_img, count = detector.detect(frame, conf_threshold, selected_classes)
                st_frame_cam.image(result_img, caption=f"Live Feed | Detected: {count}", use_container_width=True)
                time.sleep(0.01)
                st.rerun()
    else:
        if st.session_state.camera_cap is not None:
            st.session_state.camera_cap.release()
            st.session_state.camera_cap = None
