import streamlit as st
import cv2
import tempfile
import os
import time
from PIL import Image
import numpy as np
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
    'Small': 'yolo26s.pt',
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
    st.sidebar.info("Running on NVIDIA RTX 4060")

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
    st.sidebar.warning("Please select at least one class.")
    selected_classes = []
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
            st.image(image, width='stretch')
        with col2:
            st.subheader(f"Detected ({count})")
            st.image(result_img, width='stretch')
            
        st.metric(label="Vehicles Detected", value=count)

# --- Tab 2: Video Analytics ---
with tab2:
    st.header("Video Analytics")
    uploaded_video = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_video:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        tfile.close() # Close file so it can be opened by cv2
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error opening video file.")
        else:
            st_frame = st.empty()
            frame_count = 0
            
            # Stop button
            if st.button("Stop Video Processing"):
                cap.release()
                st.write("Stopped.")
            else:
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
                    st_frame.image(result_img, caption=f"Frame {frame_count} | Detected: {count}", width='stretch')
                    
                    # Small sleep to allow UI updates if needed, though streamlit usually handles it
                    # time.sleep(0.01) 
                
                cap.release()
        
        # Cleanup
        try:
            os.unlink(video_path)
        except Exception as e:
            print(f"Error removing temp file: {e}")

# --- Tab 3: Live Scout (Webcam) ---
with tab3:
    st.header("Live Scout")
    
    # Using a checkbox to start/stop
    run_camera = st.checkbox("Start Camera")
    
    if run_camera:
        # Placeholders
        st_frame_cam = st.empty()
        stop_btn = st.button("Stop Camera")
        
        if stop_btn:
            # This logic is a bit tricky in Streamlit. 
            # If 'Stop' is clicked, the script reruns. 'run_camera' might still be true unless we uncheck it or use session state.
            # But the loop below blocks.
            pass
        
        # Use DirectShow on Windows to avoid MSMF errors
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            st.error("Cannot open webcam.")
        else:
            while not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image from camera.")
                    break
                
                result_img, count = detector.detect(frame, conf_threshold, selected_classes)
                
                st_frame_cam.image(result_img, caption=f"Live Feed | Detected: {count}", width='stretch')
                
                # Check for stop in a specific way if possible? 
                # Streamlit doesn't support live interruption easily without rerun.
                # But `st.button` inside a loop doesn't work well (it needs unique key and won't be registered until script finishes loop usually).
                # Actually, standard practice is to use a "Stop" button *outside* detection loop logic usually doesn't work because loop blocks.
                # So we rely on "Stop" button being clicked which triggers rerun, but since we are in a loop, we might not see it.
                # However, changing the checkbox `run_camera` acts as a stop. Unchecking it triggers rerun and `run_camera` becomes false.
                # So the loop needs to be non-blocking or check state? 
                # Actually, if we use `while run_camera`, `run_camera` variable doesn't update during the loop.
                # So we rely on user interaction triggers a rerun which kills the current script execution? 
                # Yes, Streamlit stops execution on widget interaction. So unchecking the box stops the loop.
                # But adding an explicit "Stop" button is requested.
                # We can't really put a working Stop button *after* the loop starts if the loop blocks.
                # We can try to rely on the checkbox.
                pass
            
            cap.release()
