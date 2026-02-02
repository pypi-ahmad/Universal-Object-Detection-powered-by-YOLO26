import cv2
from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np

class CarDetector:
    """
    CarDetector class using YOLO26 for object detection.
    This class handles loading the model and running inference to detect vehicles.
    It leverages YOLO26's NMS-Free technology for faster and more accurate detections.
    """

    def __init__(self, model_name='yolo26m.pt'):
        """
        Initialize the YOLO26 model.

        Args:
            model_name (str): Name of the YOLO model to load. Defaults to 'yolo26m.pt'.
                              The model will be downloaded automatically if not present.
        """
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("Warning: CUDA not available. Using CPU.")
            
        print(f"Initializing CarDetector with {model_name} on {self.device}...")
        try:
            self.model = YOLO(model_name)
            # Moving to device explicitly, though YOLO handles it often
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @property
    def model_names(self):
        """Returns the model's class names dictionary."""
        return self.model.names

    def detect(self, image, conf_threshold=0.25, classes=None):
        """
        Run detection on an image.

        Args:
            image (PIL.Image.Image or np.ndarray): Input image.
            conf_threshold (float): Confidence threshold for filtering weak detections.
            classes (list): List of COCO class IDs to detect. Defaults to None (all classes).

        Returns:
            tuple: (PIL.Image.Image, int) - Annotated image and detection count.
        """
        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            classes=classes,
            device=self.device,
            verbose=False
        )

        result = results[0]
        
        # Convert PIL image to OpenCV format (BGR) if necessary
        # result.orig_img is the original image as a numpy array (BGR)
        img_bgr = result.orig_img.copy()

        # Custom Drawing Logic
        detections = result.boxes
        count = len(detections)
        
        # Color: Light Green (144, 238, 144)
        box_color = (144, 238, 144)
        text_color = (0, 0, 0) # Black text for contrast

        for box in detections:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            
            # Label
            label = f"{class_name}: {conf:.0%}"
            
            # Draw Bounding Box
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw Label Background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_bgr, (x1, y1 - 20), (x1 + w, y1), box_color, -1)
            
            # Draw Text
            cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        # Convert BGR to RGB for PIL/Streamlit compatibility
        annotated_frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(annotated_frame_rgb), count

if __name__ == "__main__":
    # Test block
    try:
        detector = CarDetector()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
