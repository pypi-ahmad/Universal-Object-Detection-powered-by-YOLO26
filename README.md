# Object Detection using TensorFlow and TensorFlow Hub

This repository demonstrates how to perform object detection on custom images using TensorFlow and TensorFlow Hub in a Google Colab environment. The code clones the TensorFlow Models repository, installs necessary dependencies, and sets up the environment for object detection. It then uses a pre-trained model from TensorFlow Hub to perform object detection on a sample image from a custom dataset.

## Dependencies

- TensorFlow
- TensorFlow Hub
- OpenCV
- Matplotlib
- Protobuf Compiler
- Kaggle (to download the dataset)

## Setup

### 1. Clone the TensorFlow models repository:

```bash
!git clone --depth 1 https://github.com/tensorflow/models
```

### 2. Install Protobuf Compiler and set up the object detection API:

```bash
%%bash
sudo apt install -y protobuf-compiler
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

### 3. Import necessary libraries and utilities:

```python
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import io, random, os
import scipy.misc
import zipfile
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub

import cv2 as cv
from google.colab import drive
```

### 4. Set up the path to COCO labels:

```python
PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
```

### 5. Mount Google Drive (if using Google Colab) and set up Kaggle credentials:

```python
drive.mount('/content/drive')

!pip install kaggle

os.environ['KAGGLE_USERNAME'] = ""  # replace with your Kaggle username
os.environ['KAGGLE_KEY'] = ""  # replace with your Kaggle key
```

### 6. Download and extract the custom dataset:

```bash
!kaggle datasets download -d sshikamaru/car-object-detection
data = zipfile.ZipFile('/content/car-object-detection.zip', 'r')
data.extractall()
```

## Object Detection

### 1. Load and preprocess the image:

```python
path = r"/content/data/training_images"
random_filename = random.choice([
    x for x in os.listdir(path)
    if os.path.isfile(os.path.join(path, x))
])
path=path+'/'+random_filename
print(path)
image_np = cv.imread(path) 
image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
image_np = np.expand_dims(image_np, axis=0)
```

### 2. Visualize the image:

```python
plt.figure(figsize=(10, 10))
plt.imshow(image_np[0])
plt.show()
```

### 3. Prepare the image tensor and load the pre-trained model:

```python
image_tensor = tf.convert_to_tensor(image_np)
image_tensor = tf.cast(image_tensor, tf.uint8)

detector = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1")
```

### 4. Run object detection and visualize the results:

```python
results = detector(image_tensor)
result = {key: value.numpy() for key, value in results.items()}
print(result.keys())

label_id_offset = 0
image_np_with_detections = image_np.copy()

keypoints, keypoint_scores = None, None
if 'detection_keypoints' in result:
  keypoints = result['detection_keypoints'][0]
  keypoint_scores = result['detection_keypoint_scores'][0]

viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections[0],
      result['detection_boxes'][0],
      (result['detection_classes'][0] + label_id_offset).astype(int),
      result['detection_scores'][0],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.30,
      keypoint_scores=keypoint_scores
)

plt.figure(figsize=(24,32))
plt.imshow(image_np_with_detections[0])
plt.show()
```

## Note

- **Kaggle API Token**: Ensure that you replace the Kaggle username and key in the code with your own Kaggle credentials.
- **Google Colab**: This code is tailored for Google Colab and may require modifications if you wish to run it elsewhere.

## Author
- Ahmad Mujtaba

## Acknowledgements

- The code utilizes resources from the [TensorFlow Models repository](https://github.com/tensorflow/models) and [TensorFlow Hub](https://tfhub.dev/).
- Dataset sourced from Kaggle ([sshikamaru/car-object-detection](https://www.kaggle.com/sshikamaru/car-object-detection)).
