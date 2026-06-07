# Getting Started - Implementation Guide

## Quick Start (First 2 Hours)

### Step 1: Clone/Create Project Structure

```bash
cd d:\Sem 7\computer vision\project
mkdir -p data/raw data/processed models/trained src tests docs
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Or Linux/Mac
source venv/bin/activate
```

### Step 3: Create requirements.txt

```
torch==2.0.1
torchvision==0.15.2
ultralytics==8.0.200
opencv-python==4.8.1.78
opencv-contrib-python==4.8.1.78
numpy==1.24.3
roboflow==1.1.36
pillow==10.0.0
matplotlib==3.7.2
pandas==2.0.3
pyyaml==6.0
```

Install: `pip install -r requirements.txt`

### Step 4: Download Datasets

```bash
# Create roboflow account and download via:
# pip install roboflow
# Then in Python:
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("").project("medical-pills")
dataset = project.download("yolov8")
```

---

## Phase 1: Basic Structure (Starter Code)

### `src/config.yaml`

```yaml
# Dataset Configuration
dataset:
  train_size: 0.7
  val_size: 0.15
  test_size: 0.15
  img_size: 640

# Model Configuration
model:
  name: yolov8m # yolov8n, yolov8s, yolov8m, yolov8l
  pretrained: true

# Training Configuration
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  patience: 30
  device: 0 # GPU device ID

# Inference Configuration
inference:
  confidence_threshold: 0.5
  nms_threshold: 0.45
  max_detections: 100

# Performance
performance:
  target_fps: 30
  max_latency_ms: 50
```

### `src/preprocessing.py`

```python
import cv2
import numpy as np
from pathlib import Path

class ImagePreprocessor:
    def __init__(self, img_size=640):
        self.img_size = img_size

    def resize_image(self, image):
        """Resize image to target size."""
        return cv2.resize(image, (self.img_size, self.img_size))

    def apply_gaussian_blur(self, image, kernel_size=5):
        """Apply Gaussian blur for noise reduction."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def apply_median_filter(self, image, kernel_size=5):
        """Apply median filter for noise reduction."""
        return cv2.medianBlur(image, kernel_size)

    def normalize_brightness(self, image):
        """Normalize image brightness using CLAHE."""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            return cv2.merge([l, a, b])
        return image

    def preprocess_frame(self, frame):
        """Complete preprocessing pipeline."""
        frame = self.resize_image(frame)
        frame = self.apply_gaussian_blur(frame)
        frame = self.normalize_brightness(frame)
        return frame

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ImagePreprocessor()
    test_image = cv2.imread("sample.jpg")
    processed = preprocessor.preprocess_frame(test_image)
    cv2.imwrite("processed_sample.jpg", processed)
```

### `src/dataset_loader.py`

```python
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class PillDataLoader:
    def __init__(self, data_dir, img_size=640):
        self.data_dir = Path(data_dir)
        self.img_size = img_size

    def load_dataset(self):
        """Load images and annotations."""
        images = list(self.data_dir.glob("**/*.jpg")) + \
                 list(self.data_dir.glob("**/*.png"))

        # Load corresponding annotations
        annotations = []
        for img_path in images:
            ann_path = img_path.with_suffix('.txt')
            if ann_path.exists():
                with open(ann_path) as f:
                    annotations.append(f.read())

        return images, annotations

    def split_dataset(self, images, annotations,
                     train_size=0.7, val_size=0.15):
        """Split dataset into train/val/test."""
        # First split: train + val/test
        train_imgs, temp_imgs, train_anns, temp_anns = train_test_split(
            images, annotations, test_size=(1-train_size),
            random_state=42
        )

        # Second split: val and test
        val_size_ratio = val_size / (1 - train_size)
        val_imgs, test_imgs, val_anns, test_anns = train_test_split(
            temp_imgs, temp_anns, test_size=(1-val_size_ratio),
            random_state=42
        )

        return {
            'train': (train_imgs, train_anns),
            'val': (val_imgs, val_anns),
            'test': (test_imgs, test_anns)
        }

if __name__ == "__main__":
    loader = PillDataLoader("data/raw")
    images, annotations = loader.load_dataset()
    splits = loader.split_dataset(images, annotations)
    print(f"Train: {len(splits['train'][0])}")
    print(f"Val: {len(splits['val'][0])}")
    print(f"Test: {len(splits['test'][0])}")
```

### `src/model_trainer.py`

```python
from ultralytics import YOLO
import yaml

class PillDetectionTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def prepare_dataset_yaml(self, dataset_path):
        """Create dataset.yaml for YOLO."""
        dataset_config = {
            'path': str(dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # Number of classes (pills)
            'names': ['pill']
        }

        with open(dataset_path / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f)

        return dataset_path / 'dataset.yaml'

    def train(self, dataset_yaml, output_dir='models/trained'):
        """Train YOLO model."""
        model = YOLO(f"yolov8{self.config['model']['name'][-1]}.pt")

        results = model.train(
            data=str(dataset_yaml),
            epochs=self.config['training']['epochs'],
            imgsz=self.config['dataset']['img_size'],
            batch=self.config['training']['batch_size'],
            lr0=self.config['training']['learning_rate'],
            patience=self.config['training']['patience'],
            device=self.config['training']['device'],
            project=output_dir,
            name='pill_detector',
            save=True,
            val=True,
        )

        return results, model

    def evaluate(self, model, dataset_yaml):
        """Evaluate trained model."""
        metrics = model.val(data=str(dataset_yaml))
        return metrics

if __name__ == "__main__":
    trainer = PillDetectionTrainer("src/config.yaml")
    # trainer.train("data/processed/dataset.yaml")
```

### `src/inference.py`

```python
import cv2
import time
from ultralytics import YOLO

class RealTimeDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.pill_count = 0

    def detect_pills(self, frame):
        """Detect pills in a frame."""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        return results[0]

    def draw_boxes(self, frame, detections):
        """Draw bounding boxes on frame."""
        boxes = detections.boxes

        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Pill {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def run_realtime(self, camera_id=0):
        """Run real-time detection from camera."""
        cap = cv2.VideoCapture(camera_id)
        self.pill_count = 0
        fps_counter = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect pills
            detections = self.detect_pills(frame)
            pill_count = len(detections.boxes)
            self.pill_count += pill_count if pill_count <= 5 else 0  # Avoid noise

            # Draw results
            frame = self.draw_boxes(frame, detections)

            # Display stats
            cv2.putText(frame, f"Count: {self.pill_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Calculate FPS
            fps_counter += 1
            if fps_counter % 10 == 0:
                elapsed = time.time() - start_time
                fps = 10 / elapsed
                start_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Pill Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RealTimeDetector("models/trained/best.pt")
    # detector.run_realtime()
```

---

## Running the Pipeline

### 1. Prepare Data

```bash
python src/dataset_loader.py
```

### 2. Train Model

```bash
python src/model_trainer.py
```

### 3. Run Real-Time Detection

```bash
python src/inference.py
```

---

## Debugging Checklist

- [ ] Dataset properly formatted (YOLO format)
- [ ] GPU available: `torch.cuda.is_available()`
- [ ] Model file exists before inference
- [ ] Camera accessible: `cv2.VideoCapture(0)` works
- [ ] Sufficient disk space for model training
- [ ] Dependencies installed correctly

---

## Next Immediate Actions

1. **Today**: Set up project structure and download datasets
2. **Tomorrow**: Implement preprocessing pipeline
3. **This week**: Get first model training running
4. **Next week**: Implement real-time inference

Good luck! 🚀
