"""
Real-Time Inference Module
Handles real-time pill detection from camera feeds.
"""

import cv2
import torch
import time
from pathlib import Path
from typing import Tuple, Optional, List
from ultralytics import YOLO
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeDetector:
    """Real-time pill detection from camera feed."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 nms_threshold: float = 0.45, device: str = "0"):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold
            device: GPU device ID or 'cpu'
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Automatically fall back to CPU if CUDA is not available
        if device != 'cpu' and not torch.cuda.is_available():
            logger.info("CUDA not available. Automatically falling back to CPU for inference.")
            self.device = 'cpu'
        else:
            self.device = device
        
        # Explicitly move the model to the target device and override default device settings
        try:
            self.model.to(self.device)
            self.model.overrides['device'] = self.device
        except Exception as e:
            logger.warning(f"Failed to set model device overrides: {e}")
        
        self.pill_count = 0
        self.frame_count = 0
        self.detection_history = []
        
        logger.info(f"RealTimeDetector initialized with model: {model_path}")
    
    def detect_pills(self, frame: np.ndarray) -> any:
        """
        Detect pills in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Detection results
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            device=self.device,
            verbose=False
        )
        return results[0]
    
    def draw_boxes(self, frame: np.ndarray, detections: any,
                  thickness: int = 2, text_scale: float = 0.6) -> np.ndarray:
        """
        Draw bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: Detection results
            thickness: Box thickness
            text_scale: Text scale
            
        Returns:
            Frame with drawn boxes
        """
        frame_with_boxes = frame.copy()
        boxes = detections.boxes
        
        for i, box in enumerate(boxes):
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            
            # Draw box
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # Draw label with confidence
            label = f"Pill {i+1} ({conf:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           text_scale, thickness)
            
            # Background for label
            cv2.rectangle(frame_with_boxes, (x1, y1 - 25),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Text
            cv2.putText(frame_with_boxes, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), thickness)
        
        return frame_with_boxes
    
    def draw_stats(self, frame: np.ndarray, fps: float = 0,
                  count: int = 0, processing_time: float = 0) -> np.ndarray:
        """
        Draw statistics on frame.
        
        Args:
            frame: Input frame
            fps: Frames per second
            count: Current pill count
            processing_time: Time taken for inference (ms)
            
        Returns:
            Frame with statistics
        """
        frame_with_stats = frame.copy()
        
        # Background for stats
        cv2.rectangle(frame_with_stats, (0, 0), (350, 100), (0, 0, 0), -1)
        cv2.rectangle(frame_with_stats, (0, 0), (350, 100), (0, 255, 0), 2)
        
        # Draw text
        y_offset = 25
        cv2.putText(frame_with_stats, f"Pill Count: {count}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame_with_stats, f"FPS: {fps:.1f}",
                   (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        cv2.putText(frame_with_stats, f"Latency: {processing_time:.1f}ms",
                   (10, y_offset + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        return frame_with_stats
    
    def run_realtime(self, camera_id: int = 0, frame_skip: int = 1,
                    display: bool = True) -> None:
        """
        Run real-time detection from camera.
        
        Args:
            camera_id: Camera device ID
            frame_skip: Process every nth frame (1 = all frames)
            display: Whether to display the video
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        # Get camera properties
        fps_cap = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened. FPS: {fps_cap}")
        
        self.pill_count = 0
        self.frame_count = 0
        fps_counter = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Skip frames if specified
                if self.frame_count % frame_skip != 0:
                    continue
                
                # Inference
                inference_start = time.time()
                detections = self.detect_pills(frame)
                inference_time = (time.time() - inference_start) * 1000  # Convert to ms
                
                # Count pills
                pill_count = len(detections.boxes)
                
                # Draw results
                frame_with_boxes = self.draw_boxes(frame, detections)
                
                # Calculate FPS
                fps_counter += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    fps = fps_counter / elapsed
                    fps_counter = 0
                    start_time = time.time()
                else:
                    fps = 0
                
                # Update count (simple strategy: use current detection)
                self.pill_count = pill_count
                
                # Draw statistics
                frame_final = self.draw_stats(frame_with_boxes, fps=fps, 
                                             count=self.pill_count,
                                             processing_time=inference_time)
                
                if display:
                    cv2.imshow("Pill Counter", frame_final)
                
                # Check for quit signal
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit signal received")
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info(f"Total frames processed: {self.frame_count}")
            logger.info(f"Final pill count: {self.pill_count}")
    
    def detect_from_image(self, image_path: str) -> Tuple[np.ndarray, any]:
        """
        Detect pills in a single image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (processed_frame, detections)
        """
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Failed to read image: {image_path}")
            return None, None
        
        detections = self.detect_pills(frame)
        frame_with_boxes = self.draw_boxes(frame, detections)
        
        return frame_with_boxes, detections
    
    def detect_from_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Detect pills in a video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save output video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect pills
                detections = self.detect_pills(frame)
                frame_with_boxes = self.draw_boxes(frame, detections)
                
                # Draw count
                count = len(detections.boxes)
                cv2.putText(frame_with_boxes, f"Count: {count}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if out:
                    out.write(frame_with_boxes)
                
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        finally:
            cap.release()
            if out:
                out.release()
            logger.info(f"Video processing completed. Total frames: {frame_count}")


class DetectionStatistics:
    """Tracks and analyzes detection statistics."""
    
    def __init__(self):
        self.detections = []
        self.confidences = []
        self.frame_counts = []
    
    def add_detection(self, num_pills: int, avg_confidence: float) -> None:
        """Record a detection result."""
        self.detections.append(num_pills)
        self.confidences.append(avg_confidence)
        self.frame_counts.append(len(self.frame_counts) + 1)
    
    def get_statistics(self) -> dict:
        """Get detection statistics."""
        if not self.detections:
            return {}
        
        detections_array = np.array(self.detections)
        confidences_array = np.array(self.confidences)
        
        return {
            'total_frames': len(self.detections),
            'avg_detections': float(np.mean(detections_array)),
            'std_detections': float(np.std(detections_array)),
            'max_detections': int(np.max(detections_array)),
            'min_detections': int(np.min(detections_array)),
            'avg_confidence': float(np.mean(confidences_array)),
            'std_confidence': float(np.std(confidences_array))
        }


if __name__ == "__main__":
    logger.info("Testing RealTimeDetector...")
    
    # This would require a trained model to be available
    # detector = RealTimeDetector("models/trained/pill_detector.pt")
    # detector.run_realtime(camera_id=0)
    
    logger.info("Tests completed!")