"""
GUI Module for Pill Counter Application
Provides a user-friendly interface for real-time pill counting.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import logging
from pathlib import Path
from typing import Optional
import time

from src.inference import RealTimeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PillCounterGUI:
    """GUI Application for Pill Counter."""
    
    def __init__(self, root: tk.Tk, model_path: str):
        """
        Initialize GUI.
        
        Args:
            root: Tkinter root window
            model_path: Path to trained YOLO model
        """
        self.root = root
        self.root.title("Intelligent Pill Counter")
        self.root.geometry("1200x800")
        
        # Initialize detector
        try:
            self.detector = RealTimeDetector(model_path)
            logger.info(f"Model loaded: {model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.detector = None
        
        # State variables
        self.cap = None
        self.is_running = False
        self.pill_count = 0
        self.frame_count = 0
        self.fps = 0
        self.processing_time = 0
        
        # Configure style
        self.setup_styles()
        
        # Create GUI elements
        self.create_widgets()
        
        logger.info("GUI initialized successfully")
    
    def setup_styles(self) -> None:
        """Configure GUI styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#00ff00"
        
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        style.configure("TButton", background=self.bg_color, foreground=self.fg_color)
        style.configure("TFrame", background=self.bg_color)
        
        self.root.configure(bg=self.bg_color)
    
    def create_widgets(self) -> None:
        """Create GUI widgets."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Video display
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video label
        ttk.Label(left_panel, text="Live Feed", font=("Arial", 14, "bold")).pack()
        
        self.video_label = ttk.Label(left_panel, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Right panel - Controls and stats
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=20)
        
        # Title
        ttk.Label(right_panel, text="Control Panel", 
                 font=("Arial", 16, "bold")).pack(pady=10)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(right_panel, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Count display
        count_frame = ttk.Frame(stats_frame)
        count_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(count_frame, text="Pill Count:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.count_label = ttk.Label(count_frame, text="0", 
                                    font=("Arial", 14, "bold"), 
                                    foreground=self.accent_color)
        self.count_label.pack(side=tk.LEFT, padx=10)
        
        # FPS display
        fps_frame = ttk.Frame(stats_frame)
        fps_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(fps_frame, text="FPS:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.fps_label = ttk.Label(fps_frame, text="0.0", 
                                  font=("Arial", 12), 
                                  foreground=self.accent_color)
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        # Latency display
        latency_frame = ttk.Frame(stats_frame)
        latency_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(latency_frame, text="Latency (ms):", font=("Arial", 12)).pack(side=tk.LEFT)
        self.latency_label = ttk.Label(latency_frame, text="0.0", 
                                      font=("Arial", 12), 
                                      foreground=self.accent_color)
        self.latency_label.pack(side=tk.LEFT, padx=10)
        
        # Control buttons frame
        control_frame = ttk.LabelFrame(right_panel, text="Controls")
        control_frame.pack(fill=tk.X, pady=10)
        
        # Start/Stop button
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="Start Camera",
                                   command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Camera",
                                  command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        self.reset_btn = ttk.Button(button_frame, text="Reset Count",
                                   command=self.reset_count)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(right_panel, text="Settings")
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Confidence threshold
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(conf_frame, text="Confidence:", font=("Arial", 11)).pack(side=tk.LEFT)
        
        self.conf_label = ttk.Label(conf_frame, text="0.50", font=("Arial", 11))
        
        self.conf_scale = ttk.Scale(conf_frame, from_=0.1, to=0.9,
                                   orient=tk.HORIZONTAL, command=self.update_confidence)
        self.conf_scale.set(0.5)
        
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.conf_label.pack(side=tk.LEFT)
        
        # NMS threshold
        nms_frame = ttk.Frame(settings_frame)
        nms_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(nms_frame, text="NMS:", font=("Arial", 11)).pack(side=tk.LEFT)
        
        self.nms_label = ttk.Label(nms_frame, text="0.45", font=("Arial", 11))
        
        self.nms_scale = ttk.Scale(nms_frame, from_=0.1, to=0.9,
                                  orient=tk.HORIZONTAL, command=self.update_nms)
        self.nms_scale.set(0.45)
        
        self.nms_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.nms_label.pack(side=tk.LEFT)
        
        # File operations frame
        file_frame = ttk.LabelFrame(right_panel, text="File Operations")
        file_frame.pack(fill=tk.X, pady=10)
        
        file_btn_frame = ttk.Frame(file_frame)
        file_btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(file_btn_frame, text="Open Image",
                  command=self.open_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(file_btn_frame, text="Open Video",
                  command=self.open_video).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
    
    def start_camera(self) -> None:
        """Start camera feed."""
        if self.is_running:
            messagebox.showwarning("Warning", "Camera is already running")
            return
        
        if self.detector is None:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            return
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Status: Camera running")
        
        # Start processing in separate thread
        thread = threading.Thread(target=self.process_camera, daemon=True)
        thread.start()
    
    def stop_camera(self) -> None:
        """Stop camera feed."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Status: Camera stopped")
    
    def process_camera(self) -> None:
        """Process camera frames."""
        fps_counter = 0
        fps_time = time.time()
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Inference
                start_time = time.time()
                detections = self.detector.detect_pills(frame)
                self.processing_time = (time.time() - start_time) * 1000
                
                # Draw detections
                frame = self.detector.draw_boxes(frame, detections)
                
                # Count pills
                self.pill_count = len(detections.boxes)
                
                # Update FPS
                fps_counter += 1
                elapsed = time.time() - fps_time
                if elapsed >= 1.0:
                    self.fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_time = time.time()
                
                # Display frame
                self.display_frame(frame)
                
                # Update labels
                self.update_labels()
        
        except Exception as e:
            logger.error(f"Camera processing error: {e}")
            self.is_running = False
    
    def display_frame(self, frame: object) -> None:
        """Display frame in GUI."""
        # Resize for display
        frame = cv2.resize(frame, (600, 450))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.video_label.imgtk = tk_image
        self.video_label.config(image=tk_image)
    
    def update_labels(self) -> None:
        """Update statistics labels."""
        self.count_label.config(text=str(self.pill_count))
        self.fps_label.config(text=f"{self.fps:.1f}")
        self.latency_label.config(text=f"{self.processing_time:.1f}")
        self.root.update_idletasks()
    
    def reset_count(self) -> None:
        """Reset pill count."""
        self.pill_count = 0
        self.count_label.config(text="0")
        self.status_var.set("Status: Count reset")
    
    def update_confidence(self, value: str) -> None:
        """Update confidence threshold."""
        conf = float(value)
        self.detector.conf_threshold = conf
        self.conf_label.config(text=f"{conf:.2f}")
    
    def update_nms(self, value: str) -> None:
        """Update NMS threshold."""
        nms = float(value)
        self.detector.nms_threshold = nms
        self.nms_label.config(text=f"{nms:.2f}")
    
    def open_image(self) -> None:
        """Open and process an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                frame, detections = self.detector.detect_from_image(file_path)
                if frame is not None:
                    self.display_frame(frame)
                    self.pill_count = len(detections.boxes)
                    self.update_labels()
                    self.status_var.set(f"Status: Loaded {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {e}")
    
    def open_video(self) -> None:
        """Open and process a video file."""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set(f"Status: Processing video {Path(file_path).name}...")
                output_path = Path(file_path).stem + "_detected.mp4"
                self.detector.detect_from_video(file_path, str(output_path))
                messagebox.showinfo("Success", f"Video processed. Saved to {output_path}")
                self.status_var.set("Status: Video processing completed")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process video: {e}")


def main(model_path: str = "models/trained/best.pt") -> None:
    """
    Launch GUI application.
    
    Args:
        model_path: Path to trained model
    """
    root = tk.Tk()
    app = PillCounterGUI(root, model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
