"""
Main entry point for Pill Counter application
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import ImagePreprocessor, DataAugmenter
from src.dataset_loader import PillDataLoader, YOLOAnnotationValidator
from src.model_trainer import PillDetectionTrainer, TrainingValidator
from src.inference import RealTimeDetector, DetectionStatistics
from src.gui import PillCounterGUI
import tkinter as tk

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Create necessary directories if they don't exist."""
    dirs = [
        'data/raw',
        'data/processed',
        'models/trained',
        'results/visualizations',
        'logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment setup completed")


def prepare_dataset(args):
    """Prepare dataset for training."""
    logger.info("Starting dataset preparation...")
    
    loader = PillDataLoader(args.data_dir)
    images, annotations = loader.load_dataset()
    
    # Filter valid samples
    valid_images, valid_annotations = loader.filter_valid_samples(images, annotations)
    
    # Get dataset statistics
    stats = loader.get_dataset_stats(valid_images)
    logger.info(f"Dataset statistics: {stats}")
    
    # Split dataset
    splits = loader.split_dataset(valid_images, valid_annotations)
    
    # Organize into structure
    output_dir = Path(args.output_dir)
    loader.organize_dataset_structure(splits, output_dir)
    
    # Create dataset.yaml
    yaml_path = loader.create_dataset_yaml(output_dir)
    
    logger.info(f"Dataset preparation completed. YAML: {yaml_path}")
    return yaml_path


def train_model(args):
    """Train YOLO model."""
    logger.info("Starting model training...")
    
    trainer = PillDetectionTrainer(args.config)
    
    # Validate dataset
    yaml_path = Path(args.dataset_yaml)
    is_valid, error = TrainingValidator.validate_dataset(yaml_path)
    
    if not is_valid:
        logger.error(f"Dataset validation failed: {error}")
        return
    
    # Load model
    trainer.load_model()
    trainer.print_model_summary()
    
    # Train
    try:
        results, model = trainer.train(
            str(yaml_path),
            output_dir=args.output_dir,
            resume=args.resume
        )
        
        # Save model
        save_path = trainer.save_model(args.output_dir)
        logger.info(f"Model training completed. Saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    # Validate
    logger.info("Validating model...")
    metrics = trainer.validate(str(yaml_path))
    logger.info(f"Validation metrics: {metrics}")


def run_inference(args):
    """Run real-time inference."""
    logger.info("Starting real-time inference...")
    
    detector = RealTimeDetector(
        args.model,
        conf_threshold=args.confidence,
        nms_threshold=args.nms
    )
    
    if args.camera:
        logger.info("Running camera inference...")
        detector.run_realtime(camera_id=args.camera_id)
    
    elif args.image:
        logger.info(f"Processing image: {args.image}")
        frame, detections = detector.detect_from_image(args.image)
        if frame is not None:
            import cv2
            cv2.imwrite("output_image.jpg", frame)
            logger.info(f"Saved output to output_image.jpg. Detected {len(detections.boxes)} pills")
    
    elif args.video:
        logger.info(f"Processing video: {args.video}")
        detector.detect_from_video(args.video, args.output_video)
        logger.info(f"Saved output to {args.output_video}")


def run_gui(args):
    """Launch GUI application."""
    logger.info("Launching GUI...")
    
    root = tk.Tk()
    app = PillCounterGUI(root, args.model)
    root.mainloop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Pill Counter - Deep Learning & Computer Vision"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Setup command
    subparsers.add_parser('setup', help='Setup environment')
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset')
    prepare_parser.add_argument('--data-dir', default='data/raw', help='Raw data directory')
    prepare_parser.add_argument('--output-dir', default='data/processed', help='Output directory')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', default='config.yaml', help='Config file')
    train_parser.add_argument('--dataset-yaml', default='data/processed/dataset.yaml', 
                            help='Dataset YAML file')
    train_parser.add_argument('--output-dir', default='models/trained', help='Output directory')
    train_parser.add_argument('--resume', action='store_true', help='Resume training')
    
    # Inference command
    inference_parser = subparsers.add_parser('infer', help='Run inference')
    inference_parser.add_argument('--model', default='models/trained/best.pt', help='Model path')
    inference_parser.add_argument('--camera', action='store_true', help='Use camera')
    inference_parser.add_argument('--camera-id', type=int, default=0, help='Camera ID')
    inference_parser.add_argument('--image', help='Image file path')
    inference_parser.add_argument('--video', help='Video file path')
    inference_parser.add_argument('--output-video', default='output_video.mp4', 
                                help='Output video path')
    inference_parser.add_argument('--confidence', type=float, default=0.5, 
                                help='Confidence threshold')
    inference_parser.add_argument('--nms', type=float, default=0.45, help='NMS threshold')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch GUI')
    gui_parser.add_argument('--model', default='models/trained/best.pt', help='Model path')
    
    args = parser.parse_args()
    
    # Setup environment first
    setup_environment()
    
    # Execute command
    if args.command == 'setup':
        logger.info("Environment already set up")
    
    elif args.command == 'prepare':
        prepare_dataset(args)
    
    elif args.command == 'train':
        train_model(args)
    
    elif args.command == 'infer':
        run_inference(args)
    
    elif args.command == 'gui':
        run_gui(args)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()