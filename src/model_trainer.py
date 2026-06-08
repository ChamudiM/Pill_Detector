"""
Model Training Module
Handles YOLO model training, evaluation, and checkpoint management.
"""

import torch
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional
from ultralytics import YOLO
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PillDetectionTrainer:
    """Trains YOLO model for pill detection."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to config.yaml
        """
        self.config_path = Path(config_path)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.training_results = None
        
        logger.info("PillDetectionTrainer initialized")
        logger.info(f"Device: {self._get_device()}")
    
    def _get_device(self) -> str:
        """Get device information."""
        device_id = self.config['model']['device']
        if device_id < 0:
            return "cpu"
        elif torch.cuda.is_available():
            return f"cuda:{device_id}"
        else:
            logger.warning("CUDA not available, using CPU")
            return "cpu"
    
    def prepare_dataset_yaml(self, dataset_path: str) -> Path:
        """
        Create dataset.yaml for YOLO training.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Path to created dataset.yaml
        """
        dataset_path = Path(dataset_path)
        
        dataset_config = {
            'path': str(dataset_path.absolute()),
            'train': str((dataset_path / 'train' / 'images').relative_to(dataset_path)),
            'val': str((dataset_path / 'val' / 'images').relative_to(dataset_path)),
            'test': str((dataset_path / 'test' / 'images').relative_to(dataset_path)),
            'nc': 1,  # Number of classes
            'names': ['pill']
        }
        
        yaml_file = dataset_path / 'dataset.yaml'
        
        with open(yaml_file, 'w') as f:
            yaml.dump(dataset_config, f)
        
        logger.info(f"Created dataset.yaml at {yaml_file}")
        return yaml_file
    
    def load_model(self, model_variant: Optional[str] = None) -> YOLO:
        """
        Load YOLO model.
        
        Args:
            model_variant: Model size (n, s, m, l, x). If None, uses config.
            
        Returns:
            YOLO model
        """
        if model_variant is None:
            model_variant = self.config['model']['name']
        
        logger.info(f"Loading {model_variant} model...")
        self.model = YOLO(f"{model_variant}.pt")
        
        logger.info(f"Model loaded: {model_variant}")
        return self.model
    
    def train(self, dataset_yaml: str, output_dir: str = "models/trained",
             resume: bool = False, pretrained: bool = True) -> Tuple:
        """
        Train YOLO model on pill dataset.
        
        Args:
            dataset_yaml: Path to dataset.yaml
            output_dir: Output directory for trained models
            resume: Resume from checkpoint
            pretrained: Use pretrained weights
            
        Returns:
            Tuple of (results, trained_model)
        """
        # Load model if not already loaded
        if self.model is None:
            model_name = self.config['model']['name']
            if pretrained:
                self.model = YOLO(f"{model_name}.pt")
            else:
                self.model = YOLO(f"{model_name}.yaml")
        
        logger.info("Starting training...")
        
        # Training parameters
        train_params = {
            'data': str(dataset_yaml),
            'epochs': self.config['training']['epochs'],
            'imgsz': self.config['dataset']['img_size'],
            'batch': self.config['training']['batch_size'],
            'device': self._get_device(),
            'lr0': self.config['training']['learning_rate'],
            'patience': self.config['training']['patience'],
            'momentum': self.config['training'].get('momentum', 0.937),
            'weight_decay': self.config['training'].get('weight_decay', 0.0005),
            'warmup_epochs': self.config['training'].get('warmup_epochs', 3),
            'optimizer': self.config['training'].get('optimizer', 'SGD'),
            'project': output_dir,
            'name': 'pill_detector',
            'save': True,
            'val': True,
            'verbose': True,
            'seed': 42
        }
        
        # Add augmentation parameters
        augmentation = self.config.get('augmentation', {})
        if augmentation:
            train_params.update({
                'hsv_h': augmentation.get('hsv_h', 0.015),
                'hsv_s': augmentation.get('hsv_s', 0.7),
                'hsv_v': augmentation.get('hsv_v', 0.4),
                'degrees': augmentation.get('degrees', 15.0),
                'translate': augmentation.get('translate', 0.1),
                'scale': augmentation.get('scale', 0.2),
                'flipud': augmentation.get('flipud', 0.0),
                'fliplr': augmentation.get('fliplr', 0.5),
                'mosaic': augmentation.get('mosaic', 1.0),
            })
        
        # Resume from checkpoint if specified
        if resume:
            logger.info("Resuming from checkpoint...")
            train_params['resume'] = True
        
        try:
            self.training_results = self.model.train(**train_params)
            logger.info("Training completed successfully!")
            return self.training_results, self.model
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate(self, dataset_yaml: str) -> Dict:
        """
        Validate trained model on validation set.
        
        Args:
            dataset_yaml: Path to dataset.yaml
            
        Returns:
            Dictionary with validation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting validation...")
        metrics = self.model.val(data=str(dataset_yaml), device=self._get_device())
        
        logger.info("Validation completed!")
        return metrics
    
    def test(self, dataset_yaml: str) -> Dict:
        """
        Test model on test set.
        
        Args:
            dataset_yaml: Path to dataset.yaml
            
        Returns:
            Dictionary with test metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting testing...")
        metrics = self.model.val(data=str(dataset_yaml), split='test', 
                                device=self._get_device())
        
        logger.info("Testing completed!")
        return metrics
    
    def save_model(self, save_path: str) -> Path:
        """
        Save trained model.
        
        Args:
            save_path: Path to save model
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load model first.")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_file = save_path / "pill_detector.pt"
        self.model.save(str(model_file))
        
        logger.info(f"Model saved to {model_file}")
        return model_file
    
    def load_checkpoint(self, checkpoint_path: str) -> YOLO:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Loaded YOLO model
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.model = YOLO(checkpoint_path)
        logger.info("Checkpoint loaded successfully")
        return self.model
    
    def export_model(self, format: str = "onnx", save_path: str = "models/exported") -> Path:
        """
        Export model to different format.
        
        Args:
            format: Export format (onnx, tflite, engine, etc.)
            save_path: Directory to save exported model
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting model to {format}...")
        exported_path = self.model.export(format=format, imgsz=self.config['dataset']['img_size'])
        
        logger.info(f"Model exported successfully to {exported_path}")
        return Path(exported_path)
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        info = {
            'model_type': str(self.model),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'device': self._get_device(),
            'input_size': self.config['dataset']['img_size'],
        }
        
        return info
    
    def print_model_summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        logger.info("\n" + "="*50)
        logger.info("MODEL SUMMARY")
        logger.info("="*50)
        logger.info(str(self.model))
        logger.info("="*50 + "\n")


class TrainingValidator:
    """Validates training configuration and inputs."""
    
    @staticmethod
    def validate_config(config: Dict) -> Tuple[bool, str]:
        """
        Validate training configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required keys
        required_keys = ['dataset', 'model', 'training']
        for key in required_keys:
            if key not in config:
                return False, f"Missing required key: {key}"
        
        # Validate training parameters
        training = config['training']
        if training['epochs'] < 1:
            return False, "Epochs must be >= 1"
        
        if training['batch_size'] < 1:
            return False, "Batch size must be >= 1"
        
        if not (0 < training['learning_rate'] < 1):
            return False, "Learning rate must be between 0 and 1"
        
        return True, ""
    
    @staticmethod
    def validate_dataset(dataset_yaml: Path) -> Tuple[bool, str]:
        """
        Validate dataset.yaml.
        
        Args:
            dataset_yaml: Path to dataset.yaml
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not dataset_yaml.exists():
            return False, f"Dataset yaml not found: {dataset_yaml}"
        
        try:
            with open(dataset_yaml) as f:
                config = yaml.safe_load(f)
            
            required_keys = ['path', 'train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in config:
                    return False, f"Missing key in dataset.yaml: {key}"
            
            return True, ""
        except Exception as e:
            return False, f"Error reading dataset.yaml: {e}"


if __name__ == "__main__":
    logger.info("Testing PillDetectionTrainer...")
    
    # Note: This requires config.yaml and dataset to be properly set up
    # trainer = PillDetectionTrainer("config.yaml")
    # trainer.load_model()
    # trainer.print_model_summary()
    
    logger.info("Tests completed!")