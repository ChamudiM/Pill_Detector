"""
Intelligent Pill Counter Package
"""

__version__ = "1.0.0"
__author__ = "Team"

from src.preprocessing import ImagePreprocessor, DataAugmenter
from src.dataset_loader import PillDataLoader, YOLOAnnotationValidator
from src.model_trainer import PillDetectionTrainer, TrainingValidator
from src.inference import RealTimeDetector, DetectionStatistics
from src.gui import PillCounterGUI

__all__ = [
    'ImagePreprocessor',
    'DataAugmenter',
    'PillDataLoader',
    'YOLOAnnotationValidator',
    'PillDetectionTrainer',
    'TrainingValidator',
    'RealTimeDetector',
    'DetectionStatistics',
    'PillCounterGUI'
]
