import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset loading and management for pill detection
class PillDataLoader:
    """Loads and manages pill detection datasets."""
    
    def __init__(self, data_dir: str, img_size: int = 640):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to data directory
            img_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.dataset_stats = {
            'total_images': 0,
            'total_annotations': 0,
            'image_formats': {},
            'missing_annotations': 0
        }
        logger.info(f"PillDataLoader initialized with directory: {data_dir}")
    
    def discover_images(self, formats: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')) -> List[Path]:
        """
        Discover all images in the data directory.
        
        Args:
            formats: Tuple of image format extensions
            
        Returns:
            List of image paths
        """
        images = []
        for fmt in formats:
            images.extend(self.data_dir.rglob(f'*{fmt}'))
        
        logger.info(f"Found {len(images)} images")
        return sorted(images)
    
    def load_annotation(self, ann_path: Path) -> Optional[List[str]]:
        """
        Load YOLO format annotation.
        
        YOLO format:
        <class_id> <x_center> <y_center> <width> <height>
        (All coordinates normalized 0-1)
        
        Args:
            ann_path: Path to annotation file
            
        Returns:
            List of annotation lines or None if not found
        """
        if not ann_path.exists():
            return None
        
        try:
            with open(ann_path, 'r') as f:
                lines = f.readlines()
            return lines
        except Exception as e:
            logger.warning(f"Error reading annotation {ann_path}: {e}")
            return None
    
    def load_dataset(self) -> Tuple[List[Path], List[Optional[List[str]]]]:
        """
        Load all images and their annotations.
        
        Returns:
            Tuple of (image_paths, annotations)
        """
        images = self.discover_images()
        annotations = []
        missing_count = 0
        
        for img_path in images:
            ann_path = img_path.with_suffix('.txt')
            ann = self.load_annotation(ann_path)
            annotations.append(ann)
            
            if ann is None:
                missing_count += 1
        
        self.dataset_stats['total_images'] = len(images)
        self.dataset_stats['total_annotations'] = len([a for a in annotations if a is not None])
        self.dataset_stats['missing_annotations'] = missing_count
        
        logger.info(f"Loaded {len(images)} images with {self.dataset_stats['total_annotations']} annotations")
        if missing_count > 0:
            logger.warning(f"Missing annotations: {missing_count}")
        
        return images, annotations
    
    def filter_valid_samples(self, images: List[Path], 
                            annotations: List[Optional[List[str]]]) -> Tuple[List[Path], List[List[str]]]:
        """
        Filter out images without annotations.
        
        Args:
            images: List of image paths
            annotations: List of annotations
            
        Returns:
            Tuple of (valid_images, valid_annotations)
        """
        valid_images = []
        valid_annotations = []
        
        for img, ann in zip(images, annotations):
            if ann is not None:
                valid_images.append(img)
                valid_annotations.append(ann)
        
        logger.info(f"Filtered to {len(valid_images)} valid samples")
        return valid_images, valid_annotations
    
    def split_dataset(self, images: List[Path], annotations: List[List[str]],
                     train_size: float = 0.7, val_size: float = 0.15,
                     random_state: int = 42) -> Dict[str, Tuple[List[Path], List[List[str]]]]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            images: List of image paths
            annotations: List of annotations
            train_size: Proportion for training
            val_size: Proportion for validation
            random_state: Random seed
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        # First split: train and temp (val + test)
        train_imgs, temp_imgs, train_anns, temp_anns = train_test_split(
            images, annotations,
            test_size=(1 - train_size),
            random_state=random_state
        )
        
        # Second split: val and test
        val_size_ratio = val_size / (1 - train_size)
        val_imgs, test_imgs, val_anns, test_anns = train_test_split(
            temp_imgs, temp_anns,
            test_size=(1 - val_size_ratio),
            random_state=random_state
        )
        
        splits = {
            'train': (train_imgs, train_anns),
            'val': (val_imgs, val_anns),
            'test': (test_imgs, test_anns)
        }
        
        logger.info(f"Dataset split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
        
        return splits
    
    def organize_dataset_structure(self, splits: Dict[str, Tuple[List[Path], List[List[str]]]],
                                  output_dir: Path) -> None:
        """
        Organize dataset into train/val/test directories with images and labels subdirs.
        
        Args:
            splits: Dictionary with dataset splits
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        
        for split_name, (images, annotations) in splits.items():
            # Create directories
            img_dir = output_dir / split_name / 'images'
            label_dir = output_dir / split_name / 'labels'
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images and annotations
            for img_path, ann_lines in zip(images, annotations):
                # Copy image
                img_name = img_path.name
                new_img_path = img_dir / img_name
                
                # Copy annotation
                label_name = img_path.with_suffix('.txt').name
                new_label_path = label_dir / label_name
                
                try:
                    # Read and write image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        cv2.imwrite(str(new_img_path), img)
                    
                    # Write annotation
                    with open(new_label_path, 'w') as f:
                        f.writelines(ann_lines)
                        
                except Exception as e:
                    logger.error(f"Error copying {img_path}: {e}")
            
            logger.info(f"Organized {split_name} split: {len(images)} samples")
    
    def create_dataset_yaml(self, output_dir: Path, num_classes: int = 1,
                           class_names: List[str] = None) -> Path:
        """
        Create YOLO dataset.yaml configuration file.
        
        Args:
            output_dir: Directory to save yaml
            num_classes: Number of classes
            class_names: List of class names
            
        Returns:
            Path to created yaml file
        """
        if class_names is None:
            class_names = ['pill']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_yaml = {
            'path': str(output_dir.parent.absolute()),
            'train': str((output_dir / 'train' / 'images').relative_to(output_dir.parent)),
            'val': str((output_dir / 'val' / 'images').relative_to(output_dir.parent)),
            'test': str((output_dir / 'test' / 'images').relative_to(output_dir.parent)),
            'nc': num_classes,
            'names': class_names
        }
        
        yaml_path = output_dir / 'data.yaml'
        
        # Save using simple yaml format
        with open(yaml_path, 'w') as f:
            for key, value in dataset_yaml.items():
                if isinstance(value, str):
                    f.write(f"{key}: {value}\n")
                elif isinstance(value, list):
                    f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"Created dataset.yaml at {yaml_path}")
        return yaml_path
    
    def get_dataset_stats(self, images: List[Path]) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            images: List of image paths
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_images': len(images),
            'avg_size': None,
            'min_size': None,
            'max_size': None,
            'formats': {}
        }
        
        sizes = []
        for img_path in images:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    sizes.append(img.shape)
                    fmt = img_path.suffix.lower()
                    stats['formats'][fmt] = stats['formats'].get(fmt, 0) + 1
            except Exception as e:
                logger.warning(f"Error reading {img_path}: {e}")
        
        if sizes:
            stats['avg_size'] = tuple(np.mean([s for s in sizes], axis=0).astype(int))
            stats['min_size'] = tuple(np.min([s for s in sizes], axis=0).astype(int))
            stats['max_size'] = tuple(np.max([s for s in sizes], axis=0).astype(int))
        
        logger.info(f"Dataset stats: {stats}")
        return stats


class YOLOAnnotationValidator:
    """Validates YOLO format annotations."""
    
    @staticmethod
    def validate_annotation_line(line: str) -> bool:
        """
        Validate a single YOLO annotation line.
        
        Format: <class_id> <x_center> <y_center> <width> <height>
        
        Args:
            line: Annotation line
            
        Returns:
            True if valid, False otherwise
        """
        try:
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            
            # Check ranges
            if class_id < 0:
                return False
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                return False
            if not (0 < width <= 1 and 0 < height <= 1):
                return False
            
            return True
        except:
            return False
    
    @staticmethod
    def validate_annotation_file(ann_path: Path) -> Tuple[bool, List[int]]:
        """
        Validate all lines in an annotation file.
        
        Args:
            ann_path: Path to annotation file
            
        Returns:
            Tuple of (is_valid, invalid_line_numbers)
        """
        invalid_lines = []
        
        if not ann_path.exists():
            return False, [-1]
        
        try:
            with open(ann_path, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip() and not YOLOAnnotationValidator.validate_annotation_line(line):
                        invalid_lines.append(i)
            
            return len(invalid_lines) == 0, invalid_lines
        except Exception as e:
            logger.error(f"Error validating {ann_path}: {e}")
            return False, [-1]


if __name__ == "__main__":
    # Test dataset loading
    logger.info("Testing PillDataLoader...")
    
    loader = PillDataLoader("data/raw")
    images, annotations = loader.load_dataset()
    
    logger.info(f"Discovered {len(images)} images")
    
    # Filter valid samples
    valid_images, valid_annotations = loader.filter_valid_samples(images, annotations)
    logger.info(f"Found {len(valid_images)} valid samples")
    
    # Split dataset
    if valid_images:
        splits = loader.split_dataset(valid_images, valid_annotations)
        logger.info(f"Dataset split complete")
    
    logger.info("Tests completed successfully!")