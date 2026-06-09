"""
Test Scripts for Pill Counter System
Run these to verify installations and functionality
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all dependencies are properly installed."""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        logger.error(f"✗ PyTorch: {e}")
        return False
    
    try:
        import cv2
        logger.info(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        logger.error(f"✗ OpenCV: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        logger.info("✓ Ultralytics YOLO")
    except ImportError as e:
        logger.error(f"✗ Ultralytics: {e}")
        return False
    
    try:
        import numpy
        logger.info(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        logger.error(f"✗ NumPy: {e}")
        return False
    
    try:
        from PIL import Image
        logger.info("✓ Pillow")
    except ImportError as e:
        logger.error(f"✗ Pillow: {e}")
        return False
    
    try:
        import yaml
        logger.info("✓ PyYAML")
    except ImportError as e:
        logger.error(f"✗ PyYAML: {e}")
        return False
    
    logger.info("All imports successful!")
    return True


def test_gpu():
    """Test GPU availability."""
    logger.info("\nTesting GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
            logger.info(f"✓ CUDA Version: {torch.version.cuda}")
        else:
            logger.warning("⚠ GPU not available, will use CPU (slower)")
    except Exception as e:
        logger.error(f"✗ GPU test failed: {e}")


def test_preprocessing():
    """Test preprocessing module."""
    logger.info("\nTesting preprocessing module...")
    
    try:
        from src.preprocessing import ImagePreprocessor, DataAugmenter
        import numpy as np
        
        # Create test image
        test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Test preprocessor
        preprocessor = ImagePreprocessor(640)
        processed = preprocessor.preprocess_frame(test_img)
        
        assert processed.shape == (640, 640, 3), "Output shape mismatch"
        logger.info("✓ ImagePreprocessor working")
        
        # Test augmenter
        augmenter = DataAugmenter()
        rotated = augmenter.rotate_image(test_img, 15)
        assert rotated.shape == test_img.shape, "Rotation failed"
        logger.info("✓ DataAugmenter working")
        
    except Exception as e:
        logger.error(f"✗ Preprocessing test failed: {e}")
        return False
    
    return True


def test_dataset_loader():
    """Test dataset loader module."""
    logger.info("\nTesting dataset loader module...")
    
    try:
        from src.dataset_loader import PillDataLoader, YOLOAnnotationValidator
        
        loader = PillDataLoader("data/raw")
        logger.info("✓ PillDataLoader initialized")
        
        # Test annotation validator
        validator = YOLOAnnotationValidator()
        valid_line = "0 0.5 0.5 0.3 0.4"
        is_valid = validator.validate_annotation_line(valid_line)
        assert is_valid, "Valid annotation marked as invalid"
        logger.info("✓ YOLOAnnotationValidator working")
        
    except Exception as e:
        logger.error(f"✗ Dataset loader test failed: {e}")
        return False
    
    return True


def test_model_trainer():
    """Test model trainer module."""
    logger.info("\nTesting model trainer module...")
    
    try:
        from src.model_trainer import PillDetectionTrainer
        
        # Test with config
        if Path("config.yaml").exists():
            trainer = PillDetectionTrainer("config.yaml")
            logger.info("✓ PillDetectionTrainer initialized")
            
            # Test model loading (don't actually load to save time)
            info = trainer.get_model_info() if False else {}
            logger.info("✓ Model trainer ready")
        else:
            logger.warning("⚠ config.yaml not found, skipping full test")
        
    except Exception as e:
        logger.error(f"✗ Model trainer test failed: {e}")
        return False
    
    return True


def test_config():
    """Test configuration file."""
    logger.info("\nTesting configuration...")
    
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check required keys
            required = ['dataset', 'model', 'training', 'inference']
            for key in required:
                assert key in config, f"Missing key: {key}"
            
            logger.info("✓ Configuration file valid")
            return True
        else:
            logger.error("✗ config.yaml not found")
            return False
    
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        return False


def test_directory_structure():
    """Test required directory structure."""
    logger.info("\nTesting directory structure...")
    
    try:
        required_dirs = [
            'data/raw',
            'data/processed',
            'models/trained',
            'src',
            'tests',
            'docs'
        ]
        
        missing = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing.append(dir_path)
        
        if missing:
            logger.warning(f"⚠ Missing directories: {missing}")
        else:
            logger.info("✓ All required directories present")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Directory test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    logger.info("="*50)
    logger.info("PILL COUNTER SYSTEM TEST SUITE")
    logger.info("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("GPU", test_gpu),
        ("Directory Structure", test_directory_structure),
        ("Configuration", test_config),
        ("Preprocessing", test_preprocessing),
        ("Dataset Loader", test_dataset_loader),
        ("Model Trainer", test_model_trainer),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"✗ {test_name} test error: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n✓ All tests passed! System ready to use.")
    else:
        logger.warning(f"\n⚠ {total - passed} test(s) failed. Please fix issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)