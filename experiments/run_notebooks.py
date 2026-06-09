import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

notebooks = [
    "01_environment_setup.ipynb",
    "02_data_loading_eda.ipynb",
    "03_preprocessing_augmentation.ipynb",
    "04_model_training.ipynb",
    "05_evaluation_metrics.ipynb",
    "06_realtime_inference.ipynb"
]

# Create ExecutePreprocessor using our custom kernel
ep = ExecutePreprocessor(timeout=600, kernel_name='pill_detector_kernel')

# Make sure working directory for execution is the project root so paths match
project_root = Path(__file__).resolve().parent.parent

for nb_name in notebooks:
    nb_path = project_root / "experiments" / nb_name
    print(f"Executing {nb_name}...")
    try:
        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        # Execute the notebook using the project root directory context
        ep.preprocess(nb, {'metadata': {'path': str(project_root)}})
        
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Successfully executed and saved {nb_name}")
    except Exception as e:
        print(f"Error executing {nb_name}: {e}")
