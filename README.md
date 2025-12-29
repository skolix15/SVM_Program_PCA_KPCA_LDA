# Image Classification: Comparative Evaluation of Machine Learning Models

Comparative evaluation of KNN, NCC, Linear SVM, and RBF SVM classifiers on CIFAR-10, SVHN, and Faces (emotion) datasets.

## Datasets

- **CIFAR-10**: 50,000 training / 10,000 test images (10 classes)
- **SVHN**: 73,257 training / 26,032 test images (digits 0-9)
- **Faces**: Emotion classification dataset loaded from local PNG images (60% training / 40% test split)

## Models

- **KNN**: k=10 neighbors
- **NCC**: Nearest Class Centroid
- **Linear SVM**: C ∈ {0.1, 1, 5, 10}
- **RBF SVM**: C ∈ {0.1, 1, 3}, γ ∈ {"scale", "auto"}

## Preprocessing

### Standard Pipeline (main.py)
1. Flatten images
2. L2 normalization
3. Z-score standardization
4. PCA (90% variance retention)
5. Label preprocessing

### Extended Pipeline with KPCA + LDA (main2.py)
1. Flatten images
2. L2 normalization
3. Z-score standardization
4. Kernel PCA (KPCA) with configurable kernels (rbf, poly) and n_components
5. Linear Discriminant Analysis (LDA)
6. Label preprocessing

## Installation

```bash
pipenv install
pipenv shell
```

## Usage

### Standard Pipeline (CIFAR-10 / SVHN)

```bash
python main.py
```

### Extended Pipeline with KPCA + LDA (Faces Dataset)

```bash
python main2.py
```

## Configuration

### main.py Configuration

Edit `program_basic_info` in `main.py`:

```python
program_basic_info: dict = {
    "desired_data": "cifar10",  # "cifar10" or "svhn"
    "linear_parameters": {"C": [0.1, 1, 5, 10]},
    "rbf_parameters": {
        "C": [0.1, 1, 3],
        "gamma": ["scale", "auto"]
    },
    "minimize_samples": False
}
```

### main2.py Configuration

Edit `program_basic_info` in `main2.py`:

```python
program_basic_info: dict = {
    "desired_data": "faces",  # "faces" dataset
    "linear_parameters": {"C": [0.1, 1, 5, 10]},
    "rbf_parameters": {
        "C": [0.1, 1, 3],
        "gamma": ["scale", "auto"]
    },
    "minimize_samples": False,
    "kpca_parameters": {
        "kernels": ["rbf", "poly"],  # KPCA kernel types
        "n_components": 100  # Number of KPCA components
    }
}
```

**Note**: The faces dataset should be organized in `faces_data/` directory with subfolders for each emotion class (e.g., `Angry/`, `Happy/`, `Sad/`, etc.). Each subfolder should contain PNG images.

## Project Structure

```
Program/
├── main.py                    # Main script (CIFAR-10/SVHN with PCA)
├── main2.py                   # Extended script (Faces with KPCA + LDA)
├── main_helper.py            # Main execution and aggregation logic
├── data_helper.py            # Preprocessing (PCA, KPCA, LDA)
├── dataset_helper.py         # Dataset loading (CIFAR-10, SVHN, Faces)
├── knn_ncc_evaluator.py     # KNN/NCC models
├── svm_evaluator.py         # SVM models
├── visualization_helper.py  # Visualizations
├── svhn_data/              # SVHN dataset files
├── faces_data/             # Faces dataset (emotion folders with PNG images)
└── visualizations/         # Output visualization files
```

## Output

- **Console**: Performance metrics and results tables
- **Visualizations**: 
  - Prediction examples (correct and incorrect classifications)
  - Performance metrics comparison charts
  - Time metrics comparison charts
  - For `main2.py`: Separate visualizations for each KPCA kernel configuration

## Requirements

- Python 3.12
- NumPy, scikit-learn, TensorFlow, Matplotlib, Pandas, SciPy, Pillow (PIL)

## Key Features

### New Additions

- **Faces Dataset Support**: Load emotion classification images from local directories
- **Kernel PCA (KPCA)**: Non-linear dimensionality reduction with RBF and polynomial kernels
- **Linear Discriminant Analysis (LDA)**: Supervised dimensionality reduction
- **Multi-Configuration Evaluation**: `main2.py` evaluates models across multiple KPCA kernel configurations
- **Enhanced Visualizations**: Support for faces dataset with emotion labels

---

**Course**: Computational Intelligence and Statistical Learning  
**Institution**: Aristotle University of Thessaloniki
