# Project Reorganization

This project has been reorganized into a modular structure for better maintainability and clarity.

## Directory Structure

```text
├── data/
│   ├── raw/             # Original image folders (real/fake) and dataset_index.csv
│   └── processed/       # Extracted features (.npy) and train/test splits (.csv)
├── results/             # Metrics, JSON result files, and generated plots
├── src/
│   ├── preprocessing/   # Scripts for data download and augmentation
│   ├── features/        # Feature extraction and dataset indexing/splitting
│   ├── models/          # Model training and prediction entry points
│   ├── analysis/        # Evaluation, visualization, and experiment scripts
│   ├── utils/           # Shared utility functions (e.g., JPEG compression)
│   └── config.py        # Centralized path management
├── requirements.txt     # Python dependencies
└── README.md            # You are here
```

## Running Scripts

All scripts have been updated to use a centralized path configuration in `src/config.py`. They can be run from the root directory:

```bash
# Example: Training and evaluating the model
python src/models/train_and_evaluate.py

# Example: Running a specific experiment
python src/analysis/blur.py
```

## Key Changes
- **Path Management**: Each script now adds the project root to `sys.path` and imports from `config.py` to resolve paths accurately.
- **Data Segregation**: Raw assets are kept separate from processed features and final results.
- **Modularity**: Code is grouped by logical function (preprocessing, modeling, analysis).
