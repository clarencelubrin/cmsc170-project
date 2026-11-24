# CMSC 170 Project

Classifies each Philippine province into a policy class (`INCREASE`, `RETAIN` or `DECREASE`) from PSA poverty, income and expenditure data, using three models written from scratch:

- **K-Means Clustering** (`models/clustering.py`)
- **K-Nearest Neighbors** (`models/knearest.py`)
- **Multi-Layer Perceptron** (`models/mlp.py`)

Each model is evaluated with holdout validation, k-fold cross validation and a confusion matrix (`evaluation/`).

## Requirements

- Python 3.11 or newer (developed on 3.14)
- `numpy`, `matplotlib` and Jupyter

## Setup

Run these from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy matplotlib jupyter
```

The dataset is already in `PROJECT_DATASET/`, so nothing needs to be downloaded.

## Running the code

### Run the notebooks

Open a notebook and choose **Run All**. In VS Code, open the notebook and select the `.venv` kernel instead.

| Notebook | What it does |
| --- | --- |
| `k-means-clustering.ipynb` | Clusters the provinces, labels each cluster by majority vote, then validates |
| `k-nearest-neighbors.ipynb` | KNN with cross validation, holdout validation and tuning of `k` |
| `mlp.ipynb` | MLP with a grid search of the hyperparameters, then holdout and k-fold validation |
| `model-comparison.ipynb` | Runs the three models with the same settings and charts their accuracy and metrics |

## Reproducing the results

- The results in the report are the outputs of `model-comparison.ipynb`. Its settings (train ratio, folds, clusters, `k`, hidden neurons, learning rate, iterations) are in the **Settings** cell at the top.
- The splits and the starting centroids/weights are seeded with `random.seed(1)`, so the validation results are the same on every run.
- The exploratory clustering in Step 3 of `k-means-clustering.ipynb` is not seeded, so its clusters and 3D plots can change between runs. The validation steps after it are seeded.
- The figures in `docs/assets/` were saved by hand from the notebook outputs. They are not written automatically when a notebook runs.
