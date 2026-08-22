# COMP662 Assignment 1 — Synthetic Dataset Generation

**Student ID:** 1173808  
**Repository:** https://github.com/yang-liu-1173808/1173808-COMP662A1  
**Due Date:** 23 August 2026, 5:00 p.m. NZST  

---

## Repository Structure

```
1173808-COMP662A1/
├── notebooks/
│   └── 1173808_Assignment1.ipynb   # Main Jupyter Notebook covering Tasks 1–6
├── models/
│   └── 1173808_Assignment1_final.joblib  # Trained final classification pipeline
├── data/                            # Dataset directory (train.csv / synthetic data)
├── figures/                         # Exported visualisations from EDA & evaluation
├── predict.py                       # Prediction script for new CSV datasets
├── requirements.txt                 # Environment dependencies with fixed versions
├── run.txt                          # Detailed execution guide & specifications
├── GenAI_Acknowledgement.txt        # GenAI tool usage statement
├── GitHub_URL.txt                   # GitHub repository URL for submission
└── README.md
```

## Environment Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Notebook

To run the complete analysis, generative modeling, and baseline/augmented model evaluations:

```bash
jupyter notebook notebooks/1173808_Assignment1.ipynb
```

The notebook executes top-to-bottom without external dependencies, automatically using Apple Silicon MPS / CUDA GPU if available.

## Generating Predictions

To generate bean category predictions on a new CSV containing the 10 numerical features (`Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `Eccentricity`, `ConvexArea`, `Extent`, `Solidity`, `Roundness`, `Compactness`):

```bash
python predict.py <input.csv> <output.csv>
```

**Example:**
```bash
python predict.py data/train.csv data/predictions.csv
```

The script outputs a CSV containing a single `Class` column (integers 0–4). The saved model bundle is a self-contained scikit-learn Pipeline (`StandardScaler` + `RandomForestClassifier`) that automatically handles feature scaling and column alignment.

## Submission Checklist

- [x] **Task 1:** Data understanding & concise exploratory data analysis
- [x] **Task 2:** Baseline classifier construction with justified metrics and CV
- [x] **Task 3:** Conditional VAE generative model design and architecture justification
- [x] **Task 4:** Synthetic data generation and quality assessment (fidelity & diversity)
- [x] **Task 5:** Augmented classifier performance analysis and class-level evaluation
- [x] **Task 6:** Final exportable model bundle (`models/1173808_Assignment1_final.joblib`) and `predict.py`
