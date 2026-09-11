# EEG-Based Alzheimer Detection — Deep Learning Pipeline

A comparative deep learning pipeline that trains and evaluates six neural network architectures for binary Alzheimer's classification from raw EEG recordings.

## Overview

The project takes raw multi-channel EEG data and segments it into fixed-length time windows, which are then fed directly into deep learning models (no hand-crafted features) to classify recordings as Alzheimer's or healthy. Rather than committing to a single architecture, the notebook builds six different networks — from a multi-scale CNN to a Transformer encoder — and evaluates them under an identical training and cross-validation setup so their performance can be compared directly.

The pipeline is implemented as a single Jupyter notebook (`Alzheimer_DL_Pipeline.ipynb`) using TensorFlow/Keras for modeling and scikit-learn for cross-validation and evaluation metrics.

## Objectives

- Load raw EEG signal data and segment it into uniform time windows suitable for sequence models.
- Normalize each segment per channel.
- Implement and compare six deep learning architectures on the same data and evaluation protocol.
- Evaluate each model with stratified k-fold cross-validation and aggregate accuracy, precision, recall, and F1-score across folds.
- Visualize per-model training curves, confusion matrices, and metric comparisons.

## Dataset

| Property | Value |
|---|---|
| File | `HMMS.csv` (downloaded via `gdown`, file ID `1GZyHxtjJbIVjT75yxcLiIgKeMd72KmcG`) |
| Raw input | Tabular EEG samples, first 19 feature columns used as channels, plus a `label` column |
| Segmentation | Non-overlapping 1-second windows (`fs = 250` → 250 time steps per window) |
| Model input shape | `(N, 250, 19)` — N windows × 250 time steps × 19 EEG channels |
| Task | Binary classification (class labels encoded with `LabelEncoder`) |

The dataset source beyond the Google Drive file link is not documented in the notebook. Sample counts, class balance, and the exact number of resulting windows (`N`) are not available because the notebook has not been executed (see **Results** below) — the segmentation logic in the code determines these values only at runtime.

## Methodology

1. Download `HMMS.csv` and load it with pandas.
2. Drop the index column, take the first 19 remaining feature columns as EEG channels, and encode the `label` column.
3. Segment the continuous signal into non-overlapping 1-second (250-sample) windows, assigning each window the label of its first sample.
4. Z-score normalize each window independently, per channel, across the time dimension.
5. Build six candidate model architectures (see below).
6. Train each architecture with 5-fold stratified cross-validation, using early stopping, model checkpointing, and learning-rate reduction on plateau.
7. Compute per-fold accuracy, precision, recall, and F1-score; aggregate mean and standard deviation per model.
8. Generate comparison visualizations across models and folds.

No feature selection, feature engineering, or dimensionality reduction (e.g., PCA) is used in this notebook — the raw normalized EEG windows are passed directly into the networks, which learn spatial and temporal features internally (e.g., via multi-scale convolutions or attention).

## Machine Learning Models

All six models take the same `(250, 19)` input and end in a single sigmoid output unit.

| Model | Architecture notes |
|---|---|
| Multi-Scale CNN | Five parallel 1D-conv branches with kernel sizes 3/7/15/31/63 (intended as proxies for EEG frequency bands), concatenated and pooled |
| CNN + Attention | Multi-scale conv branches → squeeze-and-excitation channel attention → soft temporal attention |
| Stacked LSTM | Three stacked LSTM layers (128 → 128 → 64 units) with dropout and recurrent dropout |
| CNN + LSTM | Two conv layers with channel attention, followed by two LSTM layers (128 → 64 units) |
| CNN + BiLSTM + Attention | Multi-scale conv backbone with channel attention, two bidirectional LSTM layers, then temporal attention |
| EEG Transformer | Conv1D token embedding → sinusoidal positional encoding → 3 Transformer encoder blocks (4 heads, `d_model=64`) → temporal attention pooling |

**Training configuration (shared across all models):**

| Parameter | Value |
|---|---|
| Epochs | 50 (with early stopping, patience 7 on validation loss) |
| Batch size | 64 |
| Cross-validation | 5-fold stratified |
| Optimizer | Adam, learning rate `1e-3` (reduced on plateau, factor 0.5, patience 4) |
| Loss | Binary cross-entropy |

## Threshold Optimization

Not implemented in this notebook. Predictions are thresholded at the default 0.5 cutoff (`y_prob >= 0.5`) when computing fold-level metrics. This differs from the handwriting-based notebook in this repository, which does include explicit threshold search (see the Verification Summary below).

## Results

The notebook has not been executed — every code cell has no execution count and no output (no printed metrics, no saved CSV, no rendered plots). As a result, **no accuracy, precision, recall, F1, or best-model figures can currently be verified or reported.**

The training and evaluation code (Cell 13) is written to produce a `results_df` and export it to `DL_EEG_results.csv`, and Cells 14–19 are written to produce the plots listed below, but none of these artifacts exist in the current state of the notebook. Running the notebook end-to-end against the EEG dataset would be required to generate actual results.

## Visualizations

The notebook defines code to generate the following plots, but since it hasn't been run, none of the corresponding images currently exist:

- Bar chart comparing mean accuracy (with error bars) across all six models
- Grid of training/validation loss and accuracy curves, averaged across folds, one panel per model
- Grid of normalized confusion matrices, one per model
- Heatmap of accuracy/precision/recall/F1 across all models
- Grouped bar chart of all four metrics across all models

Running the notebook would save these as `accuracy_comparison.png`, `loss_curves.png`, `confusion_matrices.png`, `performance_heatmap.png`, and `multi_metric_bar.png`.

## Technologies Used

- Python
- NumPy, Pandas
- TensorFlow / Keras
- scikit-learn (`StratifiedKFold`, `LabelEncoder`, classification metrics)
- Matplotlib, Seaborn
- gdown (dataset download)

## Project Structure

The repository currently consists of the notebook itself; no separate scripts, saved results, or requirements file are present.

```
project/
├── Alzheimer_DL_Pipeline.ipynb
└── README.md
```

Running the notebook as written would additionally produce per-fold model checkpoints (`best_{model_name}_fold{fold}.keras`), a results CSV (`DL_EEG_results.csv`), and the PNG plots listed above.

## Installation

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn gdown
```

No `requirements.txt` is currently included in the repository.

## Usage

1. Clone the repository.
2. Install the dependencies above.
3. Open `Alzheimer_DL_Pipeline.ipynb` in Jupyter or Google Colab.
4. Run the cells in order from top to bottom. The first cell downloads `HMMS.csv` via `gdown` — this requires the linked Google Drive file to remain accessible.
5. Training all six models with 5-fold cross-validation for up to 50 epochs each is computationally intensive; a GPU runtime is recommended.

## Limitations

- The notebook has not been executed, so no performance claims are currently verifiable.
- The label for each 1-second window is taken directly from the first raw sample in that window; how this label was originally assigned at the sample level is not documented.
- Only a fixed 0.5 classification threshold is used — no threshold tuning is performed for this pipeline.
- All evaluation is done via cross-validation on a single dataset; there is no independent external test set.
- Class balance in the resulting windowed dataset is not documented.
- The project is entirely notebook-based, with no modular scripts, tests, or packaged inference code.

## Future Improvements

*(Not yet implemented — listed as potential future work only.)*

- Execute the notebook and publish verified accuracy/precision/recall/F1 results per model.
- Commit the generated result CSV and plots to the repository.
- Add a `requirements.txt` and, if feasible, a pinned environment for reproducibility.
- Evaluate threshold tuning per model, similar to the approach used in the handwriting-based notebook.
- Validate on an independent EEG dataset or held-out cohort.
- Add model interpretability (e.g., visualizing learned attention weights over EEG channels/time).

## Research / Academic Context

This is an exploratory, notebook-based research pipeline comparing deep learning architectures for EEG-based Alzheimer's classification. It is not tied to a specific publication, and no institutional or clinical validation is documented in the notebook.

## Disclaimer

This project is an academic/research exercise in applying deep learning to EEG data. It is not a validated clinical diagnostic tool and should not be used for medical decision-making.

## Author

**Abhishek Kumar Singh**

- GitHub: [github.com/Abhi-53](https://github.com/Abhi-53)
- LinkedIn: [linkedin.com/in/abhishek-ar6327](https://linkedin.com/in/abhishek-ar6327)
