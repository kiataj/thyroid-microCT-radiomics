# Thyroid Micro-CT Radiomics

Code and data for the paper **"3D Radiomics Profiling of Thyroid Tumors Using Micro-CT"**.

Radiomics features were extracted from 3D micro-CT scans of next-generation tissue microarray (ngTMA) punches (600 µm diameter, 5 µm voxel size) from 418 thyroid tumor patients. A multilayer perceptron (MLP) classifier was trained on these features for several clinically relevant classification tasks.

---

## Repository structure

```
.
├── radiomics_v6.csv       # PyRadiomics feature matrix (one row per punch)
├── ngTMA_table.csv        # Sample metadata and labels
├── run_pipeline.py        # Main entry point
├── visualize.py           # Figure generation
├── requirements.txt       # Python dependencies
└── radiomics/             # Pipeline package
    ├── config.py          # Paths, diagnosis map, task configuration
    ├── data.py            # Data loading and label merging
    ├── preprocessing.py   # Feature filtering, batch correction, redundancy reduction
    ├── cv.py              # Stratified patient-level k-fold splitter
    ├── mlp.py             # PyTorch MLP module
    ├── models.py          # TaskRunner: cross-validation loop
    ├── evaluation.py      # Metrics, bootstrap CI, JSD
    ├── explainability.py  # SHAP analysis
    └── umap_utils.py      # UMAP projection utilities
```

---

## Data files

### `radiomics_v6.csv`
PyRadiomics feature matrix. Each row corresponds to one ngTMA punch. Features are extracted from the original image and seven filter classes: Laplacian of Gaussian (sigma = 10 µm), wavelet (Haar, level 1, all directions except HHH), logarithm, square root, gradient, exponential, and square. Six feature classes are computed per filter image: first-order statistics, GLCM, GLRLM, GLDM, GLSZM, and NGTDM. Shape features are not extracted. Total: 1302 features before filtering.

### `ngTMA_table.csv`
Sample metadata and mutation labels with the following columns:

| Column | Description |
|--------|-------------|
| `ID` | Internal sample identifier |
| `PID` | Patient identifier (used for patient-level CV grouping) |
| `tissue` | Tissue type: `N` (non-neoplastic) or `Tu` (neoplastic) |
| `TMA` | TMA block identifier |
| `Grid` | Grid identifier within the TMA block |
| `x`, `y` | Punch coordinates within the grid |
| `TERT` | TERT promoter mutation status (0 = wild-type, 1 = mutant; determined by Sanger sequencing) |
| `Diagnosis` | Histological diagnosis string |
| `Relapse p/n` | Relapse status (0 = disease-free, 1 = relapsed) |
| `BRAF p/n` | BRAF V600E mutation status (0 = wild-type, 1 = mutant; determined by IHC, VE1 clone) |
| `RAS` | RAS Q61R mutation status (0 = wild-type, 1 = mutant; determined by IHC, SP174 antibody) |

---

## Installation

Python 3.9 or later is recommended.

```bash
pip install -r requirements.txt
```

---

## Usage

Run all classification tasks:

```bash
python run_pipeline.py
```

Run specific tasks:

```bash
python run_pipeline.py --task-1          # Tissue type (non-neoplastic vs neoplastic)
python run_pipeline.py --task-2          # Tumor type (PTC vs FTN)
python run_pipeline.py --task-3          # BRAF V600E mutation status
python run_pipeline.py --tert            # TERT Mann-Whitney U analysis
python run_pipeline.py --fvptc           # FVPTC similarity analysis
python run_pipeline.py --task-2 --shap   # Run task with SHAP analysis
```

---

## Pipeline

### 1. Feature preprocessing (global, before cross-validation)

All preprocessing steps are applied to the full dataset prior to cross-validation:

1. **Zero-variance filter** — removes features with no variation across the dataset.
2. **ICC filtering** — retains features with intraclass correlation coefficient ICC > 0.75 (p < 0.05) between two scans of the same block acquired under substantially different conditions. Results are cached in `results/icc_features.txt`.
3. **Sign-preserving log transform** — `sign(x) * log1p(|x|)` applied to stabilize variance and reduce skewness.
4. **ComBat batch correction** — corrects batch effects across the six ngTMA batches (four blocks, two of which contained two grids each). Batch label is TMA + Grid.
5. **Pearson redundancy reduction** — iteratively removes one feature from each pair with |r| > 0.75 (p < 0.05). Results are cached in `results/retained_feature_names.txt`.

### 2. Classification

Each task is evaluated using stratified patient-level 5-fold cross-validation (`StratifiedGroupKFold`): all punches from one patient stay in the same fold, and class balance is maintained across folds. Within each fold, features are standardized to zero mean and unit variance using statistics from the training partition only.

The classifier is a multilayer perceptron (MLP) with three hidden layers, ReLU activations, and dropout. Optimizer: Adam (lr = 1e-3, weight decay = 1e-3). Learning rate scheduler: ExponentialLR (gamma = 0.9).

Performance is reported as mean AUC ± SD across 5 folds, with 95% confidence intervals from 2000-iteration bootstrap resampling.

### 3. Classification tasks

| Task | Description | Samples |
|------|-------------|---------|
| Task 1 | Non-neoplastic vs neoplastic tissue | 336 N, 405 Tu |
| Task 2 | PTC vs follicular thyroid neoplasm (FTN = FA + FTC) | 97 PTC, 116 FTN |
| Task 3 | BRAF V600E wild-type vs mutant (PTC only) | 159 WT, 80 mutant |
| TERT | Exploratory Mann-Whitney U test (TERT WT vs mutant) | 103 WT, 8 mutant |
| FVPTC | Out-of-distribution similarity analysis using trained Task 2 classifier | 47 FVPTC |

### 4. Outputs

All outputs are written to the `results/` directory:

- Per-fold metrics CSV
- Bootstrap confidence intervals CSV
- Aggregated validation probabilities CSV
- ROC curves CSV
- SHAP summary plots (when `--shap` is passed)
- FVPTC KDE similarity plot
- TERT Mann-Whitney results CSV

---

## Citation

If you use this code or data, please cite:

> Tajbakhsh, Kiarash, et al. "Mapping 3D Heterogeneity of Thyroid Tumors Using Micro-CT based Radiomics." *bioRxiv* (2025): 2025-06.
