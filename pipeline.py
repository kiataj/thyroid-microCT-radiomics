"""
Radiomics pipeline for thyroid tumor classification.
Implements the processing and classification steps from the manuscript:
  "Mapping 3D Heterogeneity of Thyroid Tumors Using Micro-CT based Radiomics"

Steps:
  1. Load data
  2. Sign-preserving log transform
  3. ComBat batch correction (batch = TMA + Grid)
  4. Pearson redundancy reduction (r > 0.95, p < 0.05)
  5. Classification tasks via 5-fold patient-level cross-validation:
       - Tissue type (non-neoplastic vs neoplastic)
       - Tumor type (FTN vs PTC)
       - BRAF V600E mutation status (PTC only)
       - Relapse prediction
  6. SHAP feature importance on manuscript-specified features
  7. UMAP visualisation with Silhouette score

Note: ICC-based feature filtering requires a separate reproducibility scan CSV.
If available, set ICC_CSV to its path; otherwise that step is skipped.

Dependencies:
  pandas, numpy, scipy, torch, skorch, scikit-learn, shap, umap-learn,
  pingouin, combat, matplotlib
"""

import os
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    f1_score, precision_score, recall_score, silhouette_score,
)
from scipy.spatial.distance import jensenshannon
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_CSV   = os.path.join(os.path.dirname(__file__), "embeddings_and_labels.csv")
ICC_CSV    = os.path.join(os.path.dirname(__file__), "radiomics.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature column prefixes (PyRadiomics output)
# ---------------------------------------------------------------------------
FEATURE_PREFIXES = (
    "original_",
    "wavelet-",
    "log-sigma-",
    "square_",
    "squareroot_",
    "gradient_",
    "logarithm_",
    "exponential_",
)

# ---------------------------------------------------------------------------
# Diagnosis normalisation
# TMA 2406 uses numeric codes; other TMAs use string labels.
# Mapping to a unified integer class:
#   0 = PTC   1 = FTN (FA or FTC)   2 = PDTC   3 = Oncocytic
#   4 = FVPTC (analyzed separately)   -1 = exclude
# ---------------------------------------------------------------------------
DIAGNOSIS_MAP = {
    # string labels from TMAs 64 / 93 / 94
    "PTC": 0,
    "FA":  1,
    "FTC": 1,
    "PDTC": 2,
    "FTC_PDTC": -1,
    "OC":  3,
    "OA":  3,
    "nN":  -1,
    "Nodular hyperplasia": -1,
    "Medullary Thyroid Carcinoma": -1,
    "oncocytic carcinoma und PTC": -1,
    "O_PDTC": -1,
    # numeric codes from TMA 2406 (stored as strings in CSV)
    "1": 0,    # PTC
    "2": 4,    # FVPTC (analyzed separately; excluded from FTN vs PTC task)
    "6": 0,    # PTC variant
    "4": 1,    # FTC
    "5": 1,    # FA
    "12": 1,   # FTC variant
    "3": 2,    # PDTC
    "9": 2,    # PDTC variant
    "11": 2,   # PDTC variant
    "7": 3,    # Oncocytic carcinoma
    "8": 3,    # Oncocytic variant
    "10": 3,   # Oncocytic variant
}

# Patient IDs excluded in the original analysis
EXCLUDED_IDS = [734, 560, 654]

# ---------------------------------------------------------------------------
# Manuscript SHAP features per task
# Wavelet directions LLH / HLL / LHL are equivalent after redundancy reduction;
# resolve_wavelet_feature picks whichever survived.
# ---------------------------------------------------------------------------
MANUSCRIPT_SHAP_FEATURES = {
    "tumor_type_ftn_vs_ptc": [
        # first-order statistics from wavelet LLH (direction-agnostic)
        ("wavelet", "firstorder_Median"),
        ("wavelet", "firstorder_Mean"),
        # LoG texture features
        ("log-sigma-2-mm-3D", "glrlm_ShortRunHighGrayLevelEmphasis"),
        ("log-sigma-2-mm-3D", "glcm_Autocorrelation"),
        # wavelet GLCM
        ("wavelet", "glcm_Imc2"),
    ],
    "braf_v600e": [
        # LoG first-order and texture
        ("log-sigma-2-mm-3D", "firstorder_90Percentile"),
        ("log-sigma-2-mm-3D", "firstorder_Energy"),
        ("log-sigma-2-mm-3D", "glrlm_ShortRunHighGrayLevelEmphasis"),
        # wavelet first-order
        ("wavelet", "firstorder_Median"),
        # GLDM spatial heterogeneity
        ("log-sigma-2-mm-3D", "gldm_DependenceEntropy"),
    ],
}

# Wavelet directions to search (per manuscript, LLH/HLL/LHL are equivalent)
WAVELET_DIRECTIONS = ["LLH", "HLL", "LHL", "LHH", "HHL", "HLH", "LLL", "HHH"]


def resolve_feature(prefix: str, suffix: str, available: list) -> str | None:
    """
    Return the first matching column name from available.
    For wavelet features (prefix=='wavelet'), tries all directions.
    For other prefixes, looks for exact prefix_suffix match.
    """
    if prefix == "wavelet":
        for direction in WAVELET_DIRECTIONS:
            candidate = f"wavelet-{direction}_{suffix}"
            if candidate in available:
                return candidate
    else:
        candidate = f"{prefix}_{suffix}"
        if candidate in available:
            return candidate
    return None


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data(path: str):
    df = pd.read_csv(path, low_memory=False)

    feat_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    meta_cols = ["ID", "tissue", "TMA", "Grid", "Diagnosis", "BRAF p/n", "Relapse p/n"]

    features = df[feat_cols].copy()
    meta     = df[meta_cols].copy()

    meta["diagnosis_class"] = meta["Diagnosis"].astype(str).map(DIAGNOSIS_MAP)
    meta["batch"]           = meta["TMA"].astype(str) + "_" + meta["Grid"].astype(str)
    meta["patient_id"]      = meta["ID"].fillna(-1).astype(int)

    return features, meta


# ---------------------------------------------------------------------------
# 2. Sign-preserving log transform
# ---------------------------------------------------------------------------

def sign_log_transform(features: pd.DataFrame) -> pd.DataFrame:
    arr = features.values.astype(float)
    return pd.DataFrame(
        np.sign(arr) * np.log1p(np.abs(arr)),
        index=features.index,
        columns=features.columns,
    )


# ---------------------------------------------------------------------------
# 3. ComBat batch correction
# ---------------------------------------------------------------------------

def combat_correction(features: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
    from combat.pycombat import pycombat
    from sklearn.feature_selection import VarianceThreshold

    features   = features.reset_index(drop=True)
    batch_codes = pd.Series(pd.factorize(batch.reset_index(drop=True))[0])

    # Remove zero-variance features before ComBat; constant columns cause pycombat to fail
    selector = VarianceThreshold(threshold=0.0)
    arr      = selector.fit_transform(features)
    features = pd.DataFrame(arr, columns=features.columns[selector.get_support()])
    print(f"  VarianceThreshold: {features.shape[1]} features retained")

    corrected = pycombat(features.T, batch_codes)
    corrected = corrected.T.reset_index(drop=True)

    n_before  = corrected.shape[1]
    corrected = corrected.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    n_dropped = n_before - corrected.shape[1]
    if n_dropped:
        print(f"  Dropped {n_dropped} features with NaN/Inf after ComBat")

    return corrected


# ---------------------------------------------------------------------------
# 4. ICC-based feature filtering (optional)
# ---------------------------------------------------------------------------

def icc_filter(features: pd.DataFrame, repro_csv: str, threshold=0.75, p_threshold=0.05):
    from pingouin import intraclass_corr
    from tqdm import tqdm

    repro  = pd.read_csv(repro_csv, low_memory=False)
    scan1  = repro[repro["TMA"] == "H64"].copy()
    scan2  = repro[repro["TMA"] == "V64"].copy()
    merged = scan1.merge(scan2, on=["Grid", "x", "y"], suffixes=("_s1", "_s2"))

    feat_cols = [c for c in features.columns if c in repro.columns]

    reliable = []
    for col in tqdm(feat_cols, desc="ICC"):
        c1, c2 = col + "_s1", col + "_s2"
        if c1 not in merged.columns or c2 not in merged.columns:
            continue
        pairs = merged[[c1, c2]].dropna()
        if len(pairs) < 3:
            continue
        combined = pd.DataFrame({
            "sample": list(range(len(pairs))) * 2,
            "rater":  [1] * len(pairs) + [2] * len(pairs),
            "value":  list(pairs[c1]) + list(pairs[c2]),
        })
        stats   = intraclass_corr(data=combined, targets="sample", raters="rater",
                                   ratings="value", nan_policy="omit")
        icc_val = stats["ICC"].iloc[2]
        p_val   = stats["pval"].iloc[2]
        if icc_val > threshold and p_val < p_threshold:
            reliable.append(col)

    print(f"ICC: {len(reliable)} / {len(feat_cols)} features retained")
    return features[[c for c in reliable if c in features.columns]]


# ---------------------------------------------------------------------------
# 5. Pearson redundancy reduction
# ---------------------------------------------------------------------------

def pearson_redundancy_reduction(features: pd.DataFrame,
                                  threshold=0.95, p_threshold=0.05) -> pd.DataFrame:
    from random import choice

    cols = features.columns.tolist()
    keep = np.ones(len(cols), dtype=bool)

    for col1, col2 in itertools.combinations(cols, 2):
        i1, i2 = cols.index(col1), cols.index(col2)
        if not keep[i1] or not keep[i2]:
            continue
        r, p = pearsonr(features[col1], features[col2])
        if r > threshold and p < p_threshold:
            keep[choice([i1, i2])] = False

    retained = [c for c, k in zip(cols, keep) if k]
    print(f"Redundancy reduction: {len(retained)} / {len(cols)} features retained")

    # Log retained feature names and counts
    retained_path = os.path.join(OUTPUT_DIR, "retained_features.txt")
    with open(retained_path, "w") as f:
        f.write(f"Total retained: {len(retained)}\n\n")
        # Group by filter class prefix
        from collections import Counter
        prefix_counts = Counter()
        for col in retained:
            prefix = col.split("_")[0]
            prefix_counts[prefix] += 1
        f.write("Count by filter class:\n")
        for prefix, count in sorted(prefix_counts.items()):
            f.write(f"  {prefix}: {count}\n")
        f.write("\nFull feature list:\n")
        for col in retained:
            f.write(f"  {col}\n")
    print(f"  Feature list saved to {retained_path}")

    return features[retained]


# ---------------------------------------------------------------------------
# 6. MLP model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2     = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


# ---------------------------------------------------------------------------
# 7. Patient-level stratified group k-fold
# ---------------------------------------------------------------------------

class StratifiedGroupKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits     = n_splits
        self.shuffle      = shuffle
        self.random_state = random_state

    def split(self, X, y, groups):
        groups = np.array(groups)
        y      = np.array(y)
        unique_groups = np.unique(groups)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(unique_groups)

        group_label = {
            g: np.argmax(np.bincount(y[groups == g]))
            for g in unique_groups
        }
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                               random_state=self.random_state)
        for tr_g, te_g in skf.split(unique_groups,
                                      [group_label[g] for g in unique_groups]):
            train_groups = unique_groups[tr_g]
            test_groups  = unique_groups[te_g]
            yield (np.where(np.isin(groups, train_groups))[0],
                   np.where(np.isin(groups, test_groups))[0])


# ---------------------------------------------------------------------------
# 8. UMAP + Silhouette
# ---------------------------------------------------------------------------

def compute_umap_and_plot(task_name: str, X: np.ndarray, y: np.ndarray,
                           class_labels: dict):
    from umap import UMAP

    print(f"  Computing UMAP for {task_name}...")
    reducer   = UMAP(n_neighbors=30, min_dist=0.5, n_components=2,
                     random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(X)

    sil = silhouette_score(embedding, y)
    print(f"  Silhouette score (UMAP space): {sil:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.tab10.colors
    for cls, label in class_labels.items():
        mask = y == cls
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[colors[cls % 10]], label=label, alpha=0.6, s=14, linewidths=0)
    ax.set_title(f"UMAP — {task_name}\nSilhouette: {sil:.3f}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{task_name}_umap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  UMAP plot saved to {path}")

    return embedding, sil


# ---------------------------------------------------------------------------
# 9. SHAP on manuscript features
# ---------------------------------------------------------------------------

def compute_shap_and_plot(task_name: str, model, scaler, X_bg: np.ndarray,
                           X_explain: np.ndarray, y_explain: np.ndarray,
                           feature_names: list, device: str):
    ms_specs = MANUSCRIPT_SHAP_FEATURES.get(task_name)
    if ms_specs is None:
        print(f"  No manuscript SHAP features defined for {task_name}; skipping.")
        return

    # Resolve which column survived redundancy reduction
    resolved = []
    for prefix, suffix in ms_specs:
        col = resolve_feature(prefix, suffix, feature_names)
        if col is not None:
            resolved.append(col)
        else:
            print(f"  WARNING: manuscript feature ({prefix}, {suffix}) not found in retained set")

    if not resolved:
        print(f"  No manuscript features found for {task_name}; skipping SHAP.")
        return

    print(f"  Running SHAP for {task_name} on {len(resolved)} manuscript features:")
    for col in resolved:
        print(f"    {col}")

    feat_idx = [feature_names.index(c) for c in resolved]

    model.eval()

    def predict_fn(x):
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).to(device)
            return F.softmax(model(t), dim=1).cpu().numpy()

    explainer  = shap.KernelExplainer(predict_fn, X_bg)
    shap_vals  = explainer.shap_values(X_explain, nsamples=100)

    # Extract class-1 SHAP values; handle both output formats of KernelExplainer:
    #   list of arrays  -> list[n_classes], each (n_samples, n_features)
    #   3-D numpy array -> (n_samples, n_features, n_classes)   [newer SHAP]
    if isinstance(shap_vals, list):
        sv_class1 = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv_class1 = shap_vals[:, :, 1]
    else:
        sv_class1 = shap_vals
    sv_subset  = sv_class1[:, feat_idx]
    X_subset   = X_explain[:, feat_idx]

    # Beeswarm / summary plot
    fig, ax = plt.subplots(figsize=(8, max(3, len(resolved) * 0.55)))
    shap.summary_plot(
        sv_subset, X_subset,
        feature_names=resolved,
        show=False, plot_type="dot",
        color_bar=True,
    )
    plt.title(f"SHAP — {task_name}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{task_name}_shap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP plot saved to {path}")

    # Save SHAP values for manuscript features to CSV
    df_shap = pd.DataFrame(sv_subset, columns=resolved)
    df_shap["true_label"] = y_explain
    df_shap.to_csv(os.path.join(OUTPUT_DIR, f"{task_name}_shap_values.csv"), index=False)


# ---------------------------------------------------------------------------
# 10. Probability distribution plot + Jensen-Shannon divergence
# ---------------------------------------------------------------------------

def plot_probability_distribution(task_name: str, probs: np.ndarray,
                                   targets: np.ndarray, class_labels: dict) -> float:
    """
    Overlapping histogram of predicted class-1 probability, split by true class.
    Computes Jensen-Shannon divergence between the two class distributions.
    Returns the JSD value.
    """
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Build a normalised probability density for each class over the same bins
    class_probs = {}
    for cls in sorted(class_labels.keys()):
        vals = probs[targets == cls]
        counts, _ = np.histogram(vals, bins=bin_edges, density=False)
        # Convert to a proper probability vector (sums to 1)
        total = counts.sum()
        class_probs[cls] = counts / total if total > 0 else counts + 1e-10

    # Jensen-Shannon divergence between class 0 and class 1 distributions
    p = class_probs[0]
    q = class_probs[1]
    # jensenshannon returns the distance (sqrt of divergence); square to get JSD
    jsd = jensenshannon(p, q, base=2) ** 2
    print(f"  Jensen-Shannon divergence (class distributions): {jsd:.4f}")

    # Plot
    colors = ["#e07070", "#6fa8d6"]   # red-ish for class 0, blue for class 1
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, color in zip(sorted(class_labels.keys()), colors):
        vals  = probs[targets == cls]
        label = class_labels[cls]
        ax.hist(vals, bins=bin_edges, alpha=0.55, color=color,
                label=label, density=True, edgecolor="none")

    ax.set_xlabel("Predicted probability (class 1)")
    ax.set_ylabel("Density")
    ax.set_title(f"Predicted probability distribution — {task_name}\nJSD = {jsd:.3f}")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{task_name}_prob_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Probability distribution plot saved to {path}")

    return jsd


# ---------------------------------------------------------------------------
# 12. Run one classification task
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    features: pd.DataFrame,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    class_labels: dict = None,
    hidden_dim: int = 256,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    dropout_rate: float = 0.1,
    gamma: float = 0.9,
    stratified: bool = False,
    threshold: float = 0.5,
):
    print(f"\n{'='*60}\nTask: {task_name}\n{'='*60}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        lbl = class_labels.get(int(u), str(u)) if class_labels else str(u)
        print(f"  Class {u} ({lbl}): {c} samples")

    if class_labels is None:
        class_labels = {int(u): str(u) for u in unique}

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim  = features.shape[1]
    output_dim = len(unique)

    X = features.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = labels.astype(np.int64)

    feat_names = features.columns.tolist()

    splitter = (
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        if stratified
        else GroupKFold(n_splits=5)
    )

    metrics_rows    = []
    all_val_probs   = []
    all_val_targets = []
    roc_curves      = []

    # Variables for SHAP (captured from last fold)
    shap_pipe = None
    shap_tr_idx = shap_va_idx = None

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, patient_ids)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx],   y[val_idx]

        net = NeuralNetClassifier(
            module=MLP,
            module__input_dim=input_dim,
            module__hidden_dim=hidden_dim,
            module__output_dim=output_dim,
            module__dropout_rate=dropout_rate,
            max_epochs=num_epochs,
            batch_size=batch_size,
            optimizer=optim.Adam,
            optimizer__lr=learning_rate,
            optimizer__weight_decay=weight_decay,
            criterion=nn.CrossEntropyLoss,
            device=device,
            callbacks=[("lr_scheduler", LRScheduler(policy="ExponentialLR", gamma=gamma))],
            train_split=None,
            verbose=0,
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("net", net)])
        pipe.fit(X_tr, y_tr)

        val_probs = pipe.predict_proba(X_va)
        val_preds = (val_probs[:, 1] >= threshold).astype(int)

        val_auc  = roc_auc_score(y_va, val_probs[:, 1])
        val_acc  = accuracy_score(y_va, val_preds)
        val_f1   = f1_score(y_va, val_preds, average="weighted", zero_division=0)
        val_prec = precision_score(y_va, val_preds, average="weighted", zero_division=0)
        val_rec  = recall_score(y_va, val_preds, average="weighted", zero_division=0)

        fpr, tpr, _ = roc_curve(y_va, val_probs[:, 1])
        roc_curves.append((fpr, tpr))

        all_val_probs.append(val_probs[:, 1])
        all_val_targets.append(y_va)

        metrics_rows.append({
            "fold": fold + 1,
            "val_auc": val_auc,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec,
        })
        print(f"  Fold {fold+1}: AUC={val_auc:.3f}  Acc={val_acc:.3f}")

        shap_pipe   = pipe
        shap_tr_idx = train_idx
        shap_va_idx = val_idx

    summary  = pd.DataFrame(metrics_rows)
    mean_row = summary.mean(numeric_only=True)
    std_row  = summary.std(numeric_only=True)
    print(f"\n  Mean AUC: {mean_row['val_auc']:.3f} ± {std_row['val_auc']:.3f}")

    out_path = os.path.join(OUTPUT_DIR, f"{task_name}_metrics.csv")
    summary.to_csv(out_path, index=False)

    # --- Probability distribution + JSD ---
    val_probs_cat   = np.concatenate(all_val_probs)
    val_targets_cat = np.concatenate(all_val_targets)
    jsd = plot_probability_distribution(task_name, val_probs_cat, val_targets_cat, class_labels)
    summary["jsd"] = jsd   # attach scalar to summary for reference

    # --- UMAP + Silhouette ---
    compute_umap_and_plot(task_name, X, y, class_labels)

    # --- SHAP on manuscript features (last fold model) ---
    if shap_pipe is not None and task_name in MANUSCRIPT_SHAP_FEATURES:
        scaler     = shap_pipe.named_steps["scaler"]
        model      = shap_pipe.named_steps["net"].module_
        X_tr_sc    = scaler.transform(X[shap_tr_idx]).astype(np.float32)
        X_va_sc    = scaler.transform(X[shap_va_idx]).astype(np.float32)
        bg         = X_tr_sc[np.random.choice(len(X_tr_sc), min(200, len(X_tr_sc)),
                                               replace=False)]
        compute_shap_and_plot(task_name, model, scaler, bg, X_va_sc,
                               y[shap_va_idx], feat_names, device)

    return {
        "metrics":      summary,
        "roc_curves":   roc_curves,
        "val_probs":    np.concatenate(all_val_probs),
        "val_targets":  np.concatenate(all_val_targets),
        "feature_names": feat_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load ---
    print("Loading data...")
    features, meta = load_data(DATA_CSV)
    print(f"Loaded {len(features)} samples, {features.shape[1]} features")

    # --- Log transform ---
    print("Applying sign-preserving log transform...")
    features = sign_log_transform(features)

    # --- ComBat batch correction ---
    print("Running ComBat batch correction...")
    features = combat_correction(features, meta["batch"])

    # --- ICC filtering (optional) ---
    if ICC_CSV is not None and os.path.isfile(ICC_CSV):
        print("Applying ICC feature filtering...")
        features = icc_filter(features, ICC_CSV)
    else:
        print("ICC CSV not provided; skipping ICC filtering step.")

    # --- Redundancy reduction ---
    print("Running Pearson redundancy reduction...")
    features = pearson_redundancy_reduction(features)

    # Align meta index with reset features index
    meta = meta.reset_index(drop=True)

    # Exclude known problematic patient IDs
    mask_valid = ~meta["patient_id"].isin(EXCLUDED_IDS)

    # -----------------------------------------------------------------------
    # Task 1: Tissue type (non-neoplastic N=0 vs neoplastic Tu=1)
    # -----------------------------------------------------------------------
    mask = mask_valid & meta["tissue"].isin(["N", "Tu"])
    X_t1 = features.loc[mask]
    y_t1 = (meta.loc[mask, "tissue"] == "Tu").astype(int).values
    g_t1 = meta.loc[mask, "patient_id"].values

    run_task("tissue_type", X_t1, y_t1, g_t1,
             class_labels={0: "Non-neoplastic", 1: "Neoplastic"})

    # -----------------------------------------------------------------------
    # Task 2: Tumor type (FTN=0 vs PTC=1)
    # FVPTC (class 4) excluded; only FTN (class 1) and PTC (class 0)
    # -----------------------------------------------------------------------
    mask = (
        mask_valid
        & (meta["tissue"] == "Tu")
        & meta["diagnosis_class"].isin([0, 1])
    )
    X_t2 = features.loc[mask]
    y_t2 = meta.loc[mask, "diagnosis_class"].astype(int).values
    g_t2 = meta.loc[mask, "patient_id"].values

    run_task("tumor_type_ftn_vs_ptc", X_t2, y_t2, g_t2,
             class_labels={0: "PTC", 1: "FTN"})

    # -----------------------------------------------------------------------
    # Task 3: BRAF V600E status (0=negative, 1=positive)
    # FTN excluded; includes PTC, FVPTC, PDTC, Oncocytic
    # -----------------------------------------------------------------------
    mask = (
        mask_valid
        & (meta["tissue"] == "Tu")
        & (~meta["diagnosis_class"].isin([1, -1]))
        & meta["BRAF p/n"].notna()
    )
    X_t3 = features.loc[mask]
    y_t3 = meta.loc[mask, "BRAF p/n"].astype(int).values
    g_t3 = meta.loc[mask, "patient_id"].values

    run_task("braf_v600e", X_t3, y_t3, g_t3,
             class_labels={0: "BRAF-negative", 1: "BRAF-positive"})

    # -----------------------------------------------------------------------
    # Task 4: Relapse prediction (0=no relapse, 1=relapse)
    # TMA 2406 excluded (no relapse follow-up data)
    # -----------------------------------------------------------------------
    mask = (
        mask_valid
        & (meta["TMA"] != 2406)
        & meta["Relapse p/n"].notna()
    )
    X_t4 = features.loc[mask]
    y_t4 = meta.loc[mask, "Relapse p/n"].astype(int).values
    g_t4 = meta.loc[mask, "patient_id"].values

    run_task("relapse", X_t4, y_t4, g_t4,
             class_labels={0: "Disease-free", 1: "Relapse"})

    # -----------------------------------------------------------------------
    # Task 5: TERT promoter mutation
    # Label not present in this CSV. Provide a TERT label column and
    # uncomment the block below.
    # Use stratified=True due to small positive class (n=8).
    # -----------------------------------------------------------------------
    # mask = mask_valid & meta["TERT p/n"].notna()
    # X_t5 = features.loc[mask]
    # y_t5 = meta.loc[mask, "TERT p/n"].astype(int).values
    # g_t5 = meta.loc[mask, "patient_id"].values
    # run_task("tert_mutation", X_t5, y_t5, g_t5,
    #          class_labels={0: "TERT-wildtype", 1: "TERT-mutated"},
    #          stratified=True, threshold=0.2)

    print(f"\nDone. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
