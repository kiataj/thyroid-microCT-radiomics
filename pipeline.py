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
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    f1_score, precision_score, recall_score, silhouette_score,
)
from scipy.spatial.distance import jensenshannon
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_CSV              = os.path.join(os.path.dirname(__file__), "embeddings_and_labels.csv")
ICC_CSV               = os.path.join(os.path.dirname(__file__), "radiomics.csv")
OUTPUT_DIR            = os.path.join(os.path.dirname(__file__), "results")
FEATURE_CACHE         = os.path.join(OUTPUT_DIR, "retained_feature_names.txt")
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


def sign_log_transform_arr(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


class BatchCorrector:
    """Per-batch mean/variance correction with fit/transform API."""

    def fit(self, X: np.ndarray, batches: np.ndarray) -> "BatchCorrector":
        self.grand_mean_ = X.mean(axis=0)
        gs = X.std(axis=0)
        self.grand_std_  = np.where(gs > 0, gs, 1.0)
        self.batch_stats_: dict = {}
        for b in np.unique(batches):
            mask = batches == b
            m = X[mask].mean(axis=0)
            s = X[mask].std(axis=0)
            self.batch_stats_[b] = (m, np.where(s > 0, s, 1.0))
        return self

    def transform(self, X: np.ndarray, batches: np.ndarray) -> np.ndarray:
        out = np.empty_like(X, dtype=float)
        for i, b in enumerate(batches):
            m, s = self.batch_stats_.get(b, (self.grand_mean_, self.grand_std_))
            out[i] = (X[i] - m) / s * self.grand_std_ + self.grand_mean_
        return out

    def fit_transform(self, X: np.ndarray, batches: np.ndarray) -> np.ndarray:
        return self.fit(X, batches).transform(X, batches)


def pearson_feature_mask(X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Returns boolean keep-mask; correlated pairs are resolved by dropping the higher-index column."""
    corr = np.corrcoef(X.T)
    n    = corr.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        drop = np.where(keep & (np.abs(corr[i]) > threshold))[0]
        drop = drop[drop > i]
        keep[drop] = False
    return keep


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
    X    = features.values.astype(float)
    corr = np.corrcoef(X.T)
    n    = len(cols)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        drop = np.where(keep & (np.abs(corr[i]) > threshold))[0]
        drop = drop[drop > i]
        keep[drop] = False

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
# 8. UMAP + Silhouette — saves embedding data only (no plot)
# ---------------------------------------------------------------------------

def compute_umap_and_save(task_name: str, X: np.ndarray, y: np.ndarray,
                           class_labels: dict, out_dir: str = None):
    import json
    from umap import UMAP

    print(f"  Computing UMAP for {task_name}...")
    reducer   = UMAP(n_neighbors=10, min_dist=0.1, n_components=2,
                     random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(X)

    sil = silhouette_score(embedding, y)
    print(f"  Silhouette score (UMAP space): {sil:.4f}")

    _out = out_dir or OUTPUT_DIR
    pd.DataFrame({"umap1": embedding[:, 0], "umap2": embedding[:, 1],
                  "label": y}).to_csv(
        os.path.join(_out, f"{task_name}_umap_embedding.csv"), index=False)

    with open(os.path.join(_out, f"{task_name}_umap_meta.json"), "w") as f:
        json.dump({"silhouette": sil,
                   "class_labels": {str(k): v for k, v in class_labels.items()}}, f)
    print(f"  UMAP embedding saved to {_out}")
    return embedding, sil


# ---------------------------------------------------------------------------
# 9. SHAP — saves top-5 data only (no plot)
# ---------------------------------------------------------------------------

def compute_shap_rf_and_save(task_name: str, rf_model,
                              X_explain: np.ndarray, y_explain: np.ndarray,
                              feature_names: list, out_dir: str = None):
    print(f"  Running SHAP (RF) for {task_name}...")
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_explain)

    if isinstance(shap_vals, list):
        sv_class1 = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv_class1 = shap_vals[:, :, 1]
    else:
        sv_class1 = shap_vals

    mean_abs   = np.abs(sv_class1).mean(axis=0)
    top5_idx   = np.argsort(mean_abs)[::-1][:5]
    top5_names = [feature_names[i] for i in top5_idx]

    print(f"  Top 5 SHAP features (RF):")
    for name, idx in zip(top5_names, top5_idx):
        print(f"    {name}  (mean |SHAP| = {mean_abs[idx]:.4f})")

    _out = out_dir or OUTPUT_DIR
    df_shap = pd.DataFrame(sv_class1[:, top5_idx], columns=top5_names)
    df_shap["true_label"] = y_explain
    df_shap.to_csv(os.path.join(_out, f"{task_name}_rf_shap_values.csv"), index=False)

    df_feat = pd.DataFrame(X_explain[:, top5_idx], columns=top5_names)
    df_feat.to_csv(os.path.join(_out, f"{task_name}_rf_shap_feature_values.csv"), index=False)

    pd.DataFrame({"rank": range(1, 6), "feature": top5_names,
                  "mean_abs_shap": mean_abs[top5_idx]}).to_csv(
        os.path.join(_out, f"{task_name}_rf_shap_top5.csv"), index=False)
    print(f"  RF SHAP data saved to {_out}")


# ---------------------------------------------------------------------------
# 10. Probability data + JSD — saves data only (no plot)
# ---------------------------------------------------------------------------

def save_probability_data(task_name: str, probs: np.ndarray,
                           targets: np.ndarray, class_labels: dict,
                           out_dir: str = None) -> float:
    n_bins    = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    class_hist = {}
    for cls in sorted(class_labels.keys()):
        vals   = probs[targets == cls]
        counts, _ = np.histogram(vals, bins=bin_edges, density=False)
        total  = counts.sum()
        class_hist[cls] = counts / total if total > 0 else counts + 1e-10

    jsd = jensenshannon(class_hist[0], class_hist[1], base=2) ** 2
    print(f"  Jensen-Shannon divergence: {jsd:.4f}")

    _out = out_dir or OUTPUT_DIR
    pd.DataFrame({"prob_class1": probs, "true_label": targets}).to_csv(
        os.path.join(_out, f"{task_name}_val_probs.csv"), index=False)

    import json
    with open(os.path.join(_out, f"{task_name}_prob_meta.json"), "w") as f:
        json.dump({"jsd": jsd,
                   "class_labels": {str(k): v for k, v in class_labels.items()}}, f)
    return jsd


# ---------------------------------------------------------------------------
# 12. Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(probs: np.ndarray, targets: np.ndarray,
                 n_boot: int = 2000, ci: float = 0.95,
                 threshold: float = 0.5, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(targets)
    stats = {"auc": [], "accuracy": [], "f1": [], "precision": [], "recall": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p, t = probs[idx], targets[idx]
        if len(np.unique(t)) < 2:
            continue
        preds = (p >= threshold).astype(int)
        stats["auc"].append(roc_auc_score(t, p))
        stats["accuracy"].append(accuracy_score(t, preds))
        stats["f1"].append(f1_score(t, preds, average="weighted", zero_division=0))
        stats["precision"].append(precision_score(t, preds, average="weighted", zero_division=0))
        stats["recall"].append(recall_score(t, preds, average="weighted", zero_division=0))
    alpha = (1 - ci) / 2
    return {k: (np.quantile(v, alpha), np.quantile(v, 1 - alpha))
            for k, v in stats.items() if v}


# ---------------------------------------------------------------------------
# 13. Run one classification task
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    features: pd.DataFrame,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    batches: np.ndarray,
    class_labels: dict = None,
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

    task_dir = os.path.join(OUTPUT_DIR, task_name)
    os.makedirs(task_dir, exist_ok=True)

    from sklearn.feature_selection import VarianceThreshold as VT

    output_dim = len(unique)

    X_raw          = features.values.astype(float)
    X_raw          = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y              = labels.astype(np.int64)
    feat_names_raw = features.columns.tolist()

    splitter = (
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        if stratified
        else GroupKFold(n_splits=5)
    )

    metrics_xgb = []
    probs_xgb   = []
    targets_all = []
    roc_xgb     = []

    last: dict = {}

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_raw, y, patient_ids)):
        y_tr, y_va = y[train_idx], y[val_idx]

        # 1. Log transform
        X_tr_log = sign_log_transform_arr(X_raw[train_idx])
        X_va_log = sign_log_transform_arr(X_raw[val_idx])

        # 2. VarianceThreshold — fit on train
        vt = VT(threshold=0.0)
        X_tr_vt = vt.fit_transform(X_tr_log)
        X_va_vt = vt.transform(X_va_log)
        vt_names = [feat_names_raw[i] for i in vt.get_support(indices=True)]

        # 3. Batch correction — fit on train, transform val
        bc = BatchCorrector()
        X_tr = bc.fit_transform(X_tr_vt, batches[train_idx]).astype(np.float32)
        X_va = bc.transform(X_va_vt, batches[val_idx]).astype(np.float32)

        feat_names = vt_names

        # ---- XGBoost ----
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        spw   = n_neg / n_pos if n_pos > 0 else 1.0
        base_params = dict(n_estimators=500, learning_rate=0.05, max_depth=4,
                           subsample=0.8, colsample_bytree=0.8)
        xgb   = XGBClassifier(
            **base_params,
            scale_pos_weight=spw,
            eval_metric="logloss", random_state=42,
            n_jobs=-1, verbosity=0,
        )
        pipe_xgb = Pipeline([("scaler", StandardScaler()), ("xgb", xgb)])
        pipe_xgb.fit(X_tr, y_tr)

        vp_xgb    = pipe_xgb.predict_proba(X_va)
        vp_xgb_tr = pipe_xgb.predict_proba(X_tr)
        pd_xgb  = (vp_xgb[:, 1] >= threshold).astype(int)
        fpr_x, tpr_x, _ = roc_curve(y_va, vp_xgb[:, 1])
        roc_xgb.append((fpr_x, tpr_x))
        probs_xgb.append(vp_xgb[:, 1])
        train_auc = roc_auc_score(y_tr, vp_xgb_tr[:, 1])
        val_auc   = roc_auc_score(y_va, vp_xgb[:, 1])
        metrics_xgb.append({
            "fold": fold + 1,
            "train_auc":     train_auc,
            "val_auc":       val_auc,
            "auc_gap":       train_auc - val_auc,
            "val_accuracy":  accuracy_score(y_va, pd_xgb),
            "val_f1":        f1_score(y_va, pd_xgb, average="weighted", zero_division=0),
            "val_precision": precision_score(y_va, pd_xgb, average="weighted", zero_division=0),
            "val_recall":    recall_score(y_va, pd_xgb, average="weighted", zero_division=0),
        })

        targets_all.append(y_va)
        print(f"  Fold {fold+1}: train_AUC={train_auc:.3f}  val_AUC={val_auc:.3f}  gap={train_auc - val_auc:.3f}")

        last = dict(pipe_xgb=pipe_xgb, vt=vt, bc=bc,
                    feat_names=feat_names, train_idx=train_idx, val_idx=val_idx,
                    X_tr=X_tr, X_va=X_va)

    def _save_model_results(tag, metrics_rows, val_probs_list, roc_curves):
        df = pd.DataFrame(metrics_rows)
        m, s = df.mean(numeric_only=True), df.std(numeric_only=True)
        print(f"\n  [{tag}] train AUC: {m['train_auc']:.3f} ± {s['train_auc']:.3f}  "
              f"val AUC: {m['val_auc']:.3f} ± {s['val_auc']:.3f}  "
              f"mean gap: {m['auc_gap']:.3f}")
        df.to_csv(os.path.join(task_dir, f"{task_name}_{tag}_metrics.csv"), index=False)

        probs_cat   = np.concatenate(val_probs_list)
        targets_cat = np.concatenate(targets_all)

        ci_dict = bootstrap_ci(probs_cat, targets_cat, threshold=threshold)
        print(f"  [{tag}] AUC 95% CI: [{ci_dict['auc'][0]:.3f}, {ci_dict['auc'][1]:.3f}]")
        pd.DataFrame([{"metric": k, "ci_lower": v[0], "ci_upper": v[1]}
                      for k, v in ci_dict.items()]).to_csv(
            os.path.join(task_dir, f"{task_name}_{tag}_bootstrap_ci.csv"), index=False)

        save_probability_data(f"{task_name}_{tag}", probs_cat, targets_cat, class_labels, out_dir=task_dir)

        fpr_grid = np.linspace(0, 1, 100)
        roc_rows = []
        for fold_i, (fpr, tpr) in enumerate(roc_curves):
            for f, t in zip(fpr_grid, np.interp(fpr_grid, fpr, tpr)):
                roc_rows.append({"fold": fold_i + 1, "fpr": f, "tpr": t})
        pd.DataFrame(roc_rows).to_csv(
            os.path.join(task_dir, f"{task_name}_{tag}_roc_curves.csv"), index=False)
        return df

    summary_xgb = _save_model_results("xgb", metrics_xgb, probs_xgb, roc_xgb)

    # --- UMAP (model-independent, use last fold preprocessing) ---
    if last:
        X_full_log = sign_log_transform_arr(X_raw)
        X_full_vt  = last["vt"].transform(X_full_log)
        X_full     = last["bc"].transform(X_full_vt, batches).astype(np.float32)
        compute_umap_and_save(task_name, X_full, y, class_labels, out_dir=task_dir)

    # --- SHAP (last fold XGBoost) ---
    if last:
        feat_names = last["feat_names"]

        # XGBoost SHAP
        scaler_xgb = last["pipe_xgb"].named_steps["scaler"]
        xgb_model  = last["pipe_xgb"].named_steps["xgb"]
        X_va_xgb   = scaler_xgb.transform(last["X_va"]).astype(np.float32)
        compute_shap_rf_and_save(f"{task_name}_xgb", xgb_model, X_va_xgb,
                                 y[last["val_idx"]], feat_names, out_dir=task_dir)

    return {
        "metrics_xgb":   summary_xgb,
        "feature_names": last.get("feat_names", []),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load ---
    print("Loading data...")
    features, meta = load_data(DATA_CSV)
    print(f"Loaded {len(features)} samples, {features.shape[1]} features")

    # --- Feature selection: load from cache or compute ICC + Pearson ---
    if os.path.isfile(FEATURE_CACHE):
        print(f"Loading retained features from cache: {FEATURE_CACHE}")
        with open(FEATURE_CACHE) as f:
            retained_cols = [line.strip() for line in f if line.strip()]
        retained_cols = [c for c in retained_cols if c in features.columns]
        features = features[retained_cols]
        print(f"  {len(retained_cols)} features loaded from cache.")
    else:
        if ICC_CSV is not None and os.path.isfile(ICC_CSV):
            print("Applying ICC feature filtering...")
            features = icc_filter(features, ICC_CSV)
        else:
            print("ICC CSV not provided; skipping ICC filtering step.")

        print("Running Pearson redundancy reduction (threshold=0.75) on raw features...")
        features = pearson_redundancy_reduction(features, threshold=0.75)

        with open(FEATURE_CACHE, "w") as f:
            for col in features.columns:
                f.write(col + "\n")
        print(f"  Retained feature names cached to {FEATURE_CACHE}")

    # Log transform and batch correction are performed inside each CV fold.

    meta = meta.reset_index(drop=True)
    features = features.reset_index(drop=True)

    mask_valid = ~meta["patient_id"].isin(EXCLUDED_IDS)

    # -----------------------------------------------------------------------
    # Task 1: Tissue type (non-neoplastic N=0 vs neoplastic Tu=1)
    # -----------------------------------------------------------------------
    mask = mask_valid & meta["tissue"].isin(["N", "Tu"])
    X_t1 = features.loc[mask]
    y_t1 = (meta.loc[mask, "tissue"] == "Tu").astype(int).values
    g_t1 = meta.loc[mask, "patient_id"].values
    b_t1 = meta.loc[mask, "batch"].values

    run_task("tissue_type", X_t1, y_t1, g_t1, b_t1,
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
    b_t2 = meta.loc[mask, "batch"].values

    run_task("tumor_type_ftn_vs_ptc", X_t2, y_t2, g_t2, b_t2,
             class_labels={0: "PTC", 1: "FTN"})

    # -----------------------------------------------------------------------
    # Task 3: BRAF V600E status (0=negative, 1=positive)
    # Include PTC(0), PDTC(2), Oncocytic(3), FVPTC(4) only.
    # FTN(1), unknown(-1), and NaN are excluded via explicit include list.
    # -----------------------------------------------------------------------
    mask = (
        mask_valid
        & (meta["tissue"] == "Tu")
        & meta["diagnosis_class"].isin([0, 2, 3, 4])
        & meta["BRAF p/n"].notna()
    )
    X_t3 = features.loc[mask]
    y_t3 = meta.loc[mask, "BRAF p/n"].astype(int).values
    g_t3 = meta.loc[mask, "patient_id"].values
    b_t3 = meta.loc[mask, "batch"].values

    run_task("braf_v600e", X_t3, y_t3, g_t3, b_t3,
             class_labels={0: "BRAF wild-type", 1: "BRAF mutant"})

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
    b_t4 = meta.loc[mask, "batch"].values

    run_task("relapse", X_t4, y_t4, g_t4, b_t4,
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
    # b_t5 = meta.loc[mask, "batch"].values
    # run_task("tert_mutation", X_t5, y_t5, g_t5, b_t5,
    #          class_labels={0: "TERT-wildtype", 1: "TERT-mutated"},
    #          stratified=True, threshold=0.2)

    print(f"\nDone. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
