import numpy as np
import pandas as pd
from pathlib import Path

from .config import ICC_CSV, FEATURE_CACHE, RESULTS_DIR


# ---------------------------------------------------------------------------
# Sign-preserving log transform
# ---------------------------------------------------------------------------

def sign_log_transform_arr(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


# ---------------------------------------------------------------------------
# Batch correction
# ---------------------------------------------------------------------------

class BatchCorrector:
    """Per-batch mean/variance normalisation with sklearn-style fit/transform API."""

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


# ---------------------------------------------------------------------------
# ICC-based feature filtering
# ---------------------------------------------------------------------------

def icc_filter(features: pd.DataFrame, repro_csv: str,
               threshold: float = 0.75, p_threshold: float = 0.05) -> pd.DataFrame:
    from pingouin import intraclass_corr
    from tqdm import tqdm

    repro  = pd.read_csv(repro_csv, low_memory=False)
    scan1  = repro[repro["TMA"] == "H64"].copy()
    scan2  = repro[repro["TMA"] == "V64"].copy()
    merged = scan1.merge(scan2, on=["Grid", "x", "y"], suffixes=("_s1", "_s2"))

    feat_cols = [c for c in features.columns if c in repro.columns]
    reliable  = []

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
# Pearson redundancy reduction
# ---------------------------------------------------------------------------

def pearson_redundancy_reduction(features: pd.DataFrame,
                                  threshold: float = 0.75) -> pd.DataFrame:
    from collections import Counter

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
    print(f"Pearson redundancy: {len(retained)} / {len(cols)} features retained")

    retained_path = RESULTS_DIR / "retained_features.txt"
    prefix_counts = Counter(col.split("_")[0] for col in retained)
    with open(retained_path, "w") as f:
        f.write(f"Total retained: {len(retained)}\n\nCount by filter class:\n")
        for prefix, count in sorted(prefix_counts.items()):
            f.write(f"  {prefix}: {count}\n")
        f.write("\nFull feature list:\n")
        for col in retained:
            f.write(f"  {col}\n")

    return features[retained]


# ---------------------------------------------------------------------------
# Feature selector (ICC + Pearson with caching)
# ---------------------------------------------------------------------------

class FeatureSelector:
    """Applies ICC filtering and Pearson redundancy reduction with result caching."""

    def __init__(self, icc_csv=None, pearson_threshold: float = 0.75,
                 cache_path=None):
        self.icc_csv           = str(icc_csv or ICC_CSV)
        self.pearson_threshold = pearson_threshold
        self.cache_path        = Path(cache_path or FEATURE_CACHE)

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.cache_path.exists():
            print(f"Loading retained features from cache: {self.cache_path}")
            retained = [l.strip() for l in self.cache_path.read_text().splitlines() if l.strip()]
            retained = [c for c in retained if c in features.columns]
            print(f"  {len(retained)} features loaded from cache.")
            return features[retained]

        if Path(self.icc_csv).exists():
            print("Applying ICC feature filtering...")
            features = icc_filter(features, self.icc_csv)
        else:
            print("ICC CSV not found; skipping ICC filtering.")

        print(f"Running Pearson redundancy reduction (threshold={self.pearson_threshold})...")
        features = pearson_redundancy_reduction(features, self.pearson_threshold)

        self.cache_path.parent.mkdir(exist_ok=True)
        self.cache_path.write_text("\n".join(features.columns))
        print(f"  Retained feature names cached to {self.cache_path}")

        return features
