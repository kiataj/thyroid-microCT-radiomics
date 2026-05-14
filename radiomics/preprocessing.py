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
# ComBat batch correction (global)
# ---------------------------------------------------------------------------

def combat_correct(features: pd.DataFrame, batches: pd.Series) -> pd.DataFrame:
    from combat.pycombat import pycombat
    batch_codes = pd.Series(pd.factorize(batches.astype(str))[0])
    corrected   = pycombat(features.T, batch_codes).T
    corrected.columns = features.columns
    corrected.index   = features.index
    return corrected


# ---------------------------------------------------------------------------
# ICC-based feature filtering
# ---------------------------------------------------------------------------

def icc_filter(features: pd.DataFrame, repro_csv: str,
               threshold: float = 0.75, p_threshold: float = 0.05) -> pd.DataFrame:
    from pingouin import intraclass_corr
    from tqdm import tqdm
    from sklearn.feature_selection import VarianceThreshold as VT
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(repro_csv, low_memory=False)
    df = df[df.columns.drop(list(df.filter(regex="diagnos")))]
    df = df[df.columns.drop(list(df.filter(regex="Unnamed")))]

    meta     = df[["TMA", "Grid", "x", "y"]]
    feat_cols_repro = [c for c in df.columns if c not in ["TMA", "Grid", "x", "y"]]
    rad      = df[feat_cols_repro].copy()

    # VarianceThreshold
    sel  = VT(threshold=0.0)
    arr  = sel.fit_transform(rad)
    rad  = pd.DataFrame(arr, index=rad.index, columns=rad.columns[sel.get_support()])

    # Sign-log transform
    rad = np.sign(rad) * np.log1p(np.abs(rad))

    # Split H64 / V64 and pair by (Grid, x, y)
    feat_cols = [c for c in features.columns if c in rad.columns]
    keys      = ["Grid", "x", "y"]

    h_meta = meta[meta["TMA"] == "H64"][keys].reset_index(drop=True)
    v_meta = meta[meta["TMA"] == "V64"][keys].reset_index(drop=True)
    h_rad  = rad.loc[meta["TMA"] == "H64", feat_cols].reset_index(drop=True)
    v_rad  = rad.loc[meta["TMA"] == "V64", feat_cols].reset_index(drop=True)
    h_meta["_idx"] = range(len(h_meta))
    v_meta["_idx"] = range(len(v_meta))
    pairs  = h_meta.merge(v_meta, on=keys, suffixes=("_h", "_v"))
    h64    = h_rad.iloc[pairs["_idx_h"].values].reset_index(drop=True)
    v64    = v_rad.iloc[pairs["_idx_v"].values].reset_index(drop=True)

    # Scale each rater independently (matches scale=True in original)
    h64 = pd.DataFrame(StandardScaler().fit_transform(h64), columns=feat_cols)
    v64 = pd.DataFrame(StandardScaler().fit_transform(v64), columns=feat_cols)

    n = len(h64)
    h64.insert(0, "rater", 1)
    h64.insert(1, "subject", range(1, n + 1))
    v64.insert(0, "rater", 2)
    v64.insert(1, "subject", range(1, n + 1))
    stacked   = pd.concat([h64, v64], ignore_index=True).dropna(axis="columns")
    available = [c for c in feat_cols if c in stacked.columns]

    reliable = []
    for col in tqdm(available, desc="ICC"):
        stats   = intraclass_corr(data=stacked, targets="subject", raters="rater",
                                   ratings=col, nan_policy="omit")
        icc_val = stats["ICC"].iloc[2]
        p_val   = stats["pval"].iloc[2]
        if icc_val > threshold and p_val < p_threshold:
            reliable.append(col)

    print(f"ICC: {len(reliable)} / {len(available)} features retained")
    return features[[c for c in reliable if c in features.columns]]


# ---------------------------------------------------------------------------
# Pearson redundancy reduction
# ---------------------------------------------------------------------------

def pearson_redundancy_reduction(features: pd.DataFrame,
                                  threshold: float = 0.75,
                                  p_threshold: float = 0.05) -> pd.DataFrame:
    import itertools
    from random import Random
    from scipy.stats import pearsonr
    from collections import Counter

    cols     = features.columns.tolist()
    keep     = np.ones(len(cols), dtype=bool)
    rng      = Random(42)

    for col1, col2 in itertools.combinations(cols, 2):
        i1, i2 = cols.index(col1), cols.index(col2)
        if not keep[i1] or not keep[i2]:
            continue
        r, p = pearsonr(features[col1], features[col2])
        if r > threshold and p < p_threshold:
            keep[rng.choice([i1, i2])] = False

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
