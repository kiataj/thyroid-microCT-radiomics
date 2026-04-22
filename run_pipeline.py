import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from radiomics.config import TaskConfig, EXCLUDED_IDS, RESULTS_DIR, ICC_CSV, FEATURE_CACHE
from radiomics.data import DataLoader
from radiomics.models import TaskRunner
from radiomics.preprocessing import icc_filter, pearson_redundancy_reduction


def run_tert_mwu(features: pd.DataFrame, meta: pd.DataFrame, mask_valid) -> None:
    mask = (mask_valid
            & (meta["tissue"] == "Tu")
            & meta["TERT"].isin([0, 1]))
    X    = features.loc[mask].reset_index(drop=True)
    y    = meta.loc[mask, "TERT"].astype(int).values

    X = apply_pearson_global(X)

    n_mut = (y == 1).sum()
    n_wt  = (y == 0).sum()
    print(f"\nTERT Mann-Whitney U: {n_mut} mutant / {n_wt} wild-type, {X.shape[1]} features tested")

    rows = []
    for col in X.columns:
        mut_vals = X.loc[y == 1, col].values
        wt_vals  = X.loc[y == 0, col].values
        stat, p  = mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")
        r        = 1 - (2 * stat) / (n_mut * n_wt)   # rank-biserial correlation
        rows.append({"feature": col, "U_statistic": stat, "p_value": p, "effect_size_r": r})

    df = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
    df["p_value_fdr"] = p_fdr
    df["significant_fdr05"] = reject
    df = df.sort_values("p_value_fdr").reset_index(drop=True)

    out_dir = RESULTS_DIR / "tert_mwu"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "tert_mwu_results.csv", index=False)

    sig = df[df["significant_fdr05"]]
    print(f"  Significant after FDR correction (q<0.05): {len(sig)}")
    if not sig.empty:
        print(sig[["feature", "p_value", "p_value_fdr", "effect_size_r"]].to_string(index=False))
    else:
        print("  No features survive FDR correction.")
        print("  Top 10 by raw p-value:")
        print(df.head(10)[["feature", "p_value", "p_value_fdr", "effect_size_r"]].to_string(index=False))
    print(f"  Full results saved to {out_dir / 'tert_mwu_results.csv'}")


PEARSON_CACHE = RESULTS_DIR / "pearson_retained_features.txt"


def apply_pearson_global(features: pd.DataFrame) -> pd.DataFrame:
    if PEARSON_CACHE.exists():
        cols = [l.strip() for l in PEARSON_CACHE.read_text().splitlines() if l.strip()]
        cols = [c for c in cols if c in features.columns]
        print(f"Pearson cache: {len(cols)} features loaded")
        return features[cols]
    result = pearson_redundancy_reduction(features)
    PEARSON_CACHE.write_text("\n".join(result.columns))
    return result


def main():
    print("Loading data...")
    loader         = DataLoader()
    features, meta = loader.load()

    # ICC filtering once — uses reproducibility scans, independent of task cohort
    if FEATURE_CACHE.exists():
        cols = [l.strip() for l in FEATURE_CACHE.read_text().splitlines() if l.strip()]
        cols = [c for c in cols if c in features.columns]
        features = features[cols]
        print(f"ICC cache: {len(cols)} features loaded from {FEATURE_CACHE}")
    elif ICC_CSV.exists():
        print("Applying ICC feature filtering...")
        features = icc_filter(features, str(ICC_CSV))
        FEATURE_CACHE.parent.mkdir(exist_ok=True)
        FEATURE_CACHE.write_text("\n".join(features.columns))
        print(f"ICC results cached to {FEATURE_CACHE}")
    else:
        print("ICC CSV not found; skipping ICC filtering.")

    # Pearson redundancy reduction — global, once, before any task
    features = apply_pearson_global(features)

    features = features.reset_index(drop=True)
    meta     = meta.reset_index(drop=True)

    mask_valid = ~meta["patient_id"].isin(EXCLUDED_IDS)

    # Combined BRAF label: mutant if either BRAF p/n or BRAF_2nd is 1 (OR logic)
    meta["braf_label"] = meta[["BRAF p/n", "BRAF_2nd"]].max(axis=1)

    runner = TaskRunner(results_dir=RESULTS_DIR)

    # -------------------------------------------------------------------------
    # Task 1: Tissue type — non-neoplastic (N) vs neoplastic (Tu)
    # -------------------------------------------------------------------------
    cfg  = TaskConfig(name="tissue_type",
                      class_labels={0: "Non-neoplastic", 1: "Neoplastic"},
                      hidden_dim=256, num_epochs=10, batch_size=16,
                      dropout_rate=0.6, class_weights=[1, 1])
    mask = mask_valid & meta["tissue"].isin(["N", "Tu"])
    X    = features.loc[mask].reset_index(drop=True)
    y    = (meta.loc[mask, "tissue"] == "Tu").astype(int).values
    g    = meta.loc[mask, "patient_id"].values
    b    = meta.loc[mask, "batch"].values
    runner.run(cfg, X, y, g, b)

    # -------------------------------------------------------------------------
    # Task 2: Tumor type — FTN (1) vs PTC (0)
    # TMA 2406 FTC excluded (FVPTC mislabelled as FTC in ngTMA_table)
    # -------------------------------------------------------------------------
    cfg  = TaskConfig(name="tumor_type_ftn_vs_ptc",
                      class_labels={0: "PTC", 1: "FTN"},
                      hidden_dim=128, num_epochs=5, batch_size=8,
                      dropout_rate=0.2, class_weights=[1, 1])
    mask = (mask_valid
            & (meta["tissue"] == "Tu")
            & meta["diagnosis_class"].isin([0, 1])
            & ~((meta["TMA"] == "2406") & (meta["Diagnosis"] == "FTC")))
    X    = features.loc[mask].reset_index(drop=True)
    y    = meta.loc[mask, "diagnosis_class"].astype(int).values
    g    = meta.loc[mask, "patient_id"].values
    b    = meta.loc[mask, "batch"].values
    runner.run(cfg, X, y, g, b)

    # -------------------------------------------------------------------------
    # Task 3: BRAF V600E — wild-type (0) vs mutant (1)
    # PTC, PDTC, Oncocytic, FVPTC only; FTN excluded
    # -------------------------------------------------------------------------
    cfg  = TaskConfig(name="braf_v600e",
                      class_labels={0: "BRAF wild-type", 1: "BRAF mutant"},
                      hidden_dim=256, num_epochs=5, batch_size=16,
                      dropout_rate=0.3, class_weights=[1, 2])
    mask = (mask_valid
            & (meta["tissue"] == "Tu")
            & (meta["diagnosis_class"].isin([0, 2, 3]) | meta["is_fvptc"])
            & meta["braf_label"].notna())
    X    = features.loc[mask].reset_index(drop=True)
    y    = meta.loc[mask, "braf_label"].astype(int).values
    g    = meta.loc[mask, "patient_id"].values
    b    = meta.loc[mask, "batch"].values
    runner.run(cfg, X, y, g, b)


    # -------------------------------------------------------------------------
    # TERT: Mann-Whitney U with FDR correction (exploratory, small N)
    # -------------------------------------------------------------------------
    if "TERT" in meta.columns:
        run_tert_mwu(features, meta, mask_valid)

    print(f"\nDone. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
