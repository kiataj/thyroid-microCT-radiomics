import argparse
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import VarianceThreshold
from radiomics.config import (TaskConfig, EXCLUDED_IDS, RESULTS_DIR, ICC_CSV,
                              ICC_CACHE, FEATURE_CACHE)
from radiomics.data import DataLoader
from radiomics.models import TaskRunner
from radiomics.preprocessing import (icc_filter, pearson_redundancy_reduction,
                                     sign_log_transform_arr, combat_correct)

PEARSON_THRESHOLD = 0.75


def run_fvptc_similarity(features: pd.DataFrame, meta: pd.DataFrame, mask_valid) -> None:
    import json
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from scipy.spatial.distance import jensenshannon
    from skorch import NeuralNetClassifier
    from skorch.callbacks import LRScheduler
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from radiomics.mlp import MLP

    out_dir = RESULTS_DIR / "fvptc_similarity"
    out_dir.mkdir(exist_ok=True)

    mask_train = (mask_valid
                  & (meta["tissue"] == "Tu")
                  & meta["diagnosis_class"].isin([0, 1]))
    mask_fvptc = (mask_valid
                  & (meta["tissue"] == "Tu")
                  & ((meta["diagnosis_class"] == 4) | meta["is_fvptc"]))

    X_train = features.loc[mask_train].values.astype(np.float32)
    y_train = meta.loc[mask_train, "diagnosis_class"].astype(int).values
    X_fvptc = features.loc[mask_fvptc].values.astype(np.float32)

    n_ptc   = int((y_train == 0).sum())
    n_ftn   = int((y_train == 1).sum())
    n_fvptc = len(X_fvptc)
    print(f"\nFVPTC similarity: {n_ptc} PTC + {n_ftn} FTN (train), {n_fvptc} FVPTC (test)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = NeuralNetClassifier(
        module=MLP,
        module__input_dim=X_train.shape[1],
        module__hidden_dims=(256, 128, 256),
        module__output_dim=2,
        module__dropout_rate=0.5,
        max_epochs=100,
        batch_size=32,
        optimizer=optim.Adam,
        optimizer__lr=1e-3,
        optimizer__weight_decay=1e-3,
        criterion=nn.CrossEntropyLoss,
        device=device,
        callbacks=[("lr_scheduler", LRScheduler(policy="ExponentialLR", gamma=0.9))],
        train_split=None,
        verbose=0,
    )
    clf = Pipeline([("scaler", StandardScaler()), ("net", net)])
    clf.fit(X_train, y_train)

    # class 0 = PTC, class 1 = FTN — column 0 is P(PTC)
    prob_ptc_fvptc = clf.predict_proba(X_fvptc)[:, 0]
    prob_ptc_train = clf.predict_proba(X_train)[:, 0]
    prob_ptc_ptc   = prob_ptc_train[y_train == 0]
    prob_ptc_ftn   = prob_ptc_train[y_train == 1]

    mean_p   = float(prob_ptc_fvptc.mean())
    median_p = float(np.median(prob_ptc_fvptc))
    std_p    = float(prob_ptc_fvptc.std())

    # JSD between FVPTC and each reference class (histogram, 20 bins, consistent with evaluation.py)
    bin_edges = np.linspace(0, 1, 21)

    def _hist(vals):
        counts, _ = np.histogram(vals, bins=bin_edges, density=False)
        counts = counts.astype(float)
        total = counts.sum()
        return counts / total if total > 0 else counts + 1e-10

    h_fvptc = _hist(prob_ptc_fvptc)
    h_ptc   = _hist(prob_ptc_ptc)
    h_ftn   = _hist(prob_ptc_ftn)

    jsd_vs_ptc = float(jensenshannon(h_fvptc, h_ptc, base=2) ** 2)
    jsd_vs_ftn = float(jensenshannon(h_fvptc, h_ftn, base=2) ** 2)

    print(f"  FVPTC P(PTC): mean={mean_p:.3f}  median={median_p:.3f}  std={std_p:.3f}")
    print(f"  JSD(FVPTC vs PTC)={jsd_vs_ptc:.4f}  JSD(FVPTC vs FTN)={jsd_vs_ftn:.4f}")
    closer = "PTC" if jsd_vs_ptc < jsd_vs_ftn else "FTN"
    print(f"  FVPTC is distributionally closer to {closer}")

    pd.DataFrame({"prob_ptc": prob_ptc_fvptc}).to_csv(
        out_dir / "fvptc_similarity_probabilities.csv", index=False)

    summary = {
        "n_fvptc": n_fvptc, "n_ptc_train": n_ptc, "n_ftn_train": n_ftn,
        "mean_prob_ptc": mean_p, "median_prob_ptc": median_p, "std_prob_ptc": std_p,
        "jsd_fvptc_vs_ptc": jsd_vs_ptc,
        "jsd_fvptc_vs_ftn": jsd_vs_ftn,
        "closer_to": closer,
    }
    (out_dir / "fvptc_similarity_summary.json").write_text(json.dumps(summary, indent=2))

    x_grid = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(6, 4))
    for vals, label, color in [
        (prob_ptc_ptc,   "PTC (train)",  "#2166ac"),
        (prob_ptc_ftn,   "FTN (train)",  "#d73027"),
        (prob_ptc_fvptc, "FVPTC (test)", "#4dac26"),
    ]:
        kde = gaussian_kde(vals, bw_method="scott")
        ax.plot(x_grid, kde(x_grid), label=label, color=color, lw=2)
        ax.axvline(float(vals.mean()), color=color, lw=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Predicted probability of PTC")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fvptc_similarity_kde.png", dpi=150)
    plt.close(fig)
    print(f"  Results saved to {out_dir}")


def run_tert_mwu(features: pd.DataFrame, meta: pd.DataFrame, mask_valid) -> None:
    mask = (mask_valid
            & (meta["tissue"] == "Tu")
            & meta["TERT"].isin([0, 1]))
    X    = features.loc[mask].reset_index(drop=True)
    y    = meta.loc[mask, "TERT"].astype(int).values

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
    df = df.sort_values("p_value").reset_index(drop=True)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-1", action="store_true")
    parser.add_argument("--task-2", action="store_true")
    parser.add_argument("--task-3", action="store_true")
    parser.add_argument("--tert",   action="store_true")
    parser.add_argument("--fvptc",  action="store_true")
    parser.add_argument("--shap",   action="store_true")
    args = parser.parse_args()

    selected = [args.task_1, args.task_2, args.task_3, args.tert, args.fvptc]
    run_all  = not any(selected)
    main_only = not run_all

    print("Loading data...")
    loader         = DataLoader()
    features, meta = loader.load()

    # -------------------------------------------------------------------------
    # Step 1: Zero-variance filter
    # -------------------------------------------------------------------------
    vt       = VarianceThreshold(threshold=0.0)
    features = pd.DataFrame(vt.fit_transform(features.values),
                            index=features.index,
                            columns=features.columns[vt.get_support()])
    print(f"VarianceThreshold: {features.shape[1]} features retained")

    # -------------------------------------------------------------------------
    # Step 2: ICC filtering (mask only; cached)
    # -------------------------------------------------------------------------
    if ICC_CACHE.exists():
        cols = [l.strip() for l in ICC_CACHE.read_text().splitlines() if l.strip()]
        cols = [c for c in cols if c in features.columns]
        features = features[cols]
        print(f"ICC cache: {len(cols)} features loaded from {ICC_CACHE}")
    elif ICC_CSV.exists():
        print("Applying ICC feature filtering...")
        features = icc_filter(features, str(ICC_CSV))
        ICC_CACHE.parent.mkdir(exist_ok=True)
        ICC_CACHE.write_text("\n".join(features.columns))
        print(f"ICC results cached to {ICC_CACHE}")
    else:
        print("ICC CSV not found; skipping ICC filtering.")

    # -------------------------------------------------------------------------
    # Step 3a: Sign-preserving log transform
    # -------------------------------------------------------------------------
    features = pd.DataFrame(
        sign_log_transform_arr(features.values.astype(float)),
        index=features.index,
        columns=features.columns,
    )

    # -------------------------------------------------------------------------
    # Step 3b: ComBat batch correction (global, batch = TMA + Grid)
    # -------------------------------------------------------------------------
    print("Applying ComBat batch correction...")
    features = combat_correct(features, meta["batch"])

    # -------------------------------------------------------------------------
    # Step 4: Pearson redundancy reduction (cached)
    # -------------------------------------------------------------------------
    if FEATURE_CACHE.exists():
        cols = [l.strip() for l in FEATURE_CACHE.read_text().splitlines() if l.strip()]
        cols = [c for c in cols if c in features.columns]
        features = features[cols]
        print(f"Pearson cache: {len(cols)} features loaded from {FEATURE_CACHE}")
    else:
        features = pearson_redundancy_reduction(features, threshold=PEARSON_THRESHOLD)
        FEATURE_CACHE.parent.mkdir(exist_ok=True)
        FEATURE_CACHE.write_text("\n".join(features.columns))
        print(f"Pearson results cached to {FEATURE_CACHE}")

    features = features.reset_index(drop=True)
    meta     = meta.reset_index(drop=True)

    mask_valid = ~meta["patient_id"].isin(EXCLUDED_IDS)

    meta["braf_label"] = meta["BRAF p/n"]

    runner = TaskRunner(results_dir=RESULTS_DIR)

    # -------------------------------------------------------------------------
    # Task 1: Tissue type — non-neoplastic (N) vs neoplastic (Tu)
    # -------------------------------------------------------------------------
    if run_all or args.task_1:
        cfg  = TaskConfig(name="tissue_type",
                          class_labels={0: "Non-neoplastic", 1: "Neoplastic"},
                          class_weights=[1, 1],
                          hidden_dims=(256,128,256), num_epochs=100, batch_size=16,
                          learning_rate=1e-3, weight_decay=1e-3,
                          dropout_rate=0.6, gamma=0.9)
        mask = mask_valid & meta["tissue"].isin(["N", "Tu"])
        X    = features.loc[mask].reset_index(drop=True)
        y    = (meta.loc[mask, "tissue"] == "Tu").astype(int).values
        g    = meta.loc[mask, "patient_id"].values
        runner.run(cfg, X, y, g, main_only=main_only, run_shap=args.shap)

    # -------------------------------------------------------------------------
    # Task 2: Tumor type — FTN (1) vs PTC (0)
    # -------------------------------------------------------------------------
    if run_all or args.task_2:
        cfg  = TaskConfig(name="tumor_type_ftn_vs_ptc",
                          class_labels={0: "PTC", 1: "FTN"},
                          class_weights=[1, 1],
                          stratified=True,
                          hidden_dims=(256,128,256), num_epochs=100, batch_size=32,
                          learning_rate=1e-3, weight_decay=1e-3,
                          dropout_rate=0.5, gamma=0.9)
        mask = (mask_valid
                & (meta["tissue"] == "Tu")
                & meta["diagnosis_class"].isin([0, 1]))
        X    = features.loc[mask].reset_index(drop=True)
        y    = meta.loc[mask, "diagnosis_class"].astype(int).values
        g    = meta.loc[mask, "patient_id"].values
        print(f"  Unique patients: {len(np.unique(g))}, samples: {len(g)}, synthetic IDs: {(g < -1).sum()}")
        runner.run(cfg, X, y, g, main_only=main_only, run_shap=args.shap)

    # -------------------------------------------------------------------------
    # Task 3: BRAF V600E — wild-type (0) vs mutant (1)
    # PTC, PDTC, Oncocytic, FVPTC only; FTN excluded
    # -------------------------------------------------------------------------
    if run_all or args.task_3:
        cfg  = TaskConfig(name="braf_v600e",
                          class_labels={0: "BRAF V600E wild-type", 1: "BRAF V600E mutant"},
                          class_weights=[1, 2],
                          stratified=True,
                          hidden_dims=(256,128,256), num_epochs=100, batch_size=16,
                          learning_rate=1e-3, weight_decay=1e-3,
                          dropout_rate=0.5, gamma=0.9)
        mask = (mask_valid
                & (meta["tissue"] == "Tu")
                & (meta["diagnosis_class"].isin([0, 2, 3]) | meta["is_fvptc"])
                & meta["braf_label"].notna())
        X    = features.loc[mask].reset_index(drop=True)
        y    = meta.loc[mask, "braf_label"].astype(int).values
        g    = meta.loc[mask, "patient_id"].values
        runner.run(cfg, X, y, g, main_only=main_only, run_shap=args.shap)

    # -------------------------------------------------------------------------
    # TERT: Mann-Whitney U with FDR correction (exploratory, small N)
    # -------------------------------------------------------------------------
    if (run_all or args.tert) and "TERT" in meta.columns:
        run_tert_mwu(features, meta, mask_valid)

    if run_all or args.fvptc:
        run_fvptc_similarity(features, meta, mask_valid)

    print(f"\nDone. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
