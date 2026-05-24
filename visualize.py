"""
Visualization script for radiomics pipeline results.

Reads saved CSVs / JSONs from results/<task>/ and produces all plots.
Run after pipeline.py has completed.

Usage:
    python visualize.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

MODEL       = "mlp"
MODEL_LABEL = "MLP"

# Standardised panel dimensions and font sizes
PANEL_FIGSIZE  = (5, 4)
FONT_SIZE      = 10
TICK_SIZE      = 9
LEGEND_SIZE    = 9

PROB_DIST_KDE  = True    # True: KDE density curve / False: frequency histogram

plt.rcParams.update({
    "font.size":        FONT_SIZE,
    "axes.labelsize":   FONT_SIZE,
    "xtick.labelsize":  TICK_SIZE,
    "ytick.labelsize":  TICK_SIZE,
    "legend.fontsize":  LEGEND_SIZE,
})

# Global color palette
PALETTE = {
    "main":    "#1565C0",   # deep blue   — ROC curve, UMAP class 0
    "accent":  "#C62828",   # deep red    — UMAP class 1
    "class0":  "#1565C0",   # prob dist class 0
    "class1":  "#C62828",   # prob dist class 1
    "neutral": "#757575",   # fold lines, diagonal
}
CLASS_COLORS = [PALETTE["class0"], PALETTE["class1"],
                "#2E7D32", "#E65100", "#6A1B9A", "#00838F",
                "#F9A825", "#AD1457", "#0277BD", "#4E342E"]
MODEL_COLOR = PALETTE["main"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def task_dirs():
    for name in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, name)
        if os.path.isdir(path):
            yield name, path


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# UMAP scatter (model-independent)
# ---------------------------------------------------------------------------

def plot_umap(task_name, task_dir):
    emb_path  = os.path.join(task_dir, f"{task_name}_umap_embedding.csv")
    meta_path = os.path.join(task_dir, f"{task_name}_umap_meta.json")
    if not os.path.isfile(emb_path):
        return

    emb  = pd.read_csv(emb_path)
    meta = load_json(meta_path)
    class_labels = {int(k): v for k, v in meta["class_labels"].items()}
    sil  = meta["silhouette"]

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    for cls, label in class_labels.items():
        mask = emb["label"] == cls
        ax.scatter(emb.loc[mask, "umap1"], emb.loc[mask, "umap2"],
                   c=[CLASS_COLORS[cls % len(CLASS_COLORS)]], label=label,
                   alpha=0.6, s=14, linewidths=0)
    ax.set_title(f"UMAP — {task_name}\nSilhouette: {sil:.3f}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(os.path.join(task_dir, f"{task_name}_umap.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {task_name}_umap.png")


# ---------------------------------------------------------------------------
# ROC curve — RF only
# ---------------------------------------------------------------------------

def plot_roc(task_name, task_dir):
    roc_path     = os.path.join(task_dir, f"{task_name}_{MODEL}_roc_curves.csv")
    metrics_path = os.path.join(task_dir, f"{task_name}_{MODEL}_metrics.csv")
    ci_path      = os.path.join(task_dir, f"{task_name}_{MODEL}_bootstrap_ci.csv")
    if not os.path.isfile(roc_path):
        return

    roc     = pd.read_csv(roc_path)
    metrics = pd.read_csv(metrics_path)
    folds    = sorted(roc["fold"].unique())
    fpr_grid = roc[roc["fold"] == folds[0]]["fpr"].values

    tpr_matrix = np.stack([roc[roc["fold"] == f]["tpr"].values for f in folds])
    mean_tpr   = tpr_matrix.mean(axis=0)
    std_tpr    = tpr_matrix.std(axis=0)
    mean_auc   = metrics["val_auc"].mean()
    std_auc    = metrics["val_auc"].std()

    auc_label = f"AUC = {mean_auc:.2f} ± {std_auc:.2f}"
    if os.path.isfile(ci_path):
        ci = pd.read_csv(ci_path)
        row = ci[ci["metric"] == "auc"]
        if not row.empty:
            lo, hi = row.iloc[0]["ci_lower"], row.iloc[0]["ci_upper"]
            auc_label += f"\n95% CI [{lo:.2f}, {hi:.2f}]"

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    for f in folds:
        fd = roc[roc["fold"] == f]
        ax.plot(fd["fpr"], fd["tpr"], alpha=0.2, color=MODEL_COLOR, lw=1)
    ax.plot(fpr_grid, mean_tpr, color=MODEL_COLOR, lw=2, label=auc_label)
    ax.fill_between(fpr_grid,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    alpha=0.15, color=MODEL_COLOR)
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(task_dir, f"{task_name}_roc.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {task_name}_roc.png")


# ---------------------------------------------------------------------------
# Probability distribution — RF only
# ---------------------------------------------------------------------------

def plot_prob_dist(task_name, task_dir):
    probs_path = os.path.join(task_dir, f"{task_name}_{MODEL}_val_probs.csv")
    meta_path  = os.path.join(task_dir, f"{task_name}_{MODEL}_prob_meta.json")
    if not os.path.isfile(probs_path):
        return

    df   = pd.read_csv(probs_path)
    meta = load_json(meta_path)
    class_labels = {int(k): v for k, v in meta["class_labels"].items()}
    if task_name == "braf_v600e":
        class_labels = {0: "BRAF V600E wild-type", 1: "BRAF V600E mutant"}

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    if PROB_DIST_KDE:
        from scipy.stats import gaussian_kde
        x_grid = np.linspace(0, 1, 300)
        for cls, color in zip(sorted(class_labels.keys()), CLASS_COLORS):
            vals = df.loc[df["true_label"] == cls, "prob_class1"].values
            if len(vals) < 2:
                continue
            kde     = gaussian_kde(vals, bw_method=0.2)
            density = kde(x_grid)
            ax.plot(x_grid, density, color=color, lw=2, label=class_labels[cls])
            ax.fill_between(x_grid, density, alpha=0.25, color=color)
        ax.set_ylabel("Density")
    else:
        bins = np.linspace(0, 1, 21).tolist()
        for cls, color in zip(sorted(class_labels.keys()), CLASS_COLORS):
            vals = df.loc[df["true_label"] == cls, "prob_class1"].to_numpy()
            if len(vals) < 2:
                continue
            ax.hist(vals, bins=bins, color=color, alpha=0.5,
                    label=class_labels[cls], edgecolor="none")
        ax.set_ylabel("Frequency")
    ax.set_xlabel("Predicted probability")
    if task_name == "tumor_type_ftn_vs_ptc":
        ax.legend(fontsize=9, loc="upper left")
    else:
        ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(task_dir, f"{task_name}_prob_dist.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {task_name}_prob_dist.png")


# ---------------------------------------------------------------------------
# SHAP beeswarm — RF only
# ---------------------------------------------------------------------------

def plot_shap(task_name, task_dir):
    shap_path = os.path.join(task_dir, f"{task_name}_shap_values.csv")
    feat_path = os.path.join(task_dir, f"{task_name}_shap_feature_values.csv")
    top5_path = os.path.join(task_dir, f"{task_name}_shap_top5.csv")
    if not os.path.isfile(shap_path):
        return

    df_shap = pd.read_csv(shap_path)
    df_feat = pd.read_csv(feat_path)
    top5    = pd.read_csv(top5_path)

    feature_cols = top5["feature"].tolist()
    sv = df_shap[feature_cols].values
    Xv = df_feat[feature_cols].values

    # Replace feature names with F1, F2, ... and save the mapping alongside the plot
    short_names = [f"F{i+1}" for i in range(len(feature_cols))]
    mapping = pd.DataFrame({"label": short_names, "feature": feature_cols})
    mapping.to_csv(os.path.join(task_dir, f"{task_name}_shap_feature_labels.csv"), index=False)

    shap.summary_plot(sv, Xv, feature_names=short_names,
                      show=False, plot_type="dot", color_bar=True,
                      plot_size=(9, 3))
    for ax in plt.gcf().axes:
        ax.tick_params(labelsize=TICK_SIZE)
        ax.xaxis.label.set_size(FONT_SIZE)
        ax.yaxis.label.set_size(FONT_SIZE)
    plt.gca().set_xlabel("SHAP value")
    plt.gcf().axes[0].set_title("")
    if task_name == "braf_v600e":
        plt.gca().set_xlim(-1, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(task_dir, f"{task_name}_shap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {task_name}_shap.png")
    print(f"  Feature labels:")
    for _, r in mapping.iterrows():
        print(f"    {r['label']}: {r['feature']}")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(task_name, task_dir):
    cm_path = os.path.join(task_dir, f"{task_name}_{MODEL}_confusion_matrix.csv")
    if not os.path.isfile(cm_path):
        return

    cm  = pd.read_csv(cm_path, index_col=0)
    vals = cm.values.astype(float)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(vals, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(cm.columns, fontsize=8)
    ax.set_yticklabels(cm.index, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    thresh = vals.max() / 2
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, int(vals[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if vals[i, j] > thresh else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {task_name}\n({MODEL_LABEL})")
    fig.tight_layout()
    fig.savefig(os.path.join(task_dir, f"{task_name}_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {task_name}_confusion_matrix.png")


# ---------------------------------------------------------------------------
# Model comparison — XGBoost vs Logistic Regression
# ---------------------------------------------------------------------------

def plot_model_comparison(task_name, task_dir):
    tags   = ["mlp", "lr", "svm"]
    labels = {"mlp": "MLP", "lr": "Logistic Reg.", "svm": "SVM (RBF)"}
    colors = [MODEL_COLOR, PALETTE["accent"], PALETTE["neutral"]]

    records = {}
    for tag in tags:
        mpath  = os.path.join(task_dir, f"{task_name}_{tag}_metrics.csv")
        cipath = os.path.join(task_dir, f"{task_name}_{tag}_bootstrap_ci.csv")
        if not os.path.isfile(mpath):
            continue
        metrics = pd.read_csv(mpath)
        ci      = pd.read_csv(cipath) if os.path.isfile(cipath) else pd.DataFrame()
        records[tag] = (metrics, ci)

    if len(records) < 2:
        return

    metrics_to_show = [("val_auc", "AUC"), ("val_auprc", "AUPRC")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    x = np.arange(len(tags))

    for ax, (col, ylabel) in zip(axes, metrics_to_show):
        for i, (tag, color) in enumerate(zip(tags, colors)):
            if tag not in records:
                continue
            metrics, ci = records[tag]
            if col not in metrics.columns:
                continue
            mean = metrics[col].mean()
            std  = metrics[col].std()

            ci_row = ci[ci["metric"] == col.replace("val_", "")] if not ci.empty else pd.DataFrame()
            if not ci_row.empty:
                lo = ci_row.iloc[0]["ci_lower"]
                hi = ci_row.iloc[0]["ci_upper"]
                yerr = np.array([[mean - lo], [hi - mean]])
            else:
                yerr = [[std], [std]]

            ax.bar(i, mean, color=color, alpha=0.8, label=labels[tag])
            ax.errorbar(i, mean, yerr=yerr, fmt="none", color="black", capsize=5, lw=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels([labels[t] for t in tags], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.5, 0.9)
        ax.set_title(f"{ylabel} — {task_name}")

    fig.tight_layout()
    fig.savefig(os.path.join(task_dir, f"{task_name}_model_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {task_name}_model_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for task_name, task_dir in task_dirs():
        print(f"\n[{task_name}]")
        plot_umap(task_name, task_dir)
        plot_roc(task_name, task_dir)
        plot_prob_dist(task_name, task_dir)
        plot_shap(task_name, task_dir)
        plot_confusion_matrix(task_name, task_dir)
        plot_model_comparison(task_name, task_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
