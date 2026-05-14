"""
Assembles composite figures for each task from pre-generated panel PNGs.
Run after visualize.py has completed.

Usage:
    python make_composites.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODEL = "xgb"

# Layout definition per task.
# Each entry is a list of (filename_suffix, panel_label) tuples.
# filename_suffix is appended to "{task}_" to find the PNG.
# Set panel_label to "" to omit the letter label.
# Remove or add entries here to change the composition per task.

LAYOUTS = {
    "tissue_type": [
        ("roc",        "(a) Validation ROC Curve"),
        ("prob_dist",  "(b) Validation Probability Distribution"),
    ],
    "tumor_type_ftn_vs_ptc": [
        ("roc",        "(a) Validation ROC Curve"),
        ("shap",       "(b) SHAP Summary"),
        ("prob_dist",  "(c) Validation Probability Distribution"),
    ],
    "braf_v600e": [
        ("roc",        "(a) Validation ROC Curve"),
        ("shap",       "(b) SHAP Summary"),
        ("prob_dist",  "(c) Validation Probability Distribution"),
    ],
}


def make_composite(task_name, panels, task_dir):
    images = []
    labels = []
    for suffix, label in panels:
        path = os.path.join(task_dir, f"{task_name}_{suffix}.png")
        if not os.path.isfile(path):
            print(f"  Missing: {path} — skipping panel")
            continue
        images.append(mpimg.imread(path))
        labels.append(label)

    if not images:
        print(f"  No panels found for {task_name}, skipping.")
        return

    n = len(images)

    if n == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = list(axes.flat)
    elif n == 3:
        # Panel (a) top-left, panel (c) top-right, panel (b) full-width bottom
        fig = plt.figure(figsize=(12, 10))
        gs  = GridSpec(2, 2, figure=fig)
        ax0 = fig.add_subplot(gs[0, 0])   # (a) — top-left
        ax1 = fig.add_subplot(gs[1, :])   # (b) — bottom full-width
        ax2 = fig.add_subplot(gs[0, 1])   # (c) — top-right
        axes = [ax0, ax1, ax2]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = list(axes.flat)

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.axis("off")
        if label:
            ax.text(0.5, .99, label, transform=ax.transAxes,
                    fontsize=12, va="bottom", ha="center")

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.tight_layout(pad=0.5, h_pad=0.1)
    out_path = os.path.join(task_dir, f"composite_figure_{task_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved composite_figure_{task_name}.png")


def main():
    for task_name, panels in LAYOUTS.items():
        task_dir = os.path.join(RESULTS_DIR, task_name)
        if not os.path.isdir(task_dir):
            print(f"[{task_name}] directory not found, skipping.")
            continue
        print(f"\n[{task_name}]")
        make_composite(task_name, panels, task_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
