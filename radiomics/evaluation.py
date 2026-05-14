import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def compute_fold_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                         y_true_train: np.ndarray = None, y_prob_train: np.ndarray = None,
                         threshold: float = 0.5, fold: int = None) -> dict:
    preds = (y_prob >= threshold).astype(int)
    row = {
        "val_auc":       roc_auc_score(y_true, y_prob),
        "val_auprc":     average_precision_score(y_true, y_prob),
        "val_accuracy":  accuracy_score(y_true, preds),
        "val_f1":        f1_score(y_true, preds, average="weighted", zero_division=0),
        "val_precision": precision_score(y_true, preds, average="weighted", zero_division=0),
        "val_recall":    recall_score(y_true, preds, average="weighted", zero_division=0),
    }
    if y_true_train is not None and y_prob_train is not None:
        tr_preds = (y_prob_train >= threshold).astype(int)
        row["train_auc"]       = roc_auc_score(y_true_train, y_prob_train)
        row["train_auprc"]     = average_precision_score(y_true_train, y_prob_train)
        row["train_accuracy"]  = accuracy_score(y_true_train, tr_preds)
        row["train_f1"]        = f1_score(y_true_train, tr_preds, average="weighted", zero_division=0)
        row["train_precision"] = precision_score(y_true_train, tr_preds, average="weighted", zero_division=0)
        row["train_recall"]    = recall_score(y_true_train, tr_preds, average="weighted", zero_division=0)
        row["auc_gap"]         = row["train_auc"] - row["val_auc"]
        row["auprc_gap"]       = row["train_auprc"] - row["val_auprc"]
    if fold is not None:
        row["fold"] = fold
    return row


def bootstrap_ci(probs: np.ndarray, targets: np.ndarray,
                 n_boot: int = 2000, ci: float = 0.95,
                 threshold: float = 0.5, seed: int = 42) -> dict:
    rng   = np.random.default_rng(seed)
    n     = len(targets)
    stats: dict[str, list] = {
        "auc": [], "auprc": [], "accuracy": [], "f1": [], "precision": [], "recall": []
    }
    for _ in range(n_boot):
        idx   = rng.integers(0, n, size=n)
        p, t  = probs[idx], targets[idx]
        if len(np.unique(t)) < 2:
            continue
        preds = (p >= threshold).astype(int)
        stats["auc"].append(roc_auc_score(t, p))
        stats["auprc"].append(average_precision_score(t, p))
        stats["accuracy"].append(accuracy_score(t, preds))
        stats["f1"].append(f1_score(t, preds, average="weighted", zero_division=0))
        stats["precision"].append(precision_score(t, preds, average="weighted", zero_division=0))
        stats["recall"].append(recall_score(t, preds, average="weighted", zero_division=0))
    alpha = (1 - ci) / 2
    return {k: (np.quantile(v, alpha), np.quantile(v, 1 - alpha))
            for k, v in stats.items() if v}


def save_probability_data(task_name: str, probs: np.ndarray,
                           targets: np.ndarray, class_labels: dict,
                           out_dir: str) -> float:
    n_bins     = 20
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    class_hist = {}
    for cls in sorted(class_labels.keys()):
        vals  = probs[targets == cls]
        counts, _ = np.histogram(vals, bins=bin_edges, density=False)
        total = counts.sum()
        class_hist[cls] = counts / total if total > 0 else counts + 1e-10

    jsd = float(jensenshannon(class_hist[0], class_hist[1], base=2) ** 2)
    print(f"  Jensen-Shannon divergence: {jsd:.4f}")

    pd.DataFrame({"prob_class1": probs, "true_label": targets}).to_csv(
        os.path.join(out_dir, f"{task_name}_val_probs.csv"), index=False)

    with open(os.path.join(out_dir, f"{task_name}_prob_meta.json"), "w") as f:
        json.dump({"jsd": jsd,
                   "class_labels": {str(k): v for k, v in class_labels.items()}}, f)
    return jsd
