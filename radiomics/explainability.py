import os

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F


def compute_shap_and_save(task_name: str, model,
                           X_explain: np.ndarray, y_explain: np.ndarray,
                           feature_names: list, out_dir: str, top_n: int = 5) -> None:
    print(f"  Running SHAP for {task_name}...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_explain)

    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv = shap_vals[:, :, 1]
    else:
        sv = shap_vals

    _save_shap(task_name, sv, X_explain, y_explain, feature_names, out_dir, top_n)


def compute_shap_mlp_and_save(task_name: str, model, device: str,
                               X_bg: np.ndarray, X_explain: np.ndarray,
                               y_explain: np.ndarray, feature_names: list,
                               out_dir: str, top_n: int = 5) -> None:
    print(f"  Running SHAP (MLP) for {task_name}...")
    model.eval()

    def predict_fn(x):
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).to(device)
            return F.softmax(model(t), dim=1).cpu().numpy()

    explainer = shap.KernelExplainer(predict_fn, X_bg)
    shap_vals = explainer.shap_values(X_explain, nsamples=100)

    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv = shap_vals[:, :, 1]
    else:
        sv = shap_vals

    _save_shap(task_name, sv, X_explain, y_explain, feature_names, out_dir, top_n)


def compute_shap_sklearn_and_save(task_name: str, clf_pipeline,
                                   X_bg: np.ndarray, X_explain: np.ndarray,
                                   y_explain: np.ndarray,
                                   feature_names: list, out_dir: str, top_n: int = 5) -> None:
    print(f"  Running SHAP for {task_name}...")
    scaler = clf_pipeline.named_steps["scaler"]
    module = clf_pipeline.named_steps["net"].module_
    device = next(module.parameters()).device

    module.eval()
    X_bg_sc  = scaler.transform(X_bg).astype(np.float32)
    X_exp_sc = scaler.transform(X_explain).astype(np.float32)
    clip_lo  = X_bg_sc.min(axis=0) - 3.0
    clip_hi  = X_bg_sc.max(axis=0) + 3.0
    X_exp_sc = np.clip(X_exp_sc, clip_lo, clip_hi)
    bg_t  = torch.tensor(X_bg_sc,  device=device)
    exp_t = torch.tensor(X_exp_sc, device=device)

    explainer = shap.GradientExplainer(module, bg_t)
    shap_vals = explainer.shap_values(exp_t)

    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1])
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv = shap_vals[:, :, 1]
    else:
        sv = np.array(shap_vals)

    _save_shap(task_name, sv, X_explain, y_explain, feature_names, out_dir, top_n)


def _save_shap(task_name, sv, X_explain, y_explain, feature_names, out_dir, top_n):
    mean_abs  = np.abs(sv).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]

    print(f"  Top {top_n} SHAP features:")
    for name, idx in zip(top_names, top_idx):
        print(f"    {name}  (mean |SHAP| = {mean_abs[idx]:.4f})")

    pd.DataFrame(sv[:, top_idx], columns=top_names).assign(
        true_label=y_explain
    ).to_csv(os.path.join(out_dir, f"{task_name}_shap_values.csv"), index=False)

    pd.DataFrame(X_explain[:, top_idx], columns=top_names).to_csv(
        os.path.join(out_dir, f"{task_name}_shap_feature_values.csv"), index=False)

    pd.DataFrame({
        "rank":          range(1, top_n + 1),
        "feature":       top_names,
        "mean_abs_shap": mean_abs[top_idx],
    }).to_csv(os.path.join(out_dir, f"{task_name}_shap_top{top_n}.csv"), index=False)
