import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler
from skorch.dataset import ValidSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .config import TaskConfig, RESULTS_DIR
from .cv import StratifiedGroupKFold
from .evaluation import compute_fold_metrics, bootstrap_ci, save_probability_data
from .explainability import compute_shap_mlp_and_save
from .preprocessing import BatchCorrector, sign_log_transform_arr


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, dropout_rate=0.4):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2     = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


class TaskRunner:
    def __init__(self, results_dir=None):
        self.results_dir = str(results_dir or RESULTS_DIR)

    def run(self, config: TaskConfig, features: pd.DataFrame,
            labels: np.ndarray, patient_ids: np.ndarray,
            batches: np.ndarray) -> dict:
        task_name = config.name
        print(f"\n{'='*60}\nTask: {task_name}\n{'='*60}")

        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            lbl = config.class_labels.get(int(u), str(u))
            print(f"  Class {u} ({lbl}): {c} samples")

        task_dir = os.path.join(self.results_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        device         = "cuda" if torch.cuda.is_available() else "cpu"
        X_raw          = np.nan_to_num(features.values.astype(float),
                                       nan=0.0, posinf=0.0, neginf=0.0)
        y              = labels.astype(np.int64)
        feat_names_raw = features.columns.tolist()
        output_dim     = len(unique)

        splitter = (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                    if config.stratified else GroupKFold(n_splits=5))

        metrics_rows = []
        probs_list   = []
        targets_list = []
        roc_curves   = []
        last: dict   = {}

        for fold, (train_idx, val_idx) in enumerate(splitter.split(X_raw, y, patient_ids)):
            y_tr, y_va = y[train_idx], y[val_idx]

            X_tr_log = sign_log_transform_arr(X_raw[train_idx])
            X_va_log = sign_log_transform_arr(X_raw[val_idx])

            vt = VarianceThreshold(threshold=0.0)
            X_tr_vt   = vt.fit_transform(X_tr_log)
            X_va_vt   = vt.transform(X_va_log)
            feat_names = [feat_names_raw[i] for i in vt.get_support(indices=True)]

            bc = BatchCorrector()
            X_tr = bc.fit_transform(X_tr_vt, batches[train_idx]).astype(np.float32)
            X_va = bc.transform(X_va_vt, batches[val_idx]).astype(np.float32)

            cw = config.class_weights
            criterion_weights = (torch.tensor(cw, dtype=torch.float32).to(device)
                                 if cw is not None else None)

            input_dim = X_tr.shape[1]
            net = NeuralNetClassifier(
                module=MLP,
                module__input_dim=input_dim,
                module__hidden_dim=config.hidden_dim,
                module__output_dim=output_dim,
                module__dropout_rate=config.dropout_rate,
                max_epochs=config.num_epochs,
                batch_size=config.batch_size,
                optimizer=optim.Adam,
                optimizer__lr=1e-3,
                optimizer__weight_decay=1e-2,
                criterion=nn.CrossEntropyLoss,
                criterion__weight=criterion_weights,
                device=device,
                callbacks=[
                    ("lr_scheduler", LRScheduler(policy="ExponentialLR", gamma=0.9)),
                ],
                train_split=ValidSplit(0.1, stratified=True),
                verbose=0,
            )
            scaler  = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr).astype(np.float32)
            X_va_sc = scaler.transform(X_va).astype(np.float32)
            net.fit(X_tr_sc, y_tr)

            vp    = net.predict_proba(X_va_sc)[:, 1]
            vp_tr = net.predict_proba(X_tr_sc)[:, 1]
            fpr, tpr, _ = roc_curve(y_va, vp)
            roc_curves.append((fpr, tpr))
            probs_list.append(vp)
            targets_list.append(y_va)
            row = compute_fold_metrics(y_va, vp,
                                       y_true_train=y_tr, y_prob_train=vp_tr,
                                       threshold=config.threshold, fold=fold + 1)
            metrics_rows.append(row)
            print(f"  Fold {fold+1}: train_AUC={row['train_auc']:.3f}  val_AUC={row['val_auc']:.3f}  gap={row['auc_gap']:.3f}")

            last = dict(net=net, scaler=scaler, vt=vt, bc=bc, feat_names=feat_names,
                        val_idx=val_idx, X_tr_sc=X_tr_sc, X_va_sc=X_va_sc, device=device)

        self._save_results("mlp", task_name, task_dir, metrics_rows,
                           probs_list, targets_list, roc_curves,
                           config.class_labels, config.threshold)

        if last:
            mlp_model = last["net"].module_
            rng    = np.random.RandomState(42)
            bg_idx = rng.choice(len(last["X_tr_sc"]), size=min(50, len(last["X_tr_sc"])), replace=False)
            compute_shap_mlp_and_save(task_name, mlp_model, last["device"],
                                      last["X_tr_sc"][bg_idx], last["X_va_sc"],
                                      y[last["val_idx"]], last["feat_names"],
                                      out_dir=task_dir)

        return {"feature_names": last.get("feat_names", [])}

    def _save_results(self, tag, task_name, task_dir, metrics_rows,
                      probs_list, targets_list, roc_curves,
                      class_labels, threshold):
        df = pd.DataFrame(metrics_rows)
        m  = df.mean(numeric_only=True)
        s  = df.std(numeric_only=True)
        print(f"\n  [{tag}] train AUC: {m['train_auc']:.3f} ± {s['train_auc']:.3f}  "
              f"val AUC: {m['val_auc']:.3f} ± {s['val_auc']:.3f}  "
              f"mean gap: {m['auc_gap']:.3f}")
        df.to_csv(os.path.join(task_dir, f"{task_name}_{tag}_metrics.csv"), index=False)

        probs_cat   = np.concatenate(probs_list)
        targets_cat = np.concatenate(targets_list)

        ci_dict = bootstrap_ci(probs_cat, targets_cat, threshold=threshold)
        print(f"  [{tag}] AUC 95% CI: [{ci_dict['auc'][0]:.3f}, {ci_dict['auc'][1]:.3f}]")
        pd.DataFrame([{"metric": k, "ci_lower": v[0], "ci_upper": v[1]}
                      for k, v in ci_dict.items()]).to_csv(
            os.path.join(task_dir, f"{task_name}_{tag}_bootstrap_ci.csv"), index=False)

        save_probability_data(f"{task_name}_{tag}", probs_cat, targets_cat,
                              class_labels, out_dir=task_dir)

        fpr_grid = np.linspace(0, 1, 100)
        roc_rows = []
        for fold_i, (fpr, tpr) in enumerate(roc_curves):
            for f, t in zip(fpr_grid, np.interp(fpr_grid, fpr, tpr)):
                roc_rows.append({"fold": fold_i + 1, "fpr": f, "tpr": t})
        pd.DataFrame(roc_rows).to_csv(
            os.path.join(task_dir, f"{task_name}_{tag}_roc_curves.csv"), index=False)
