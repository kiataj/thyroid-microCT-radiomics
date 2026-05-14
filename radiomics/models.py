import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler
from skorch.dataset import ValidSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from .config import TaskConfig, RESULTS_DIR
from .cv import StratifiedGroupKFold
from .evaluation import compute_fold_metrics, bootstrap_ci, save_probability_data
from .explainability import compute_shap_sklearn_and_save
from .mlp import MLP
from .preprocessing import BatchCorrector


class TaskRunner:
    def __init__(self, results_dir=None):
        self.results_dir = str(results_dir or RESULTS_DIR)

    def run(self, config: TaskConfig, features: pd.DataFrame,
            labels: np.ndarray, patient_ids: np.ndarray,
            batches: np.ndarray, main_only: bool = False) -> dict:
        task_name = config.name
        print(f"\n{'='*60}\nTask: {task_name}\n{'='*60}")

        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            lbl = config.class_labels.get(int(u), str(u))
            print(f"  Class {u} ({lbl}): {c} samples")

        task_dir = os.path.join(self.results_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        y              = labels.astype(np.int64)
        feat_names_raw = features.columns.tolist()
        X_raw = np.nan_to_num(features.values.astype(float),
                              nan=0.0, posinf=0.0, neginf=0.0)

        splitter = (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                    if config.stratified else GroupKFold(n_splits=5))
        fold_splits = list(splitter.split(X_raw, y, patient_ids))

        cw           = config.class_weights
        class_weight = ({i: w for i, w in enumerate(cw)} if cw is not None else None)

        def lr_factory():
            base = LogisticRegression(
                max_iter     = 1000,
                class_weight = class_weight,
                random_state = 42,
                solver       = "lbfgs",
            )
            return GridSearchCV(base, {"C": [0.01, 0.1, 1.0, 10.0]},
                                cv=3, scoring="roc_auc", refit=True, n_jobs=-1)

        def svm_factory():
            base = SVC(
                kernel       = "rbf",
                probability  = True,
                class_weight = class_weight,
                random_state = 42,
            )
            return GridSearchCV(base, {"C": [0.1, 1.0, 10.0]},
                                cv=3, scoring="roc_auc", refit=True, n_jobs=-1)

        device     = "cuda" if torch.cuda.is_available() else "cpu"
        output_dim = len(np.unique(y))

        def mlp_factory():
            net = NeuralNetClassifier(
                module              = MLP,
                module__input_dim   = X_raw.shape[1],
                module__hidden_dim  = config.hidden_dim,
                module__output_dim  = output_dim,
                module__dropout_rate= config.dropout_rate,
                max_epochs          = config.num_epochs,
                batch_size          = config.batch_size,
                optimizer           = optim.Adam,
                optimizer__lr       = config.learning_rate,
                optimizer__weight_decay = config.weight_decay,
                criterion           = nn.CrossEntropyLoss,
                device              = device,
                callbacks           = [
                    ("lr_scheduler", LRScheduler(policy="ExponentialLR", gamma=config.gamma)),
                ],
                train_split         = ValidSplit(0.1, stratified=True),
                verbose             = 0,
            )
            return Pipeline([("scaler", StandardScaler()), ("net", net)])

        mlp_last = self._run_cv("mlp", mlp_factory, fold_splits, X_raw, y, batches,
                                config, feat_names_raw, task_dir, task_name, scale=False)

        if not main_only:
            self._run_cv("lr", lr_factory, fold_splits, X_raw, y, batches,
                         config, feat_names_raw, task_dir, task_name, scale=True)

            self._run_cv("svm", svm_factory, fold_splits, X_raw, y, batches,
                         config, feat_names_raw, task_dir, task_name, scale=True)

        if mlp_last:
            X_oof  = mlp_last["X_oof"]
            bg_idx = np.random.default_rng(42).choice(len(X_oof), min(100, len(X_oof)), replace=False)
            compute_shap_sklearn_and_save(task_name, mlp_last["clf"],
                                          X_oof[bg_idx], X_oof,
                                          y[mlp_last["oof_val_idx"]],
                                          mlp_last["feat_names"], out_dir=task_dir)

        return {"feature_names": mlp_last.get("feat_names", [])}

    def _run_cv(self, tag, clf_factory, fold_splits, X_raw, y, batches,
                config, feat_names, task_dir, task_name, scale=False):
        metrics_rows  = []
        probs_list    = []
        targets_list  = []
        roc_curves    = []
        all_X_va      = []
        all_val_idx   = []
        last: dict    = {}

        for fold, (train_idx, val_idx) in enumerate(fold_splits):
            y_tr, y_va = y[train_idx], y[val_idx]
            bc   = BatchCorrector()
            X_tr = bc.fit_transform(X_raw[train_idx], batches[train_idx]).astype(np.float32)
            X_va = bc.transform(X_raw[val_idx], batches[val_idx]).astype(np.float32)

            if scale:
                sc   = StandardScaler()
                X_tr = sc.fit_transform(X_tr).astype(np.float32)
                X_va = sc.transform(X_va).astype(np.float32)

            clf = clf_factory()
            clf.fit(X_tr, y_tr)

            vp    = clf.predict_proba(X_va)[:, 1]
            vp_tr = clf.predict_proba(X_tr)[:, 1]
            fpr, tpr, _ = roc_curve(y_va, vp)
            roc_curves.append((fpr, tpr))
            probs_list.append(vp)
            targets_list.append(y_va)
            row = compute_fold_metrics(y_va, vp,
                                       y_true_train=y_tr, y_prob_train=vp_tr,
                                       threshold=config.threshold, fold=fold + 1)
            metrics_rows.append(row)
            print(f"  [{tag}] Fold {fold+1}: train_AUC={row['train_auc']:.3f}  val_AUC={row['val_auc']:.3f}  val_AUPRC={row['val_auprc']:.3f}")

            all_X_va.append(X_va)
            all_val_idx.append(val_idx)
            last = dict(clf=clf, feat_names=feat_names)

        self._save_results(tag, task_name, task_dir, metrics_rows,
                           probs_list, targets_list, roc_curves,
                           config.class_labels, config.threshold)
        if last and all_X_va:
            last["X_oof"]      = np.concatenate(all_X_va, axis=0)
            last["oof_val_idx"] = np.concatenate(all_val_idx, axis=0)
        return last

    def _save_results(self, tag, task_name, task_dir, metrics_rows,
                      probs_list, targets_list, roc_curves,
                      class_labels, threshold):
        df = pd.DataFrame(metrics_rows)
        m  = df.mean(numeric_only=True)
        s  = df.std(numeric_only=True)
        print(f"\n  [{tag}] train AUC: {m['train_auc']:.3f} ± {s['train_auc']:.3f}  "
              f"val AUC: {m['val_auc']:.3f} ± {s['val_auc']:.3f}  "
              f"val AUPRC: {m['val_auprc']:.3f} ± {s['val_auprc']:.3f}")
        df.to_csv(os.path.join(task_dir, f"{task_name}_{tag}_metrics.csv"), index=False)

        probs_cat   = np.concatenate(probs_list)
        targets_cat = np.concatenate(targets_list)

        ci_dict = bootstrap_ci(probs_cat, targets_cat, threshold=threshold)
        print(f"  [{tag}] AUC 95% CI: [{ci_dict['auc'][0]:.3f}, {ci_dict['auc'][1]:.3f}]  "
              f"AUPRC 95% CI: [{ci_dict['auprc'][0]:.3f}, {ci_dict['auprc'][1]:.3f}]")
        pd.DataFrame([{"metric": k, "ci_lower": v[0], "ci_upper": v[1]}
                      for k, v in ci_dict.items()]).to_csv(
            os.path.join(task_dir, f"{task_name}_{tag}_bootstrap_ci.csv"), index=False)

        save_probability_data(f"{task_name}_{tag}", probs_cat, targets_cat,
                              class_labels, out_dir=task_dir)

        preds_cat = (probs_cat >= threshold).astype(int)
        cm = confusion_matrix(targets_cat, preds_cat)
        pd.DataFrame(cm,
                     index=[f"Actual_{k}" for k in sorted(class_labels.keys())],
                     columns=[f"Predicted_{k}" for k in sorted(class_labels.keys())]).to_csv(
            os.path.join(task_dir, f"{task_name}_{tag}_confusion_matrix.csv"))

        fpr_grid = np.linspace(0, 1, 100)
        roc_rows = []
        for fold_i, (fpr, tpr) in enumerate(roc_curves):
            for f, t in zip(fpr_grid, np.interp(fpr_grid, fpr, tpr)):
                roc_rows.append({"fold": fold_i + 1, "fpr": f, "tpr": t})
        pd.DataFrame(roc_rows).to_csv(
            os.path.join(task_dir, f"{task_name}_{tag}_roc_curves.csv"), index=False)
