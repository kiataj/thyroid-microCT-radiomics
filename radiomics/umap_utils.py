import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def compute_umap_and_save(task_name: str, X: np.ndarray, y: np.ndarray,
                           class_labels: dict, out_dir: str) -> tuple:
    from umap import UMAP

    print(f"  Computing UMAP for {task_name}...")
    reducer   = UMAP(n_neighbors=10, min_dist=0.1, n_components=2,
                     random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(X)

    sil = silhouette_score(embedding, y)
    print(f"  Silhouette score (UMAP space): {sil:.4f}")

    pd.DataFrame({"umap1": embedding[:, 0], "umap2": embedding[:, 1],
                  "label": y}).to_csv(
        os.path.join(out_dir, f"{task_name}_umap_embedding.csv"), index=False)

    with open(os.path.join(out_dir, f"{task_name}_umap_meta.json"), "w") as f:
        json.dump({"silhouette": sil,
                   "class_labels": {str(k): v for k, v in class_labels.items()}}, f)

    return embedding, sil
