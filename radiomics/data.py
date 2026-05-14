import numpy as np
import pandas as pd

from .config import DATA_CSV, DATA_CSV_EMB, LABELS_CSV, FEATURE_PREFIXES, DIAGNOSIS_MAP


class DataLoader:
    """Loads radiomics features and clinical labels, merges on (TMA, Grid, x, y).

    Features come from Radiomics_v6.csv. Diagnosis (for FVPTC detection via
    numeric code "2" in TMAs 64/93/94) is joined from embeddings_and_labels.csv.
    Clinical labels come from ngTMA_table.
    """

    def __init__(self, features_csv=None, labels_csv=None):
        self.features_csv = features_csv or DATA_CSV
        self.labels_csv   = labels_csv   or LABELS_CSV

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        emb    = pd.read_csv(self.features_csv, low_memory=False)
        labels = pd.read_csv(self.labels_csv)

        feat_cols  = [c for c in emb.columns if c.startswith(FEATURE_PREFIXES)]
        merge_keys = ["TMA", "Grid", "x", "y"]

        # Radiomics_v6.csv has x/y stored as pandas Series repr — extract the integer value
        for col in ["x", "y"]:
            if emb[col].astype(str).str.contains("dtype").any():
                emb[col] = emb[col].astype(str).str.split().str[1].astype(int)

        # Radiomics_v6.csv uses Grid "B" for TMA 2406; labels use "A" — normalise
        emb.loc[(emb["TMA"].astype(str) == "2406") & (emb["Grid"].astype(str) == "B"), "Grid"] = "A"

        for col in merge_keys:
            emb[col]    = emb[col].astype(str)
            labels[col] = labels[col].astype(str)

        outcome_cols = ["ID", "PID", "tissue", "Diagnosis", "BRAF p/n", "BRAF_2nd",
                        "Relapse p/n", "RAS", "TERT", "malignancy"]
        outcome_cols = [c for c in outcome_cols if c in labels.columns]

        # Join clinical labels
        merged = (
            emb[merge_keys + feat_cols]
            .merge(labels[merge_keys + outcome_cols], on=merge_keys, how="left")
        )

        # Join Diagnosis from embeddings_and_labels.csv to detect FVPTC (code "2")
        try:
            emb_old = pd.read_csv(DATA_CSV_EMB, low_memory=False,
                                  usecols=merge_keys + ["Diagnosis"])
            for col in merge_keys:
                emb_old[col] = emb_old[col].astype(str)
            merged = merged.merge(emb_old[merge_keys + ["Diagnosis"]],
                                  on=merge_keys, how="left",
                                  suffixes=("", "_emb"))
        except Exception:
            merged["Diagnosis_emb"] = None

        features = merged[feat_cols].reset_index(drop=True)

        keep = merge_keys + outcome_cols
        if "Diagnosis_emb" in merged.columns:
            keep = keep + ["Diagnosis_emb"]
        meta = merged[keep].copy().reset_index(drop=True)

        diag_col = "Diagnosis" if "Diagnosis" in meta.columns else None
        meta["diagnosis_class"] = (
            meta[diag_col].astype(str).map(DIAGNOSIS_MAP)
            if diag_col else pd.Series(dtype=float)
        )

        meta["is_fvptc"] = (
            meta["Diagnosis_emb"].astype(str) == "2"
            if "Diagnosis_emb" in meta.columns
            else False
        )

        meta["batch"]      = meta["TMA"].astype(str) + "_" + meta["Grid"].astype(str)
        mask_missing = meta["PID"].isna()
        if mask_missing.any():
            next_id = int(meta["PID"].max() + 1) if not meta["PID"].isna().all() else 0
            meta.loc[mask_missing, "PID"] = np.arange(next_id, next_id + mask_missing.sum())
            print(f"  {mask_missing.sum()} samples had no PID — assigned synthetic unique PIDs")

        meta["patient_id"] = meta["PID"].astype(int)

        for col in ["BRAF p/n", "BRAF_2nd", "Relapse p/n", "RAS", "TERT"]:
            if col in meta.columns:
                meta[col] = pd.to_numeric(meta[col], errors="coerce")

        print(f"Loaded {len(features)} samples, {features.shape[1]} features")
        return features, meta
