from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_CSV     = ROOT / "radiomics_v6.csv"
DATA_CSV_EMB = ROOT / "embeddings_and_labels.csv"
LABELS_CSV  = ROOT / "ngTMA_table.csv"
ICC_CSV     = ROOT / "radiomics.csv"
FEATURE_CACHE = RESULTS_DIR / "retained_feature_names.txt"

RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Feature column prefixes (PyRadiomics output)
# ---------------------------------------------------------------------------
FEATURE_PREFIXES = (
    "original_",
    "wavelet-",
    "log-sigma-",
    "square_",
    "squareroot_",
    "gradient_",
    "logarithm_",
    "exponential_",
)

# ---------------------------------------------------------------------------
# Diagnosis normalisation
#   0 = PTC   1 = FTN (FA / FTC)   2 = PDTC
#   3 = Oncocytic   4 = FVPTC   -1 = exclude
# ---------------------------------------------------------------------------
DIAGNOSIS_MAP: dict[str, int] = {
    # String labels (TMAs 64 / 93 / 94 and ngTMA_table)
    "PTC":    0,
    "FVPTC":  4,
    "FA":     1,
    "FTC":    1,
    "PDTC":   2,
    "OC":     3,
    "OA":     3,
    "FTC_PDTC":                       1,
    "nN":                            -1,
    "NH":                            -1,
    "MTC":                           -1,
    "Missing":                       -1,
    "Nodular hyperplasia":           -1,
    "Medullary Thyroid Carcinoma":   -1,
    "oncocytic carcinoma und PTC":    0,
    "O_PDTC":                         2,
    "nan":                           -1,
    # Numeric codes from TMAs 64 / 93 / 94 (embeddings_and_labels.csv)
    "1":  0,   # Papillär classic → PTC
    "2":  4,   # Papillär follikuläre Variante → FVPTC
    "3":  2,   # Papillär poorly diff. Anteil → PDTC
    "4":  1,   # Follikulär minimally invasive → FTN
    "5":  1,   # Follikulär widely invasive → FTN
    "6":  0,   # Papilläres Karzinom spezielle Variante → PTC
    "7":  3,   # Follikulär onkozytär minimally invasive → Oncocytic
    "8":  3,   # Follikulär onkozytär widely invasive → Oncocytic
    "9":  2,   # Follikulär poorly diff. Anteil → PDTC
    "10": 3,   # Follikuläres onkozytäres Adenom → Oncocytic
    "11": 2,   # Follikulär poorly differentiated onkozytär → PDTC
    "12": 1,   # Follikuläres Adenom → FTN
}

# Patient IDs excluded from all analyses
EXCLUDED_IDS: list[int] = [734, 560, 654]


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig:
    name: str
    class_labels: dict
    stratified: bool = False
    threshold: float = 0.5
    class_weights: list = None
    hidden_dim: int = 128
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    dropout_rate: float = 0.4
    gamma: float = 0.9
