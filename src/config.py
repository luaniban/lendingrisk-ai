"""Configurações compartilhadas entre treinamento e interface."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "modelo_risco.joblib"
METRICS_PATH = REPORT_DIR / "metricas.json"

TARGET = "loan_status"
DATE_COLUMN = "issue_d"
SEED = 42

# Apenas informações conhecidas no início do empréstimo.
# Não usar last_fico_range_high, total_pymnt, recoveries, last_pymnt_d etc.
NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "fico_range_low",
    "inq_last_6mths", "revol_util",
]
CATEGORICAL_FEATURES = ["term", "purpose", "home_ownership", "emp_length"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Estes dois estados já têm resultado definido.
CLASS_MAP = {"Fully Paid": 0, "Charged Off": 1}
