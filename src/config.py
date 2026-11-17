"""Configurações compartilhadas entre auditoria, treinamento, interface e LLM."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"
MODEL_PATH = MODEL_DIR / "modelo_risco.joblib"
METRICS_PATH = REPORT_DIR / "metricas.json"
AUDIT_PATH = REPORT_DIR / "auditoria_dataset.json"
DEFAULT_DATA_FILE = "accepted_2007_to_2018Q4.csv"

TARGET = "loan_status"
DATE_COLUMN = "issue_d"
SEED = 42

# Versão do formato de reports/metricas.json lido pela interface.
METRICS_SCHEMA_VERSION = 2

# Amostragem padrão do treinamento (fração uniforme por linha + teto de linhas).
DEFAULT_SAMPLE_FRACTION = 0.15
DEFAULT_MAX_ROWS = 150_000

# Apenas informações conhecidas no início do empréstimo.
NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "fico_range_low",
    "inq_last_6mths", "revol_util",
]
CATEGORICAL_FEATURES = ["term", "purpose", "home_ownership", "emp_length"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Colunas conhecidas somente DEPOIS da concessão (ou derivadas do desfecho).
# Nunca podem entrar em FEATURES — há teste automatizado garantindo isso.
LEAKAGE_COLUMNS = {
    TARGET, "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "out_prncp", "out_prncp_inv", "hardship_flag",
    "debt_settlement_flag", "settlement_status", "pymnt_plan",
}

# Estes dois estados têm desfecho definido. Os demais (Current, Late, In Grace
# Period, Default e "Does not meet the credit policy. Status:...") ficam fora.
CLASS_MAP = {"Fully Paid": 0, "Charged Off": 1}
CLASS_NAMES = {0: "Fully Paid", 1: "Charged Off"}

# Rótulos em português (com unidades) para interface e LLM.
FEATURE_LABELS = {
    "loan_amnt": "Valor solicitado (USD)",
    "annual_inc": "Renda anual declarada (USD)",
    "dti": "Relação dívida/renda — DTI (%)",
    "fico_range_low": "Score FICO (limite inferior da faixa)",
    "inq_last_6mths": "Consultas de crédito nos últimos 6 meses",
    "revol_util": "Uso do crédito rotativo (%)",
    "term": "Prazo",
    "purpose": "Finalidade",
    "home_ownership": "Situação de moradia",
    "emp_length": "Tempo de emprego",
}

CATEGORY_LABELS = {
    "term": {"36 months": "36 meses", "60 months": "60 meses"},
    "purpose": {
        "debt_consolidation": "Consolidação de dívidas",
        "credit_card": "Quitar cartão de crédito",
        "home_improvement": "Reforma da casa",
        "other": "Outra",
        "major_purchase": "Compra de alto valor",
        "medical": "Despesas médicas",
        "small_business": "Pequeno negócio",
        "car": "Carro",
        "vacation": "Férias",
        "moving": "Mudança",
        "house": "Compra de casa",
        "wedding": "Casamento",
        "renewable_energy": "Energia renovável",
        "educational": "Educação",
    },
    "home_ownership": {
        "RENT": "Aluguel",
        "MORTGAGE": "Imóvel financiado (hipoteca)",
        "OWN": "Imóvel próprio quitado",
        "OTHER": "Outra",
        "NONE": "Nenhuma",
        "ANY": "Não especificada (ANY)",
    },
    "emp_length": {
        "< 1 year": "Menos de 1 ano",
        "1 year": "1 ano",
        **{f"{n} years": f"{n} anos" for n in range(2, 10)},
        "10+ years": "10 anos ou mais",
    },
}

# LLM local (Ollama). Variáveis de ambiente permitem trocar sem editar código.
OLLAMA_URL = os.environ.get("LENDINGRISK_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
LLM_MODEL = os.environ.get("LENDINGRISK_LLM_MODEL", "llama3.2:3b")
