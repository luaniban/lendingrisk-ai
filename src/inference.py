"""Carregamento de artefatos, validação de entradas e previsão para a interface.

Mantido fora do app.py para ser testável sem Streamlit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES, CATEGORY_LABELS, FEATURE_LABELS, FEATURES,
    METRICS_SCHEMA_VERSION, NUMERIC_FEATURES,
)

REQUIRED_METRIC_KEYS = ["modelo_escolhido", "limiar", "atributos", "metricas_teste", "comparacao_validacao"]
MISSING_LABEL = "Não informado"


class ArtefatosInvalidos(RuntimeError):
    """Modelo/relatórios ausentes, corrompidos ou incompatíveis com o código atual."""


def load_metrics(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ArtefatosInvalidos(f"Arquivo de métricas não encontrado: {path.name}.")
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtefatosInvalidos(f"{path.name} está corrompido ({exc}). Treine novamente.") from exc
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if missing:
        raise ArtefatosInvalidos(f"{path.name} não tem as chaves {missing}. Treine novamente.")
    if list(metrics["atributos"]) != FEATURES:
        raise ArtefatosInvalidos(
            "O modelo salvo foi treinado com atributos diferentes dos definidos em src/config.py: "
            f"{metrics['atributos']} ≠ {FEATURES}. Treine novamente."
        )
    threshold = metrics["limiar"]
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ArtefatosInvalidos(f"Limiar inválido em {path.name}: {threshold!r}.")
    return metrics


def load_model(path: Path):
    """Carrega o Pipeline. Use APENAS arquivos gerados localmente (joblib executa código)."""
    path = Path(path)
    if not path.is_file():
        raise ArtefatosInvalidos(f"Modelo não encontrado: {path.name}.")
    try:
        model = joblib.load(path)
    except Exception as exc:  # arquivo truncado, versão incompatível etc.
        raise ArtefatosInvalidos(f"Não foi possível carregar {path.name}: {exc}") from exc
    if not hasattr(model, "predict_proba"):
        raise ArtefatosInvalidos(f"{path.name} não contém um classificador com predict_proba.")
    return model


def version_warnings(metrics: dict) -> list[str]:
    """Avisa se o modelo foi treinado com versões diferentes das instaladas agora."""
    import sklearn
    import xgboost
    trained = metrics.get("ambiente", {})
    warnings = []
    for key, current in [("scikit_learn", sklearn.__version__), ("xgboost", xgboost.__version__)]:
        if trained.get(key) and trained[key] != current:
            warnings.append(f"Modelo treinado com {key} {trained[key]}, mas está instalado {current}.")
    if metrics.get("versao_esquema") != METRICS_SCHEMA_VERSION:
        warnings.append("metricas.json foi gerado por uma versão antiga do treinamento; alguns painéis podem faltar.")
    return warnings


def validate_record(record: dict, schema: dict | None = None) -> tuple[dict, list[str]]:
    """Confere tipos/faixas. Erros impeditivos levantam ValueError; avisos são devolvidos.

    Categorias desconhecidas não quebram o modelo (OneHotEncoder ignora), mas geram aviso.
    """
    missing = [f for f in FEATURES if f not in record]
    if missing:
        raise ValueError(f"Campos ausentes: {missing}")
    clean, warnings = {}, []
    for col in NUMERIC_FEATURES:
        value = record[col]
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{FEATURE_LABELS[col]}: valor numérico inválido ({value!r}).") from None
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{FEATURE_LABELS[col]}: valor não pode ser vazio/infinito.")
        if value < 0:
            raise ValueError(f"{FEATURE_LABELS[col]}: valor não pode ser negativo.")
        stats = (schema or {}).get("numericos", {}).get(col)
        if stats and not stats["p01"] <= value <= stats["p99"]:
            warnings.append(
                f"{FEATURE_LABELS[col]} = {value:g} está fora da faixa usual do treino "
                f"(1%–99%: {stats['p01']:g} a {stats['p99']:g}); a estimativa é menos confiável."
            )
        clean[col] = value
    for col in CATEGORICAL_FEATURES:
        value = record[col]
        if value is None or (isinstance(value, float) and math.isnan(value)) or value == MISSING_LABEL:
            clean[col] = np.nan
            continue
        value = str(value).strip()
        known = (schema or {}).get("categoricos", {}).get(col)
        if known and value not in known:
            warnings.append(f"{FEATURE_LABELS[col]}: categoria '{value}' não apareceu no treino (será ignorada).")
        clean[col] = value
    return clean, warnings


def predict_probability(model, record: dict) -> float:
    frame = pd.DataFrame([record])[FEATURES]
    probability = float(model.predict_proba(frame)[0, 1])
    if not 0.0 <= probability <= 1.0 or math.isnan(probability):
        raise ValueError(f"O modelo devolveu uma probabilidade inválida: {probability!r}")
    return probability


def category_label(column: str, value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING_LABEL
    return CATEGORY_LABELS.get(column, {}).get(value, str(value))


def _brl_number(value: float, decimals: int = 0) -> str:
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def format_record_pt(record: dict) -> dict:
    """Valores legíveis em português, com unidades — usados na tela e no prompt da LLM."""
    out = {}
    for col in FEATURES:
        value = record[col]
        if col in ("loan_amnt", "annual_inc"):
            text = f"US$ {_brl_number(value)}"
        elif col in ("dti", "revol_util"):
            text = f"{_brl_number(value, 1)}%"
        elif col in NUMERIC_FEATURES:
            text = _brl_number(value)
        else:
            text = category_label(col, value)
        out[FEATURE_LABELS[col]] = text
    return out


def percent_pt(value: float, decimals: int = 1) -> str:
    return f"{_brl_number(value * 100, decimals)}%"
