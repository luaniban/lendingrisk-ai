"""Métricas, escolha de limiar, calibração e análises por período/subgrupo.

Nomenclatura: `average_precision` é a Average Precision (AP) do scikit-learn —
soma ponderada das precisões ao longo dos limiares. NÃO é a área trapezoidal
sob a curva precisão–recall, embora ambas resumam a mesma curva.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, log_loss,
    precision_score, recall_score, roc_auc_score,
)

THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    b2 = beta ** 2
    denominator = b2 * precision + recall
    return float((1 + b2) * precision * recall / denominator) if denominator else 0.0


def scores(y, probabilities, threshold: float) -> dict:
    """Métricas de ranking (AP, ROC-AUC), de probabilidade (Brier, log loss) e de classe."""
    y = np.asarray(y).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    two_classes = len(np.unique(y)) == 2
    result = {
        "average_precision": average_precision_score(y, probabilities) if two_classes else np.nan,
        "roc_auc": roc_auc_score(y, probabilities) if two_classes else np.nan,
        "brier": brier_score_loss(y, probabilities, labels=[0, 1]),
        "log_loss": log_loss(y, np.clip(probabilities, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "recall": recall,
        "precision": precision,
        "f1": f_beta(precision, recall, beta=1.0),
        "f2": f_beta(precision, recall, beta=2.0),
        "acuracia": accuracy_score(y, predictions),
        "taxa_positiva_real": y.mean(),
        "taxa_positiva_prevista": predictions.mean(),
    }
    return {key: (round(float(value), 5) if not pd.isna(value) else None) for key, value in result.items()}


def reference_scores(y) -> dict:
    """Valores de referência de classificadores triviais, para interpretar AP e Brier."""
    prevalence = float(np.mean(y))
    return {
        "average_precision_aleatorio": round(prevalence, 5),
        "brier_prevalencia_constante": round(prevalence * (1 - prevalence), 5),
        "acuracia_sempre_fully_paid": round(1 - prevalence, 5),
    }


def threshold_curve(y, probabilities, grid=THRESHOLD_GRID, beta: float = 2.0) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    rows = []
    for threshold in grid:
        pred = probabilities >= threshold
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        rows.append({
            "limiar": float(threshold), "precision": precision, "recall": recall,
            "f2": f_beta(precision, recall, beta), "taxa_prevista_positiva": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def choose_threshold(y, probabilities, grid=THRESHOLD_GRID, beta: float = 2.0) -> float:
    """Maximiza F2 (prioriza recall) na VALIDAÇÃO — nunca no teste.

    Em empate, fica o menor limiar da grade (o primeiro a atingir o máximo).
    """
    curve = threshold_curve(y, probabilities, grid, beta)
    best = curve.loc[curve["f2"].idxmax()]
    return round(float(best["limiar"]), 3)


def calibration_table(y, probabilities, n_bins: int = 10) -> pd.DataFrame:
    """Agrupa por quantis do score: probabilidade média prevista x taxa observada."""
    frame = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(probabilities, dtype=float)})
    frame["faixa"] = pd.qcut(frame["p"].rank(method="first"), q=min(n_bins, len(frame)), labels=False)
    table = frame.groupby("faixa").agg(
        prob_media_prevista=("p", "mean"), taxa_observada=("y", "mean"), linhas=("y", "size"),
        prob_min=("p", "min"), prob_max=("p", "max"),
    ).reset_index()
    return table


def expected_calibration_error(y, probabilities, n_bins: int = 10) -> float:
    """ECE com faixas por quantis: média ponderada de |prevista - observada|."""
    table = calibration_table(y, probabilities, n_bins)
    weights = table["linhas"] / table["linhas"].sum()
    return round(float((weights * (table["prob_media_prevista"] - table["taxa_observada"]).abs()).sum()), 5)


def _group_metrics(y, p, threshold) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = p >= threshold
    negatives = (y == 0).sum()
    two_classes = len(np.unique(y)) == 2
    return {
        "linhas": int(len(y)),
        "taxa_charged_off": round(float(y.mean()), 5),
        "score_medio": round(float(p.mean()), 5),
        "average_precision": round(float(average_precision_score(y, p)), 5) if two_classes else None,
        "roc_auc": round(float(roc_auc_score(y, p)), 5) if two_classes else None,
        "recall": round(float(recall_score(y, pred, zero_division=0)), 5),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 5),
        "taxa_falso_positivo": round(float((pred & (y == 0)).sum() / negatives), 5) if negatives else None,
    }


def metrics_by_group(frame: pd.DataFrame, group_col: str, threshold: float,
                     y_col: str = "target", p_col: str = "score", min_rows: int = 100) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(group_col, dropna=False):
        if len(group) < min_rows:
            continue
        rows.append({"atributo": group_col, "valor": "ausente" if pd.isna(value) else value,
                     **_group_metrics(group[y_col], group[p_col], threshold)})
    return pd.DataFrame(rows)


def metrics_by_period(frame: pd.DataFrame, date_col: str, threshold: float,
                      y_col: str = "target", p_col: str = "score", freq: str = "Q") -> pd.DataFrame:
    periods = frame[date_col].dt.to_period(freq).astype(str)
    rows = []
    for period, group in frame.groupby(periods):
        rows.append({"periodo": period, **_group_metrics(group[y_col], group[p_col], threshold)})
    return pd.DataFrame(rows)
