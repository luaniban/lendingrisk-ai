"""Treinamento reprodutível de 3 classificadores no Lending Club Dataset.

Uso: python -m src.training --data data/accepted_2007_to_2018Q4.csv.gz
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    CATEGORICAL_FEATURES, CLASS_MAP, DATE_COLUMN, FEATURES,
    METRICS_PATH, MODEL_DIR, MODEL_PATH, NUMERIC_FEATURES,
    REPORT_DIR, SEED, TARGET,
)


def load_dataset(path: Path, sample_fraction: float = 0.15, max_rows: int = 150_000) -> pd.DataFrame:
    """Lê só colunas autorizadas em chunks e amostra por todo o arquivo."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    header = set(pd.read_csv(path, nrows=0).columns)
    required = FEATURES + [TARGET, DATE_COLUMN]
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(
            f"O CSV não tem as colunas esperadas: {missing}. "
            "Confirme se é o arquivo de empréstimos ACEITOS do Lending Club."
        )
    chunks = []
    for idx, chunk in enumerate(
        pd.read_csv(path, usecols=required, chunksize=50_000, dtype=str, low_memory=False)
    ):
        chunk[TARGET] = chunk[TARGET].str.strip()
        chunk = chunk.loc[chunk[TARGET].isin(CLASS_MAP)].copy()
        if chunk.empty:
            continue
        size = min(len(chunk), max(1_000, round(len(chunk) * sample_fraction)))
        chunks.append(chunk.sample(n=size, random_state=SEED + idx))
    if not chunks:
        raise ValueError("Nenhuma linha com 'Fully Paid' ou 'Charged Off' foi encontrada.")
    data = pd.concat(chunks, ignore_index=True)
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=SEED).reset_index(drop=True)

    for col in NUMERIC_FEATURES:
        # Converte números que venham com vírgula de milhar ou símbolo de porcentagem.
        data[col] = pd.to_numeric(
            data[col].str.replace(",", "", regex=False)
                     .str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype("string").str.strip().replace("", pd.NA)
        data[col] = data[col].astype(object).where(data[col].notna(), np.nan)

    # Exemplo de data no dataset original: "Dec-2018".
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], format="%b-%Y", errors="coerce")
    data = data.dropna(subset=[DATE_COLUMN]).copy()
    data["target"] = data[TARGET].map(CLASS_MAP).astype(int)
    if len(data) < 200 or data["target"].nunique() != 2:
        raise ValueError("Dataset insuficiente para treinar as duas classes.")
    return data.sort_values(DATE_COLUMN).reset_index(drop=True)


def temporal_split(data: pd.DataFrame):
    """Treino 70%, validação 15%, teste 15%; meses não cruzam partições."""
    n = len(data)
    cutoff_train = data.iloc[int(n * 0.70)][DATE_COLUMN]
    cutoff_val = data.iloc[int(n * 0.85)][DATE_COLUMN]
    train = data[data[DATE_COLUMN] < cutoff_train]
    val = data[(data[DATE_COLUMN] >= cutoff_train) & (data[DATE_COLUMN] < cutoff_val)]
    test = data[data[DATE_COLUMN] >= cutoff_val]
    for name, subset in [("treino", train), ("validação", val), ("teste", test)]:
        if subset.empty or subset["target"].nunique() < 2:
            raise ValueError(
                f"Partição {name} vazia ou sem ambas as classes. "
                "Confira o formato/intervalo de issue_d e a quantidade de exemplos."
            )
    return train, val, test


def build_pipeline(model):
    preprocessing = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ],
    )
    return Pipeline([("prep", preprocessing), ("model", model)])


def scores(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(y, probabilities)), 5),
        "recall": round(float(recall_score(y, predictions, zero_division=0)), 5),
        "precision": round(float(precision_score(y, predictions, zero_division=0)), 5),
        "f1": round(float(f1_score(y, predictions, zero_division=0)), 5),
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 5),
        "brier": round(float(brier_score_loss(y, probabilities)), 5),
        "acuracia": round(float(accuracy_score(y, predictions)), 5),
    }


def choose_threshold(y, probabilities) -> float:
    """Maximiza F2 na validação (prioriza recall), nunca no teste."""
    best_f2, best_threshold = -1.0, 0.5
    for threshold in np.linspace(0.10, 0.90, 81):
        pred = probabilities >= threshold
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) else 0
        if f2 > best_f2:
            best_f2, best_threshold = f2, float(threshold)
    return round(best_threshold, 3)


def train(data_path: Path, fraction: float = 0.15, max_rows: int = 150_000):
    print("Lendo e preparando dados (pode levar alguns minutos)...")
    data = load_dataset(data_path, sample_fraction=fraction, max_rows=max_rows)
    train_set, val_set, test_set = temporal_split(data)
    print("Linhas:", {"total": len(data), "treino": len(train_set), "val": len(val_set), "teste": len(test_set)})
    x_train, y_train = train_set[FEATURES], train_set["target"]
    x_val, y_val = val_set[FEATURES], val_set["target"]
    x_test, y_test = test_set[FEATURES], test_set["target"]
    candidates = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=14, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=160, max_depth=4, learning_rate=0.08, subsample=0.85,
            colsample_bytree=0.9, random_state=SEED, n_jobs=4,
            tree_method="hist", eval_metric="logloss",
        ),
    }
    comparison = []
    best_name, best_model, best_pr_auc, best_threshold = None, None, -1, 0.5
    for name, classifier in candidates.items():
        print(f"Treinando {name}...")
        pipeline = build_pipeline(classifier)
        pipeline.fit(x_train, y_train)
        val_probabilities = pipeline.predict_proba(x_val)[:, 1]
        threshold = choose_threshold(y_val, val_probabilities)
        metrics = scores(y_val, val_probabilities, threshold)
        comparison.append({"modelo": name, "limiar": threshold, **metrics})
        print(f"  PR-AUC validação = {metrics['pr_auc']:.4f}; recall = {metrics['recall']:.4f}")
        if metrics["pr_auc"] > best_pr_auc:
            best_name, best_model = name, pipeline
            best_pr_auc, best_threshold = metrics["pr_auc"], threshold

    # Teste é calculado uma única vez, com modelo/limiar escolhidos sem olhar seu resultado.
    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    test_metrics = scores(y_test, test_probabilities, best_threshold)
    confusion = confusion_matrix(y_test, test_probabilities >= best_threshold, labels=[0, 1])
    print(f"Modelo escolhido: {best_name}, PR-AUC teste: {test_metrics['pr_auc']:.4f}")

    # Importância GLOBAL por permutação: não prova causalidade nem explica individualmente cada previsão.
    sample_val = val_set.sample(n=min(600, len(val_set)), random_state=SEED)
    importance = permutation_importance(
        best_model, sample_val[FEATURES], sample_val["target"],
        n_repeats=2, random_state=SEED, scoring="average_precision", n_jobs=1,
    )
    importance_df = pd.DataFrame({
        "atributo": FEATURES,
        "queda_pr_auc_media": importance.importances_mean,
        "desvio": importance.importances_std,
    }).sort_values("queda_pr_auc_media", ascending=False)

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(best_model, MODEL_PATH)
    pd.DataFrame(comparison).to_csv(REPORT_DIR / "comparacao_modelos.csv", index=False)
    importance_df.to_csv(REPORT_DIR / "importancia_atributos.csv", index=False)
    pd.DataFrame(confusion, index=["real_pago", "real_nao_pago"], columns=[
        "previsto_pago", "previsto_nao_pago"
    ]).to_csv(REPORT_DIR / "matriz_confusao_teste.csv")
    payload = {
        "modelo_escolhido": best_name,
        "limiar": best_threshold,
        "classes": {"0": "Fully Paid", "1": "Charged Off"},
        "atributos": FEATURES,
        "linhas": {"total": len(data), "treino": len(train_set), "validacao": len(val_set), "teste": len(test_set)},
        "periodos": {
            "treino": [str(train_set[DATE_COLUMN].min().date()), str(train_set[DATE_COLUMN].max().date())],
            "validacao": [str(val_set[DATE_COLUMN].min().date()), str(val_set[DATE_COLUMN].max().date())],
            "teste": [str(test_set[DATE_COLUMN].min().date()), str(test_set[DATE_COLUMN].max().date())],
        },
        "comparacao_validacao": comparison,
        "metricas_teste": test_metrics,
        "matriz_confusao_teste": confusion.tolist(),
        "taxa_classe_positiva_teste": round(float(y_test.mean()), 5),
        "criado_em_utc": datetime.now(timezone.utc).isoformat(),
    }
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Arquivos gerados em: {MODEL_DIR} e {REPORT_DIR}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina os modelos LendingRisk AI")
    parser.add_argument("--data", required=True, type=Path, help="Caminho do CSV ou CSV.GZ")
    parser.add_argument("--fraction", default=0.15, type=float, help="Fração amostrada de cada bloco")
    parser.add_argument("--max-rows", default=150_000, type=int, help="Máximo de linhas da amostra")
    args = parser.parse_args()
    if not (0 < args.fraction <= 1):
        parser.error("--fraction deve estar entre 0 e 1")
    train(args.data, fraction=args.fraction, max_rows=args.max_rows)
