"""Treinamento reprodutível de 3 classificadores no Lending Club Dataset.

Uso: python -m src.training --data data/accepted_2007_to_2018Q4.csv

Regras metodológicas (ver docs/experimentos.md):
- pré-processamento dentro do Pipeline, ajustado SOMENTE no treino;
- modelo escolhido pela Average Precision (AP) na VALIDAÇÃO;
- limiar escolhido maximizando F2 na VALIDAÇÃO;
- teste usado UMA vez, apenas para o modelo/limiar já escolhidos.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import figures
from src.config import (
    CATEGORICAL_FEATURES, CLASS_NAMES, DATE_COLUMN, DEFAULT_MAX_ROWS,
    DEFAULT_SAMPLE_FRACTION, FEATURES, METRICS_PATH, METRICS_SCHEMA_VERSION, MISSING_CATEGORY,
    MODEL_PATH, NUMERIC_FEATURES, REPORT_DIR, SEED,
)
from src.data import describe_partition, load_dataset, sha256_file, temporal_split
from src.evaluation import (  # noqa: F401  (scores/choose_threshold reexportados)
    calibration_table, choose_threshold, expected_calibration_error,
    metrics_by_group, metrics_by_period, reference_scores, scores, threshold_curve,
)

SELECTION_METRIC = "average_precision"


def make_candidates(seed: int = SEED) -> dict:
    """Os três modelos definidos no Entregável 2 (hiperparâmetros iniciais do starter)."""
    return {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=14, min_samples_leaf=3, random_state=seed, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=160, max_depth=4, learning_rate=0.08, subsample=0.85,
            colsample_bytree=0.9, random_state=seed, n_jobs=4,
            tree_method="hist", eval_metric="logloss",
        ),
    }


def build_pipeline(model, missing_as_category: bool = False):
    """missing_as_category=False (oficial): ausentes categóricos recebem a moda do treino.
    True (variante v2): ausentes viram a categoria própria MISSING_CATEGORY."""
    categorical_imputer = (SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY) if missing_as_category
                           else SimpleImputer(strategy="most_frequent"))
    preprocessing = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("imputer", categorical_imputer),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ],
    )
    return Pipeline([("prep", preprocessing), ("model", model)])


def input_schema(train_set: pd.DataFrame) -> dict:
    """Faixas e categorias vistas no TREINO — usadas pela interface para validar entradas."""
    numeric = {}
    for col in NUMERIC_FEATURES:
        values = train_set[col].dropna()
        numeric[col] = {
            "min": float(values.min()), "p01": float(values.quantile(0.01)),
            "mediana": float(values.median()), "p99": float(values.quantile(0.99)),
            "max": float(values.max()), "ausentes": round(float(train_set[col].isna().mean()), 5),
        }
    categorical = {
        col: [str(v) for v in train_set[col].value_counts().index]
        for col in CATEGORICAL_FEATURES
    }
    missing_cat = {col: round(float(train_set[col].isna().mean()), 5) for col in CATEGORICAL_FEATURES}
    return {"numericos": numeric, "categoricos": categorical, "ausentes_categoricos": missing_cat}


def environment() -> dict:
    return {
        "python": sys.version.split()[0], "plataforma": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "joblib": joblib.__version__,
    }


def _log(message: str, verbose: bool):
    if verbose:
        print(message, flush=True)


def train(
    data_path: Path,
    fraction: float = DEFAULT_SAMPLE_FRACTION,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    maturity_filter: bool = True,
    model_path: Path = MODEL_PATH,
    report_dir: Path = REPORT_DIR,
    make_figures: bool = True,
    importance_rows: int = 5_000,
    importance_repeats: int = 5,
    seed: int = SEED,
    verbose: bool = True,
    compute_hash: bool = True,
    missing_as_category: bool = False,
):
    started = time.time()
    data_path = Path(data_path)
    report_dir = Path(report_dir)
    figures_dir = report_dir / "figures"
    metrics_path = report_dir / METRICS_PATH.name

    _log("Lendo e preparando dados (pode levar alguns minutos)...", verbose)
    data, data_info = load_dataset(
        data_path, sample_fraction=fraction, max_rows=max_rows,
        maturity_filter=maturity_filter, seed=seed,
    )
    if compute_hash:
        data_info["sha256"] = sha256_file(data_path)
    train_set, val_set, test_set = temporal_split(data)
    partitions = {"treino": describe_partition(train_set), "validacao": describe_partition(val_set),
                  "teste": describe_partition(test_set)}
    _log(f"Partições: {json.dumps(partitions, ensure_ascii=False)}", verbose)

    x_train, y_train = train_set[FEATURES], train_set["target"]
    x_val, y_val = val_set[FEATURES], val_set["target"]
    x_test, y_test = test_set[FEATURES], test_set["target"]

    candidates = make_candidates(seed)
    comparison, fitted, val_probs, curves, calibration_val = [], {}, {}, [], []
    for name, classifier in candidates.items():
        _log(f"Treinando {name}...", verbose)
        t0 = time.time()
        pipeline = build_pipeline(classifier, missing_as_category)
        pipeline.fit(x_train, y_train)  # pré-processamento ajustado só no treino
        probabilities = pipeline.predict_proba(x_val)[:, 1]
        threshold = choose_threshold(y_val, probabilities)
        metrics = scores(y_val, probabilities, threshold)
        metrics["ece"] = expected_calibration_error(y_val, probabilities)
        comparison.append({"modelo": name, "limiar": threshold, **metrics,
                           "tempo_treino_s": round(time.time() - t0, 1)})
        fitted[name], val_probs[name] = pipeline, probabilities
        curve = threshold_curve(y_val, probabilities)
        curve.insert(0, "modelo", name)
        curves.append(curve)
        table = calibration_table(y_val, probabilities)
        table.insert(0, "modelo", name)
        calibration_val.append(table)
        _log(f"  AP validação = {metrics['average_precision']:.4f}; ROC-AUC = {metrics['roc_auc']:.4f}; "
             f"limiar F2 = {threshold:.2f}; recall = {metrics['recall']:.4f}", verbose)

    # Seleção usando SOMENTE a validação (empate: primeiro da lista, o mais simples).
    best = max(comparison, key=lambda row: row[SELECTION_METRIC])
    best_name, best_threshold = best["modelo"], best["limiar"]
    best_model = fitted[best_name]

    # Teste: calculado uma única vez, com modelo e limiar já escolhidos.
    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    test_metrics = scores(y_test, test_probabilities, best_threshold)
    test_metrics["ece"] = expected_calibration_error(y_test, test_probabilities)
    confusion = confusion_matrix(y_test, test_probabilities >= best_threshold, labels=[0, 1])
    _log(f"Modelo escolhido: {best_name} (limiar {best_threshold:.2f}); "
         f"AP teste = {test_metrics['average_precision']:.4f}", verbose)

    # Importância GLOBAL por permutação na validação: não explica previsões individuais.
    sample_val = val_set.sample(n=min(importance_rows, len(val_set)), random_state=seed)
    importance = permutation_importance(
        best_model, sample_val[FEATURES], sample_val["target"],
        n_repeats=importance_repeats, random_state=seed, scoring="average_precision", n_jobs=1,
    )
    importance_df = pd.DataFrame({
        "atributo": FEATURES,
        "queda_ap_media": importance.importances_mean,
        "desvio": importance.importances_std,
    }).sort_values("queda_ap_media", ascending=False)

    # Análises de erro, calibração e estabilidade (descritivas; não alteram a escolha).
    scored_val = val_set.assign(score=val_probs[best_name])
    scored_test = test_set.assign(score=test_probabilities)
    stability = pd.concat([
        metrics_by_period(scored_val, DATE_COLUMN, best_threshold).assign(particao="validação"),
        metrics_by_period(scored_test, DATE_COLUMN, best_threshold).assign(particao="teste"),
    ], ignore_index=True)
    subgroups = pd.concat(
        [metrics_by_group(scored_test, col, best_threshold) for col in CATEGORICAL_FEATURES],
        ignore_index=True,
    )
    calibration_test = calibration_table(y_test, test_probabilities)

    # ---- Gravação dos artefatos ----
    model_path = Path(model_path)
    model_path.parent.mkdir(exist_ok=True, parents=True)
    report_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(best_model, model_path)
    pd.DataFrame(comparison).to_csv(report_dir / "comparacao_modelos.csv", index=False)
    importance_df.to_csv(report_dir / "importancia_atributos.csv", index=False)
    pd.DataFrame(confusion, index=["real_fully_paid", "real_charged_off"], columns=[
        "previsto_fully_paid", "previsto_charged_off"
    ]).to_csv(report_dir / "matriz_confusao_teste.csv")
    pd.concat(curves).to_csv(report_dir / "curva_limiar_validacao.csv", index=False)
    pd.concat(calibration_val).to_csv(report_dir / "calibracao_validacao.csv", index=False)
    calibration_test.to_csv(report_dir / "calibracao_teste.csv", index=False)
    stability.to_csv(report_dir / "estabilidade_temporal.csv", index=False)
    subgroups.to_csv(report_dir / "analise_subgrupos_teste.csv", index=False)

    figure_files = []
    if make_figures:
        _log("Gerando gráficos...", verbose)
        val_curves = {name: (y_val, p) for name, p in val_probs.items()}
        prevalence_val = float(y_val.mean())
        specs = [
            ("curvas_pr_validacao.png", figures.plot_pr_curves,
             (val_curves, prevalence_val), "Precisão × recall na validação (3 modelos)"),
            ("curvas_roc_validacao.png", figures.plot_roc_curves, (val_curves,), "Curvas ROC na validação"),
            ("calibracao_validacao.png", figures.plot_calibration,
             ({n: t for n, t in zip(val_probs, calibration_val)},), "Calibração na validação"),
            ("limiar_validacao.png", figures.plot_threshold_curve,
             (pd.concat(curves).loc[lambda df: df["modelo"] == best_name], best_threshold),
             f"Escolha do limiar por F2 — {best_name} (validação)"),
            ("matriz_confusao_teste.png", figures.plot_confusion_matrix, (confusion,),
             f"Matriz de confusão no teste — {best_name}"),
            ("distribuicao_scores_teste.png", figures.plot_score_distribution,
             (y_test, test_probabilities, best_threshold), "Distribuição dos scores no teste por classe real"),
            ("importancia_atributos.png", figures.plot_importance, (importance_df,),
             f"Importância global por permutação ({best_name}, validação)"),
            ("estabilidade_temporal.png", figures.plot_stability, (stability,),
             f"Estabilidade por mês de emissão — {best_name}"),
            ("calibracao_teste.png", figures.plot_calibration, ({best_name: calibration_test},),
             f"Calibração no teste — {best_name}"),
        ]
        for filename, function, args, title in specs:
            function(*args, figures_dir / filename, title)
            figure_files.append(f"figures/{filename}")

    payload = {
        "versao_esquema": METRICS_SCHEMA_VERSION,
        "modelo_escolhido": best_name,
        "limiar": best_threshold,
        "classes": {str(k): v for k, v in CLASS_NAMES.items()},
        "atributos": FEATURES,
        "atributos_numericos": NUMERIC_FEATURES,
        "atributos_categoricos": CATEGORICAL_FEATURES,
        "criterios": {
            "selecao_modelo": "maior Average Precision (AP) na validação",
            "limiar": "maior F2 na validação (grade 0,01–0,99, passo 0,01)",
            "teste": "avaliado uma única vez com o modelo e o limiar já escolhidos",
            "categoricos_ausentes": (f"categoria própria '{MISSING_CATEGORY}'" if missing_as_category
                                     else "moda do treino"),
        },
        "variante": {"ausente_como_categoria": missing_as_category},
        "dataset": data_info,
        "linhas": {k: v["linhas"] for k, v in partitions.items()} | {"total": len(data)},
        "particoes": partitions,
        "periodos": {k: [v["inicio"], v["fim"]] for k, v in partitions.items()},
        "comparacao_validacao": comparison,
        "metricas_teste": test_metrics,
        "referencias_teste": reference_scores(y_test),
        "matriz_confusao_teste": confusion.tolist(),
        "taxa_classe_positiva_teste": round(float(y_test.mean()), 5),
        "importancia_global": importance_df.round(5).to_dict(orient="records"),
        "esquema_entrada": input_schema(train_set),
        "parametros_modelos": {name: {k: v for k, v in est.get_params().items()
                                      if isinstance(v, (int, float, str, bool, type(None)))}
                               for name, est in candidates.items()},
        "permutacao": {"linhas_validacao": int(len(sample_val)), "repeticoes": importance_repeats},
        "figuras": figure_files,
        "semente": seed,
        "ambiente": environment(),
        "duracao_segundos": round(time.time() - started, 1),
        "criado_em_utc": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_history(report_dir, payload, fraction, max_rows, maturity_filter, missing_as_category)
    _log(f"Arquivos gerados em: {model_path.parent} e {report_dir} ({payload['duracao_segundos']} s)", verbose)
    return payload


def _append_history(report_dir: Path, payload: dict, fraction, max_rows, maturity_filter,
                    missing_as_category: bool = False):
    """Registra cada execução sem sobrescrever as anteriores (rastreabilidade)."""
    row = {
        "criado_em_utc": payload["criado_em_utc"],
        "arquivo": payload["dataset"]["arquivo"],
        "sha256": payload["dataset"].get("sha256", ""),
        "fracao": fraction, "max_linhas": max_rows, "filtro_maturidade": maturity_filter,
        "ausente_como_categoria": missing_as_category,
        "linhas_total": payload["linhas"]["total"],
        **{f"ap_val_{r['modelo']}": r["average_precision"] for r in payload["comparacao_validacao"]},
        "modelo_escolhido": payload["modelo_escolhido"], "limiar": payload["limiar"],
        "ap_teste": payload["metricas_teste"]["average_precision"],
        "roc_auc_teste": payload["metricas_teste"]["roc_auc"],
        "recall_teste": payload["metricas_teste"]["recall"],
        "precision_teste": payload["metricas_teste"]["precision"],
        "duracao_s": payload["duracao_segundos"],
    }
    history = report_dir / "historico_execucoes.csv"
    frame = pd.DataFrame([row])
    if history.exists():
        frame = pd.concat([pd.read_csv(history), frame], ignore_index=True)
    frame.to_csv(history, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Treina os modelos LendingRisk AI")
    parser.add_argument("--data", required=True, type=Path, help="Caminho do CSV ou CSV.GZ")
    parser.add_argument("--fraction", default=DEFAULT_SAMPLE_FRACTION, type=float,
                        help="Fração amostrada uniformemente (por linha) do arquivo (padrão 1 = tudo)")
    parser.add_argument("--max-rows", default=DEFAULT_MAX_ROWS, type=int,
                        help="Máximo de linhas após filtros (padrão/0 = sem teto)")
    parser.add_argument("--no-maturity-filter", action="store_true",
                        help="Desativa o filtro de maturidade (reproduz o comportamento do starter)")
    parser.add_argument("--no-figures", action="store_true", help="Não gera os PNGs")
    parser.add_argument("--missing-as-category", action="store_true",
                        help="Variante: ausentes categóricos viram categoria própria (padrão: moda do treino)")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH,
                        help=f"Onde salvar o modelo (padrão: {MODEL_PATH.relative_to(MODEL_PATH.parents[1])})")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR,
                        help="Pasta das métricas, tabelas e gráficos (padrão: reports/)")
    args = parser.parse_args(argv)
    if not (0 < args.fraction <= 1):
        parser.error("--fraction deve estar entre 0 e 1")
    if args.max_rows is not None and args.max_rows < 0:
        parser.error("--max-rows deve ser >= 0")
    train(args.data, fraction=args.fraction, max_rows=args.max_rows or None,
          maturity_filter=not args.no_maturity_filter, make_figures=not args.no_figures,
          model_path=args.model_path, report_dir=args.report_dir, missing_as_category=args.missing_as_category)


if __name__ == "__main__":
    main()
