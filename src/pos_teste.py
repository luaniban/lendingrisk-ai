"""Análises complementares feitas DEPOIS de o teste reservado ter sido usado.

Nada aqui altera o modelo oficial, o limiar oficial ou reports/metricas.json. Como o
teste já foi visto, estes resultados são descritivos (análise de sensibilidade e
variante), e não substituem o resultado principal. Ver docs/experimentos.md, seção 5.

Uso:
  python -m src.pos_teste limiar --data data/lending_club/accepted_2007_to_2018Q4.csv
  python -m src.pos_teste comparar
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURES, METRICS_PATH, MODEL_PATH, REPORT_DIR
from src.data import load_dataset, sha256_file, temporal_split
from src.evaluation import THRESHOLD_GRID, choose_threshold, f_beta, scores, threshold_curve
from src.inference import load_metrics, load_model

OUT_DIR = REPORT_DIR / "analises_pos_teste"
VARIANT_DIR = OUT_DIR / "v2_ausente_como_categoria"
WARNING = ("Análise feita DEPOIS do uso do teste reservado. É descritiva e não substitui o resultado "
           "oficial (reports/metricas.json).")
FLAG_SHARE = 0.20  # política alternativa: sinalizar os 20% de maior risco (taxa fixada na validação)


# ----------------------------------------------------------------------------------------
# Sensibilidade do limiar
# ----------------------------------------------------------------------------------------
def threshold_options(y_val, p_val, official_threshold: float) -> list[dict]:
    """Limiares alternativos, todos definidos SOMENTE com a validação."""
    curve = threshold_curve(y_val, p_val, THRESHOLD_GRID)
    curve["f1"] = [f_beta(p, r, beta=1.0) for p, r in zip(curve["precision"], curve["recall"])]
    best_f1 = round(float(curve.loc[curve["f1"].idxmax(), "limiar"]), 3)
    top_share = round(float(np.quantile(np.asarray(p_val, dtype=float), 1 - FLAG_SHARE)), 4)
    return [
        {"criterio": "Máximo F2 (oficial)", "limiar": float(official_threshold),
         "descricao": "Prioriza recall: perder um Charged Off pesa mais que um alarme falso."},
        {"criterio": "Máximo F1", "limiar": best_f1,
         "descricao": "Equilibra precision e recall."},
        {"criterio": f"Sinalizar {round(FLAG_SHARE * 100)}% de maior risco", "limiar": top_share,
         "descricao": "Política de crédito com taxa de sinalização fixa (quantil dos scores da validação)."},
    ]


def sensitivity_table(y_val, p_val, y_test, p_test, options: list[dict]) -> pd.DataFrame:
    rows = []
    for option in options:
        for partition, y, p in (("validação", y_val, p_val), ("teste", y_test, p_test)):
            metrics = scores(y, p, option["limiar"])
            predicted = np.asarray(p) >= option["limiar"]
            actual = np.asarray(y).astype(int) == 1
            rows.append({
                "criterio": option["criterio"], "limiar": option["limiar"], "particao": partition,
                "taxa_sinalizada": metrics["taxa_positiva_prevista"], "recall": metrics["recall"],
                "precision": metrics["precision"], "f1": metrics["f1"], "f2": metrics["f2"],
                "acuracia": metrics["acuracia"],
                "verdadeiros_positivos": int((predicted & actual).sum()),
                "falsos_positivos": int((predicted & ~actual).sum()),
                "falsos_negativos": int((~predicted & actual).sum()),
                "verdadeiros_negativos": int((~predicted & ~actual).sum()),
            })
    return pd.DataFrame(rows)


def run_threshold_analysis(data_path: Path, out_dir: Path = OUT_DIR, metrics_path: Path = METRICS_PATH,
                           model_path: Path = MODEL_PATH, verbose: bool = True) -> dict:
    metrics = load_metrics(metrics_path)
    dataset = metrics["dataset"]
    sampling = dataset.get("amostragem", {})
    data_path = Path(data_path)
    sha = sha256_file(data_path)
    if dataset.get("sha256") and sha != dataset["sha256"]:
        raise ValueError("O arquivo de dados não é o mesmo do treino oficial (SHA-256 diferente).")
    if verbose:
        print("Recriando as partições do treino oficial (mesma amostragem, filtro e semente)...", flush=True)
    data, _ = load_dataset(data_path, sample_fraction=sampling.get("fracao", 1.0),
                           max_rows=sampling.get("max_linhas"),
                           maturity_filter=dataset.get("filtro_maturidade", {}).get("ativo", True),
                           seed=sampling.get("semente", metrics.get("semente")))
    _, val_set, test_set = temporal_split(data)
    for name, part in (("validacao", val_set), ("teste", test_set)):
        if len(part) != metrics["particoes"][name]["linhas"]:
            raise ValueError(f"A partição '{name}' recriada não bate com o treino oficial.")

    model = load_model(model_path)
    y_val, y_test = val_set["target"].to_numpy(), test_set["target"].to_numpy()
    p_val = model.predict_proba(val_set[FEATURES])[:, 1]
    p_test = model.predict_proba(test_set[FEATURES])[:, 1]

    # Conferências: as previsões reproduzem o limiar e as métricas oficiais.
    official = float(metrics["limiar"])
    reproduced = {
        "limiar_f2_validacao": choose_threshold(y_val, p_val),
        "ap_teste": scores(y_test, p_test, official)["average_precision"],
    }
    checks = {
        "limiar_reproduzido": abs(reproduced["limiar_f2_validacao"] - official) < 1e-9,
        "ap_teste_reproduzida": abs(reproduced["ap_teste"] - metrics["metricas_teste"]["average_precision"]) < 1e-5,
    }
    if not all(checks.values()):
        raise ValueError(f"As previsões não reproduzem o treino oficial: {reproduced}")

    options = threshold_options(y_val, p_val, official)
    table = sensitivity_table(y_val, p_val, y_test, p_test, options)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "sensibilidade_limiar.csv", index=False)
    payload = {
        "aviso": WARNING, "modelo": metrics["modelo_escolhido"], "limiar_oficial": official,
        "dataset_sha256": sha, "conferencias": checks, "criterios": options,
        "resultados": table.to_dict(orient="records"),
        "criado_em_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "sensibilidade_limiar.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    if verbose:
        print(table[table["particao"] == "teste"][["criterio", "limiar", "taxa_sinalizada", "recall",
                                                    "precision", "acuracia"]].to_string(index=False))
    return payload


# ----------------------------------------------------------------------------------------
# Comparação oficial (v1) x variante (v2)
# ----------------------------------------------------------------------------------------
COMPARED_METRICS = ["average_precision", "roc_auc", "brier", "ece", "recall", "precision", "f2",
                    "acuracia", "taxa_positiva_prevista"]


def _missing_subgroup(report_dir: Path, column: str = "emp_length") -> dict:
    table = pd.read_csv(report_dir / "analise_subgrupos_teste.csv")
    row = table[(table["atributo"] == column) & (table["valor"] == "ausente")]
    if row.empty:
        return {}
    row = row.iloc[0]
    return {"linhas": int(row["linhas"]), "taxa_charged_off_real": float(row["taxa_charged_off"]),
            "score_medio_previsto": float(row["score_medio"]), "recall": float(row["recall"]),
            "precision": float(row["precision"])}


def compare_variants(official_dir: Path = REPORT_DIR, variant_dir: Path = VARIANT_DIR,
                     out_dir: Path = OUT_DIR) -> dict:
    versions = {"v1_oficial_moda": official_dir, "v2_ausente_como_categoria": variant_dir}
    loaded = {name: json.loads((path / METRICS_PATH.name).read_text(encoding="utf-8"))
              for name, path in versions.items()}
    rows = []
    for name, metrics in loaded.items():
        rows.append({"versao": name, "modelo_escolhido": metrics["modelo_escolhido"], "limiar": metrics["limiar"],
                     **{f"teste_{key}": metrics["metricas_teste"].get(key) for key in COMPARED_METRICS},
                     **{f"emp_length_ausente_{key}": value
                        for key, value in _missing_subgroup(versions[name]).items()}})
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparacao_v1_v2.csv", index=False)
    payload = {"aviso": WARNING, "versoes": {k: str(v) for k, v in versions.items()},
               "comparacao": table.to_dict(orient="records"),
               "criado_em_utc": datetime.now(timezone.utc).isoformat()}
    (out_dir / "comparacao_v1_v2.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Análises complementares pós-teste (não alteram o oficial)")
    commands = parser.add_subparsers(dest="comando", required=True)
    threshold = commands.add_parser("limiar", help="Sensibilidade do limiar (validação e teste)")
    threshold.add_argument("--data", required=True, type=Path)
    commands.add_parser("comparar", help="Compara o oficial (v1) com a variante v2")
    args = parser.parse_args(argv)
    if args.comando == "limiar":
        run_threshold_analysis(args.data)
    else:
        result = compare_variants()
        print(pd.DataFrame(result["comparacao"]).T.to_string())


if __name__ == "__main__":
    main()
