"""Auditoria do dataset real, SEM treinar modelos.

Uso: python -m src.audit --data data/lending_club/accepted_2007_to_2018Q4.csv

Mede: colunas, linhas, status de loan_status, período de emissão, valores
ausentes, coortes por ano/prazo (quantos já têm desfecho e a taxa de Charged
Off) e o efeito do filtro de maturidade nas partições temporais.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES, CLASS_MAP, DATE_COLUMN, FEATURES, LEAKAGE_COLUMNS,
    NUMERIC_FEATURES, REPORT_DIR, TARGET,
)
from src.data import (
    CHUNK_SIZE, REQUIRED_COLUMNS, check_columns, clean_features, describe_partition,
    maturity_mask, parse_issue_date, sha256_file, temporal_split,
)


def _numeric_summary(values: pd.Series) -> dict:
    values = values.dropna()
    q = values.quantile([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]).to_numpy()
    return {"min": q[0], "p01": q[1], "p25": q[2], "mediana": q[3], "p75": q[4], "p99": q[5], "max": q[6],
            "media": float(values.mean())}


def audit(path: Path, report_dir: Path = REPORT_DIR, chunksize: int = CHUNK_SIZE,
          make_figures: bool = True, compute_hash: bool = True) -> dict:
    path = Path(path)
    report_dir = Path(report_dir)
    header = check_columns(path)

    status_counts: Counter = Counter()
    rows_read, invalid_dates = 0, 0
    cohort_parts, final_parts = [], []
    for chunk in pd.read_csv(path, usecols=REQUIRED_COLUMNS, chunksize=chunksize, dtype=str):
        rows_read += len(chunk)
        status = chunk[TARGET].astype("string").str.strip()
        status_counts.update(status.fillna("<vazio>").value_counts().to_dict())
        dates = parse_issue_date(chunk[DATE_COLUMN])
        invalid_dates += int(dates.isna().sum())
        term = chunk["term"].astype("string").str.strip()
        final = status.isin(list(CLASS_MAP)).fillna(False)
        cohort_parts.append(pd.DataFrame({
            DATE_COLUMN: dates, "term": term, "final": final.to_numpy(dtype=bool),
            "charged_off": (status == "Charged Off").fillna(False).to_numpy(dtype=bool),
        }))
        selected = final.to_numpy(dtype=bool) & dates.notna().to_numpy()
        if selected.any():
            part = clean_features(chunk.loc[selected])
            part[DATE_COLUMN] = dates[selected].to_numpy()
            part["target"] = status[selected].map(CLASS_MAP).astype(int).to_numpy()
            final_parts.append(part)

    cohorts_raw = pd.concat(cohort_parts, ignore_index=True)
    finals = pd.concat(final_parts, ignore_index=True)
    reference = cohorts_raw[DATE_COLUMN].max()

    # ---- Coortes (ano de emissão x prazo) ----
    valid = cohorts_raw.dropna(subset=[DATE_COLUMN]).copy()
    valid["ano"] = valid[DATE_COLUMN].dt.year
    valid["maturado"] = maturity_mask(valid[DATE_COLUMN], valid["term"], reference)
    cohorts = valid.groupby(["ano", "term"]).agg(
        total=("final", "size"), finalizados=("final", "sum"), charged_off=("charged_off", "sum"),
        maturados=("maturado", "sum"),
    ).reset_index().rename(columns={"term": "prazo"})
    cohorts["pct_finalizado"] = (cohorts["finalizados"] / cohorts["total"]).round(5)
    cohorts["taxa_charged_off_finalizados"] = (cohorts["charged_off"] / cohorts["finalizados"]).round(5)
    cohorts["pct_maturado"] = (cohorts["maturados"] / cohorts["total"]).round(5)

    def _summary(mask: pd.Series) -> dict:
        subset = valid[mask]
        finals_n = int(subset["final"].sum())
        return {
            "emprestimos": int(len(subset)), "com_desfecho_final": finals_n,
            "pct_com_desfecho_final": round(finals_n / len(subset), 5) if len(subset) else None,
            "taxa_charged_off_entre_finalizados": round(float(subset["charged_off"].sum()) / finals_n, 5)
            if finals_n else None,
        }

    maturity = {
        "data_referencia": str(reference.date()),
        "criterio": "issue_d + prazo <= data de referência (maior issue_d do arquivo)",
        "coortes_maturadas": _summary(valid["maturado"]),
        "coortes_nao_maturadas": _summary(~valid["maturado"]),
    }

    # ---- Partições temporais com TODOS os finalizados (sem amostragem) ----
    finals = finals.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)
    policies = {}
    for label, frame in [
        ("sem_filtro_maturidade (starter)", finals),
        ("com_filtro_maturidade", finals[maturity_mask(finals[DATE_COLUMN], finals["term"], reference)]),
    ]:
        frame = frame.reset_index(drop=True)
        train, val, test = temporal_split(frame)
        policies[label] = {
            "linhas": int(len(frame)), "taxa_charged_off": round(float(frame["target"].mean()), 5),
            "treino": describe_partition(train), "validacao": describe_partition(val),
            "teste": describe_partition(test),
        }

    result = {
        "arquivo": path.name,
        "tamanho_bytes": path.stat().st_size,
        "sha256": sha256_file(path) if compute_hash else None,
        "colunas_no_arquivo": len(header),
        "colunas_usadas": REQUIRED_COLUMNS,
        "colunas_de_vazamento_presentes_no_arquivo": sorted(c for c in LEAKAGE_COLUMNS if c in header),
        "features_com_vazamento": sorted(set(FEATURES) & LEAKAGE_COLUMNS),
        "linhas_lidas": rows_read,
        "datas_invalidas_ou_ausentes": invalid_dates,
        "periodo_emissao": [str(valid[DATE_COLUMN].min().date()), str(reference.date())],
        "contagem_status": dict(sorted(status_counts.items(), key=lambda kv: -kv[1])),
        "finalizados": {
            "linhas": int(len(finals)),
            "fully_paid": int((finals["target"] == 0).sum()),
            "charged_off": int((finals["target"] == 1).sum()),
            "taxa_charged_off": round(float(finals["target"].mean()), 5),
        },
        "ausentes_entre_finalizados": {c: round(float(finals[c].isna().mean()), 5) for c in FEATURES},
        "resumo_numerico_finalizados": {
            c: {k: round(float(v), 4) for k, v in _numeric_summary(finals[c]).items()} for c in NUMERIC_FEATURES
        },
        "categorias_finalizados": {
            c: {str(k): int(v) for k, v in finals[c].value_counts(dropna=False).items()}
            for c in CATEGORICAL_FEATURES
        },
        "maturidade": maturity,
        "particoes_por_politica": policies,
        "criado_em_utc": datetime.now(timezone.utc).isoformat(),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "auditoria_dataset.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    cohorts.to_csv(report_dir / "auditoria_coortes.csv", index=False)
    if make_figures:
        from src import figures
        figures.plot_cohorts(cohorts, report_dir / "figures" / "auditoria_coortes.png",
                             "Coortes por ano de emissão e prazo (dataset completo)")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auditoria do Lending Club (sem treino)")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    result = audit(args.data, make_figures=not args.no_figures)
    print(json.dumps({k: result[k] for k in [
        "arquivo", "sha256", "colunas_no_arquivo", "linhas_lidas", "periodo_emissao",
        "contagem_status", "finalizados", "maturidade", "particoes_por_politica",
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
