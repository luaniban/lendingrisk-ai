"""Leitura, limpeza, filtros e divisão temporal do Lending Club (sem leakage).

Decisões documentadas em docs/experimentos.md:
- amostragem uniforme por linha (Bernoulli) — o sorteio é feito para TODAS as
  linhas do arquivo, então a amostra não depende da ordem do arquivo nem do
  tamanho dos blocos com desfecho final;
- filtro de maturidade: mantém apenas empréstimos cujo prazo contratual já
  terminou na data de referência do arquivo, reduzindo o viés de seleção de
  quem "terminou cedo" (quitação antecipada ou baixa precoce);
- datas lidas sem depender do locale do sistema ("Dec-2015" -> 2015-12-01).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES, CLASS_MAP, DATE_COLUMN, DEFAULT_MAX_ROWS,
    DEFAULT_SAMPLE_FRACTION, FEATURES, NUMERIC_FEATURES, SEED, TARGET,
)

CHUNK_SIZE = 100_000
REQUIRED_COLUMNS = FEATURES + [TARGET, DATE_COLUMN]
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1
)}


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def check_columns(path: Path, required: list[str] | None = None) -> list[str]:
    """Confere o cabeçalho e devolve a lista de colunas do arquivo."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {path}. Veja a seção 'Dataset' do README.md."
        )
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [col for col in (required or REQUIRED_COLUMNS) if col not in header]
    if missing:
        raise ValueError(
            f"O CSV não tem as colunas esperadas: {missing}. "
            "Confirme se é o arquivo de empréstimos ACEITOS do Lending Club."
        )
    return header


def parse_issue_date(values: pd.Series) -> pd.Series:
    """Converte 'Dec-2015' (ou '2015-12-01') para o 1º dia do mês; inválidos viram NaT."""
    text = values.astype("string").str.strip()
    parts = text.str.extract(r"^([A-Za-z]{3})-(\d{4})$")
    month = parts[0].str.title().map(_MONTHS).astype("float")
    year = pd.to_numeric(parts[1], errors="coerce")
    valid = month.notna() & year.notna()
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if valid.any():
        result.loc[valid] = pd.to_datetime(
            pd.DataFrame({"year": year[valid], "month": month[valid], "day": 1})
        )
    iso = ~valid & text.str.match(r"^\d{4}-\d{2}(-\d{2})?$", na=False)
    if iso.any():
        parsed = pd.to_datetime(text[iso].str.slice(0, 7) + "-01", format="%Y-%m-%d", errors="coerce")
        result.loc[iso] = parsed
    return result


def parse_numeric(values: pd.Series) -> pd.Series:
    """Aceita números com separador de milhar ',' ou com símbolo '%'."""
    cleaned = (
        values.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def parse_categorical(values: pd.Series) -> pd.Series:
    """Remove espaços (ex.: ' 36 months') e usa np.nan para ausentes."""
    text = values.astype("string").str.strip().replace({"": pd.NA, "n/a": pd.NA})
    return text.astype(object).where(text.notna(), np.nan)


def term_months(values: pd.Series) -> pd.Series:
    """'36 months' -> 36.0; formatos desconhecidos viram NaN."""
    return pd.to_numeric(
        values.astype("string").str.extract(r"(\d+)")[0], errors="coerce"
    ).astype(float)


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in NUMERIC_FEATURES:
        out[col] = parse_numeric(frame[col])
    for col in CATEGORICAL_FEATURES:
        out[col] = parse_categorical(frame[col])
    return out


def maturity_mask(issue_dates: pd.Series, terms: pd.Series, reference: pd.Timestamp) -> pd.Series:
    """True quando emissão + prazo contratual <= data de referência (prazo já encerrado)."""
    months = term_months(terms)
    known = months.notna() & issue_dates.notna()
    end_index = issue_dates.dt.year * 12 + issue_dates.dt.month - 1 + months
    reference_index = reference.year * 12 + reference.month - 1
    return (known & (end_index <= reference_index)).astype(bool)


def load_dataset(
    path: Path,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    maturity_filter: bool = True,
    reference_date: pd.Timestamp | None = None,
    seed: int = SEED,
    chunksize: int = CHUNK_SIZE,
) -> tuple[pd.DataFrame, dict]:
    """Lê só as colunas autorizadas, em blocos, e devolve (dados, registro_de_auditoria).

    O registro contém contagens de todas as etapas para documentação/reprodutibilidade.
    """
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction deve estar entre 0 (exclusivo) e 1.")
    path = Path(path)
    check_columns(path)

    rng = np.random.default_rng(seed)
    status_counts: Counter = Counter()
    rows_read = 0
    invalid_dates_final = 0
    date_min, date_max = None, None
    kept = []
    for chunk in pd.read_csv(path, usecols=REQUIRED_COLUMNS, chunksize=chunksize, dtype=str):
        rows_read += len(chunk)
        draw = rng.random(len(chunk))  # sorteio para todas as linhas lidas
        status = chunk[TARGET].astype("string").str.strip()
        status_counts.update(status.fillna("<vazio>").value_counts().to_dict())
        dates = parse_issue_date(chunk[DATE_COLUMN])
        if dates.notna().any():
            chunk_min, chunk_max = dates.min(), dates.max()
            date_min = chunk_min if date_min is None else min(date_min, chunk_min)
            date_max = chunk_max if date_max is None else max(date_max, chunk_max)
        final = status.isin(list(CLASS_MAP)).fillna(False).to_numpy(dtype=bool)
        invalid_dates_final += int((final & dates.isna().to_numpy()).sum())
        mask = final & dates.notna().to_numpy() & (draw < sample_fraction)
        if not mask.any():
            continue
        selected = chunk.loc[mask]
        cleaned = clean_features(selected)
        cleaned[TARGET] = status[mask].astype(str).to_numpy()
        cleaned[DATE_COLUMN] = dates[mask].to_numpy()
        kept.append(cleaned)

    final_total = sum(status_counts.get(label, 0) for label in CLASS_MAP)
    if not kept:
        raise ValueError("Nenhuma linha com 'Fully Paid' ou 'Charged Off' (e data válida) foi encontrada.")
    data = pd.concat(kept, ignore_index=True)
    sampled_rows = len(data)

    maturity = {"ativo": bool(maturity_filter)}
    if maturity_filter:
        reference = pd.Timestamp(reference_date) if reference_date is not None else date_max
        matured = maturity_mask(data[DATE_COLUMN], data["term"], reference)
        maturity.update({
            "data_referencia": str(reference.date()),
            "criterio": "issue_d + prazo (term) <= data de referência (maior issue_d do arquivo)",
            "linhas_removidas": int((~matured).sum()),
        })
        data = data.loc[matured].reset_index(drop=True)

    capped = False
    if max_rows is not None and len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        capped = True

    data["target"] = data[TARGET].map(CLASS_MAP).astype(int)
    if len(data) < 200 or data["target"].nunique() != 2:
        raise ValueError("Dataset insuficiente para treinar as duas classes.")
    data = data.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)

    info = {
        "arquivo": path.name,
        "tamanho_bytes": path.stat().st_size,
        "linhas_lidas": rows_read,
        "contagem_status": dict(sorted(status_counts.items(), key=lambda kv: -kv[1])),
        "linhas_com_desfecho_final": int(final_total),
        "datas_invalidas_entre_finalizados": invalid_dates_final,
        "periodo_emissao_arquivo": [
            str(date_min.date()) if date_min is not None else None,
            str(date_max.date()) if date_max is not None else None,
        ],
        "amostragem": {
            "metodo": "Bernoulli por linha (uniforme) + teto aleatório",
            "fracao": sample_fraction,
            "max_linhas": max_rows,
            "semente": seed,
            "linhas_finalizadas_sorteadas": sampled_rows,
            "teto_aplicado": capped,
        },
        "filtro_maturidade": maturity,
        "linhas_utilizadas": len(data),
        "taxa_charged_off": round(float(data["target"].mean()), 5),
    }
    return data, info


def temporal_split(data: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Treino ~70%, validação ~15%, teste ~15%, por ordem de issue_d; meses não cruzam partições."""
    if not data[DATE_COLUMN].is_monotonic_increasing:
        data = data.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)
    n = len(data)
    cutoff_train = data.iloc[int(n * train_frac)][DATE_COLUMN]
    cutoff_val = data.iloc[int(n * (train_frac + val_frac))][DATE_COLUMN]
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


def describe_partition(subset: pd.DataFrame) -> dict:
    return {
        "linhas": int(len(subset)),
        "inicio": str(subset[DATE_COLUMN].min().date()),
        "fim": str(subset[DATE_COLUMN].max().date()),
        "taxa_charged_off": round(float(subset["target"].mean()), 5),
        "proporcao_60_meses": round(float((subset["term"] == "60 months").mean()), 5),
    }
