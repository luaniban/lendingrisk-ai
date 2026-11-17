"""Gerador de dados SINTÉTICOS para os testes automatizados.

ATENÇÃO: estes dados NÃO são do Lending Club. Imitam apenas o FORMATO do
arquivo (nomes de colunas, textos como ' 36 months', 'Dec-2015', '45.3%',
status não finais, linhas de rodapé) para testar o código. Nenhuma métrica
obtida com eles tem valor científico. Os arquivos são gravados apenas em
diretórios temporários — nunca em data/ ou reports/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "car"]
HOMES = ["RENT", "MORTGAGE", "OWN"]
EMP = ["< 1 year", "1 year", "3 years", "5 years", "10+ years", None]


def month_label(year: int, month: int) -> str:
    return f"{MONTHS[month - 1]}-{year}"


def make_synthetic_frame(n: int = 3000, seed: int = 0, start_year: int = 2010, months: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, months, n)
    years = start_year + offsets // 12
    month_numbers = offsets % 12 + 1
    term60 = rng.random(n) < 0.3
    fico = rng.integers(132, 170, n) * 5  # 660–845
    dti = rng.uniform(0, 40, n)
    revol = rng.uniform(0, 100, n)
    logit = -1.6 + 0.05 * (dti - 18) - 0.025 * (fico - 700) + 0.8 * term60 + 0.01 * (revol - 50)
    charged = rng.random(n) < 1 / (1 + np.exp(-logit))
    status = np.where(charged, "Charged Off", "Fully Paid").astype(object)
    # alguns status não finais / fora da política, que devem ser descartados
    other = rng.random(n)
    status[other < 0.05] = "Current"
    status[(other >= 0.05) & (other < 0.07)] = "Late (31-120 days)"
    status[(other >= 0.07) & (other < 0.08)] = "Does not meet the credit policy. Status:Fully Paid"
    revol_text = [f"{v:.1f}%" if i % 3 == 0 else f"{v:.1f}" for i, v in enumerate(revol)]
    frame = pd.DataFrame({
        "id": np.arange(n),
        "loan_amnt": rng.integers(2, 70, n) * 500.0,
        "term": np.where(term60, " 60 months", " 36 months"),
        "annual_inc": np.round(rng.lognormal(11, 0.5, n), 0),
        "dti": np.round(dti, 2),
        "fico_range_low": fico,
        "inq_last_6mths": rng.poisson(1.0, n),
        "revol_util": revol_text,
        "purpose": rng.choice(PURPOSES, n),
        "home_ownership": rng.choice(HOMES, n),
        "emp_length": rng.choice(np.array(EMP, dtype=object), n),
        "issue_d": [month_label(y, m) for y, m in zip(years, month_numbers)],
        "loan_status": status,
        # colunas pós-concessão presentes no arquivo real — nunca podem ser usadas
        "total_pymnt": rng.uniform(0, 40000, n),
        "last_fico_range_high": rng.integers(500, 850, n),
    })
    return frame


def write_synthetic_csv(path: Path, n: int = 3000, seed: int = 0, footer: bool = True, **kwargs) -> Path:
    """Grava CSV (ou CSV.GZ pela extensão) com linhas de rodapé como no arquivo original."""
    frame = make_synthetic_frame(n=n, seed=seed, **kwargs)
    if footer:
        junk = pd.DataFrame({"id": ["Total amount funded in policy code 1: 1234", "Total amount funded in policy code 2: 99"]})
        frame = pd.concat([frame, junk], ignore_index=True)
    path = Path(path)
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)
    return path
