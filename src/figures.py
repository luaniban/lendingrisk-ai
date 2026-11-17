"""Gráficos estáticos (PNG) gerados a partir dos números reais do treinamento.

Cada figura tem um CSV correspondente em reports/ com os mesmos valores, para
leitura sem depender de cor (acessibilidade) e para auditoria.
Paleta categórica validada para daltonismo (slots 1–3 da paleta de referência).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem interface gráfica — funciona em servidor/terminal
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from src.config import FEATURE_LABELS  # noqa: E402

# Mesmos tokens do tema da interface (.streamlit/config.toml): papel, tinta e grafite.
SURFACE = "#f7f5f0"   # papel (fundo do app)
INK = "#23262b"       # tinta
INK_2 = "#66645e"     # grafite
MUTED = "#8b887f"
GRID = "#e3ded3"
AXIS = "#cfc8b9"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
MODEL_COLORS = {"Regressão Logística": SLOTS[0], "Random Forest": SLOTS[1], "XGBoost": SLOTS[2]}
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
})


def _n(value: float, decimals: int = 3) -> str:
    """Número com vírgula decimal (pt-BR)."""
    return f"{value:.{decimals}f}".replace(".", ",")


def _decimal_comma(fig):
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    formatter = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(formatter)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    _decimal_comma(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _model_color(name: str, index: int) -> str:
    return MODEL_COLORS.get(name, SLOTS[index % len(SLOTS)])


def plot_pr_curves(curves: dict, prevalence: float, path: Path, title: str):
    """curves: {nome_modelo: (y, prob)}."""
    from sklearn.metrics import average_precision_score
    fig, ax = plt.subplots(figsize=(7, 4.8))
    upper = 0.0
    for i, (name, (y, p)) in enumerate(curves.items()):
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, color=_model_color(name, i), label=f"{name} (AP = {_n(ap)})")
        upper = max(upper, precision[recall >= 0.01].max())
    ax.axhline(prevalence, color=MUTED, linewidth=1)
    ax.annotate(f"classificador aleatório (AP ≈ {_n(prevalence)})", xy=(0.01, prevalence),
                xytext=(0, -4), textcoords="offset points", ha="left", va="top", color=INK_2, fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, min(1.0, upper * 1.25)), xlabel="Recall (Charged Off detectados)",
           ylabel="Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_roc_curves(curves: dict, path: Path, title: str):
    from sklearn.metrics import roc_auc_score
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (name, (y, p)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=_model_color(name, i), label=f"{name} (ROC-AUC = {_n(roc_auc_score(y, p))})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_calibration(tables: dict, path: Path, title: str):
    """tables: {nome: DataFrame de evaluation.calibration_table}."""
    fig, ax = plt.subplots(figsize=(6, 5))
    upper = 0.0
    for i, (name, table) in enumerate(tables.items()):
        ax.plot(table["prob_media_prevista"], table["taxa_observada"], marker="o",
                color=_model_color(name, i), label=name, markeredgecolor=SURFACE, markeredgewidth=1.5)
        upper = max(upper, table["prob_media_prevista"].max(), table["taxa_observada"].max())
    upper = min(1.0, upper * 1.1 + 0.02)
    ax.plot([0, upper], [0, upper], color=MUTED, linewidth=1)
    ax.annotate("calibração perfeita", xy=(upper * 0.97, upper * 0.97), ha="right", va="bottom",
                rotation=0, color=INK_2, fontsize=8, xytext=(0, 6), textcoords="offset points")
    ax.set(xlim=(0, upper), ylim=(0, upper), xlabel="Probabilidade média prevista (faixas por decil)",
           ylabel="Taxa observada de Charged Off")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_threshold_curve(curve: pd.DataFrame, chosen: float, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = [("precision", "Precision"), ("recall", "Recall"), ("f2", "F2")]
    for i, (col, label) in enumerate(series):
        ax.plot(curve["limiar"], curve[col], color=SLOTS[i], label=label)
    ax.axvline(chosen, color=INK_2, linewidth=1)
    ax.annotate(f"limiar escolhido = {_n(chosen, 2)}", xy=(chosen, 1.0), xytext=(4, -2),
                textcoords="offset points", va="top", color=INK, fontsize=9)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Limiar sobre a probabilidade estimada", ylabel="Valor")
    ax.set_title(title)
    ax.legend(loc="center right")
    _save(fig, path)


def plot_confusion_matrix(matrix, path: Path, title: str):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.grid(False)
    row_share = matrix / matrix.sum(axis=1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)
    labels = ["Fully Paid (0)", "Charged Off (1)"]
    for i in range(2):
        for j in range(2):
            color = "#ffffff" if row_share[i, j] > 0.55 else INK
            ax.text(j, i, f"{matrix[i, j]:,}".replace(",", ".") + f"\n{_n(row_share[i, j] * 100, 1)}% da linha",
                    ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks([0, 1], [f"Previsto\n{lab}" for lab in labels])
    ax.set_yticks([0, 1], [f"Real\n{lab}" for lab in labels])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    _save(fig, path)


def plot_score_distribution(y, p, threshold: float, path: Path, title: str):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    bins = np.linspace(0, max(0.05, p.max()), 41)
    for i, (cls, label) in enumerate([(0, "Fully Paid (real)"), (1, "Charged Off (real)")]):
        ax.hist(p[y == cls], bins=bins, density=True, histtype="step", linewidth=2, color=SLOTS[i], label=label)
    ax.axvline(threshold, color=INK_2, linewidth=1)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.12)
    ax.annotate(f"limiar = {_n(threshold, 2)}", xy=(threshold, top * 1.1), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="top", color=INK, fontsize=9)
    ax.set(xlabel="Probabilidade estimada de Charged Off", ylabel="Densidade")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_importance(importance: pd.DataFrame, path: Path, title: str):
    data = importance.sort_values("queda_ap_media")
    labels = [FEATURE_LABELS.get(a, a) for a in data["atributo"]]
    from matplotlib.ticker import MaxNLocator
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(labels, data["queda_ap_media"], xerr=data["desvio"], color=SLOTS[0], height=0.6,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.set(xlabel="Queda média da AP ao embaralhar o atributo (validação; barras = desvio-padrão)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    _save(fig, path)


def plot_stability(table: pd.DataFrame, path: Path, title: str):
    """table: colunas periodo, particao, average_precision, taxa_charged_off (modelo final)."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    x = np.arange(len(table))
    for ax, col, label in [(axes[0], "average_precision", "AP por mês"),
                           (axes[1], "taxa_charged_off", "Taxa real de Charged Off")]:
        for i, (part, group) in enumerate(table.groupby("particao", sort=False)):
            idx = x[table["particao"].to_numpy() == part]
            ax.plot(idx, group[col], marker="o", color=SLOTS[i], label=part,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_ylabel(label)
        ax.set_ylim(0, max(table[col].max() * 1.25, 0.05))
    axes[0].legend(loc="lower left")
    axes[1].set_xticks(x, table["periodo"], rotation=45, ha="right")
    axes[0].set_title(title)
    _save(fig, path)


def plot_cohorts(cohorts: pd.DataFrame, path: Path, title: str):
    """cohorts: ano, prazo, pct_finalizado, taxa_charged_off_finalizados."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for i, (term, group) in enumerate(cohorts.groupby("prazo")):
        group = group.sort_values("ano")
        axes[0].plot(group["ano"], group["pct_finalizado"] * 100, marker="o", color=SLOTS[i],
                     label=term.replace("months", "meses"), markeredgecolor=SURFACE, markeredgewidth=1.5)
        axes[1].plot(group["ano"], group["taxa_charged_off_finalizados"] * 100, marker="o", color=SLOTS[i],
                     label=term.replace("months", "meses"), markeredgecolor=SURFACE, markeredgewidth=1.5)
    axes[0].set(ylabel="% com desfecho final", ylim=(0, 105), xlabel="Ano de emissão")
    axes[0].set_title("Empréstimos já finalizados")
    axes[1].set(ylabel="% Charged Off entre finalizados", xlabel="Ano de emissão")
    axes[1].set_title("Taxa de Charged Off (só finalizados)")
    axes[0].legend(loc="lower left", title="Prazo")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    _save(fig, path)
