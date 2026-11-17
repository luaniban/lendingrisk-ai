"""Interface de demonstração do LendingRisk AI (projeto final CARCDSI, IFSP Caraguatatuba).

Executar a partir da raiz do projeto:  python -m streamlit run app.py

Layout de "dossiê de crédito": a proposta (formulário) fica ao lado do parecer
(resultado). O tema (cores e fontes) está em .streamlit/config.toml; os poucos
elementos próprios (régua de risco, carimbo, ficha técnica) usam o CSS de STYLE.
"""
from __future__ import annotations

import html
import math
import time

import pandas as pd
import streamlit as st

from src import config
from src.config import FEATURE_LABELS, NUMERIC_FEATURES
from src.inference import (
    MISSING_LABEL, ArtefatosInvalidos, category_label, format_record_pt, load_metrics,
    load_model, percent_pt, predict_probability, validate_record, version_warnings,
)
from src.llm import (
    CLOSING_SENTENCE, LLMIndisponivel, check_ollama, explain_checked,
)

st.set_page_config(page_title="LendingRisk AI", page_icon=":material/account_balance:", layout="wide")

METRIC_INFO = {
    "average_precision": ("Average Precision (AP)", "Resumo da curva precisão × recall. Um modelo aleatório teria AP ≈ taxa de Charged Off."),
    "roc_auc": ("ROC-AUC", "Chance de um Charged Off receber score maior que um Fully Paid (0,5 = aleatório)."),
    "brier": ("Brier score", "Erro quadrático médio da probabilidade (menor é melhor)."),
    "log_loss": ("Log loss", "Penaliza probabilidades confiantes e erradas (menor é melhor)."),
    "ece": ("ECE (erro de calibração)", "Diferença média entre probabilidade prevista e taxa observada (menor é melhor)."),
    "recall": ("Recall", "Fração dos Charged Off reais que o modelo sinalizou acima do limiar."),
    "precision": ("Precision", "Fração dos sinalizados que de fato terminaram em Charged Off."),
    "f1": ("F1", "Média harmônica de precision e recall."),
    "f2": ("F2", "Como o F1, mas dá mais peso ao recall (critério do limiar)."),
    "acuracia": ("Acurácia", "Acertos totais. Enganosa com classes desbalanceadas."),
    "taxa_positiva_real": ("Taxa real de Charged Off", "Proporção de Charged Off no conjunto."),
    "taxa_positiva_prevista": ("Taxa sinalizada pelo modelo", "Proporção de casos acima do limiar."),
}
# Rótulos curtos no formulário; a descrição completa fica no "?" de cada campo.
FORM_LABELS = {
    "loan_amnt": "Valor solicitado (US$)", "term": "Prazo",
    "annual_inc": "Renda anual (US$)", "dti": "Dívida/renda, DTI (%)",
    "fico_range_low": "Score FICO", "inq_last_6mths": "Consultas em 6 meses",
    "revol_util": "Uso do rotativo (%)", "purpose": "Finalidade",
    "home_ownership": "Moradia", "emp_length": "Tempo de emprego",
}
FIELD_HELP = {
    "loan_amnt": "Valor do empréstimo solicitado, em dólares americanos (USD).",
    "annual_inc": "Renda ANUAL declarada pelo solicitante, em USD (não é a renda mensal).",
    "dti": "Dívidas mensais (exceto hipoteca e o próprio empréstimo) divididas pela renda mensal, em %.",
    "fico_range_low": "Limite inferior da faixa de score FICO (escala de 300 a 850) no momento do pedido.",
    "inq_last_6mths": "Quantidade de consultas de crédito feitas nos últimos 6 meses.",
    "revol_util": "Percentual do limite de crédito rotativo (ex.: cartão) em uso. Pode passar de 100%.",
    "term": "Prazo contratual do empréstimo.",
    "purpose": "Finalidade declarada do empréstimo.",
    "home_ownership": "Situação de moradia declarada.",
    "emp_length": "Tempo no emprego atual. 'Não informado' usa a imputação aprendida no treino.",
}
STEPS = {"loan_amnt": 500.0, "annual_inc": 1000.0, "dti": 0.5, "fico_range_low": 5,
         "inq_last_6mths": 1, "revol_util": 1.0}
INTEGER_FIELDS = {"fico_range_low", "inq_last_6mths"}
FORMATS = {"loan_amnt": "%.0f", "annual_inc": "%.0f", "dti": "%.2f", "revol_util": "%.1f"}
FORM_ROWS = [("loan_amnt", "term"), ("annual_inc", "dti"), ("fico_range_low", "inq_last_6mths"),
             ("revol_util", "purpose"), ("home_ownership", "emp_length")]
FIGURE_CAPTIONS = {
    "figures/curvas_pr_validacao.png": "Curvas precisão × recall dos três modelos na validação (critério de seleção: AP).",
    "figures/curvas_roc_validacao.png": "Curvas ROC dos três modelos na validação.",
    "figures/calibracao_validacao.png": "Calibração na validação: pontos sobre a diagonal indicam probabilidade bem calibrada.",
    "figures/limiar_validacao.png": "Precision, recall e F2 por limiar na validação; a linha vertical é o limiar escolhido.",
    "figures/matriz_confusao_teste.png": "Matriz de confusão no teste reservado (avaliado uma única vez).",
    "figures/distribuicao_scores_teste.png": "Distribuição das probabilidades no teste, separada pela classe real.",
    "figures/importancia_atributos.png": "Importância global por permutação, medida na validação. Não explica casos individuais.",
    "figures/estabilidade_temporal.png": "AP e taxa real de Charged Off por mês de emissão (validação e teste).",
    "figures/calibracao_teste.png": "Calibração do modelo escolhido no teste.",
}
PARTITION_LABELS = {"treino": "Treino", "validacao": "Validação", "teste": "Teste"}
MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

# Paleta igual à de .streamlit/config.toml. Bordas finas e quentes, sem sombras.
STYLE = """
<style>
:root {
  --lr-paper: #f7f5f0; --lr-paper-2: #ece8df; --lr-ink: #23262b; --lr-graphite: #66645e;
  --lr-faint: #8b887f; --lr-rule: #d9d3c6; --lr-pen: #1f4e8c; --lr-stamp: #a8322a; --lr-ledger: #2e6b41;
  --lr-serif: "Source Serif", "Source Serif 4", Georgia, serif;
}
[data-testid="stMainBlockContainer"] { padding-top: 4rem; }
.lr-eyebrow { font-size: .72rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase;
  color: var(--lr-graphite); margin-bottom: .4rem; }
.lr-masthead { display: flex; justify-content: space-between; align-items: flex-end; gap: .5rem 1.5rem;
  flex-wrap: wrap; padding-bottom: .9rem; border-bottom: 1px solid var(--lr-rule); }
.lr-title { font-family: var(--lr-serif); font-size: 2rem; font-weight: 600; line-height: 1.1;
  letter-spacing: -.015em; color: var(--lr-ink); }
.lr-byline { font-size: .8rem; line-height: 1.45; color: var(--lr-graphite); }
.lr-lead { font-size: .95rem; color: var(--lr-graphite); margin-bottom: .75rem; text-wrap: pretty; }
.lr-hero-label { font-size: .9rem; color: var(--lr-graphite); }
.lr-hero { font-family: var(--lr-serif); font-size: 3.6rem; font-weight: 600; line-height: 1;
  letter-spacing: -.02em; font-variant-numeric: lining-nums tabular-nums; color: var(--lr-ink); margin: .3rem 0 .9rem; }
.lr-verdict { font-size: 1rem; color: var(--lr-ink); margin-bottom: .35rem; }
.lr-stamp { display: inline-block; margin-left: .4rem; padding: .1rem .5rem; border: 2px solid currentColor;
  border-radius: 3px; font-size: .78rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }
.lr-stamp.co { color: var(--lr-stamp); }
.lr-stamp.fp { color: var(--lr-ledger); }
.lr-text { font-size: .9rem; line-height: 1.55; color: var(--lr-graphite); text-wrap: pretty; }
.lr-ruler { margin: 2.1rem 0 1.1rem; }
.lr-track { position: relative; height: 8px; border-radius: 2px; background: var(--lr-paper-2);
  box-shadow: inset 0 0 0 1px var(--lr-rule); }
.lr-zone { position: absolute; top: 0; bottom: 0; right: 0; border-radius: 0 2px 2px 0;
  background: repeating-linear-gradient(135deg, rgba(168, 50, 42, .38) 0 2px, transparent 2px 6px); }
.lr-cut { position: absolute; top: -14px; bottom: -6px; width: 2px; margin-left: -1px; background: var(--lr-ink); }
.lr-cut span { position: absolute; bottom: 100%; left: 5px; margin-bottom: -.2rem; white-space: nowrap;
  font-size: .72rem; font-weight: 600; color: var(--lr-ink); }
.lr-avg { position: absolute; top: -5px; bottom: -5px; margin-left: -1px; border-left: 2px dotted var(--lr-graphite); }
.lr-mark { position: absolute; top: 50%; width: 16px; height: 16px; border-radius: 50%;
  transform: translate(-50%, -50%); box-shadow: 0 0 0 2px var(--lr-paper); }
.lr-mark.co { background: var(--lr-stamp); }
.lr-mark.fp { background: var(--lr-ledger); }
.lr-ticks { position: relative; height: 1rem; margin-top: .4rem; font-size: .7rem; color: var(--lr-faint);
  font-variant-numeric: tabular-nums; }
.lr-ticks span { position: absolute; transform: translateX(-50%); }
.lr-ticks span:first-child { transform: none; }
.lr-ticks span:last-child { transform: translateX(-100%); }
.lr-legend { display: flex; flex-wrap: wrap; gap: .3rem 1.1rem; margin-top: .7rem; font-size: .75rem;
  color: var(--lr-graphite); }
.lr-legend i { display: inline-block; margin-right: .4rem; vertical-align: -1px; }
.lr-sw-mark { width: 10px; height: 10px; border-radius: 50%; }
.lr-sw-mark.co { background: var(--lr-stamp); }
.lr-sw-mark.fp { background: var(--lr-ledger); }
.lr-sw-cut { width: 2px; height: 12px; background: var(--lr-ink); }
.lr-sw-avg { width: 0; height: 12px; border-left: 2px dotted var(--lr-graphite); }
.lr-sw-zone { width: 14px; height: 10px; box-shadow: inset 0 0 0 1px var(--lr-rule);
  background: repeating-linear-gradient(135deg, rgba(168, 50, 42, .5) 0 2px, transparent 2px 5px); }
.lr-aside { border-left: 2px solid var(--lr-rule); padding: .15rem 0 .15rem .8rem; margin: .9rem 0;
  font-size: .85rem; line-height: 1.5; color: var(--lr-graphite); }
.lr-section { margin-top: 1.9rem; padding-top: 1.1rem; border-top: 1px solid var(--lr-rule); }
.lr-rank { margin: 0 0 .5rem 1.15rem; padding: 0; font-size: .92rem; line-height: 1.6; color: var(--lr-ink); }
.lr-memo { margin: .4rem 0 .25rem; padding: .2rem 0 .2rem 1rem; border-left: 2px solid var(--lr-pen);
  font-family: var(--lr-serif); font-size: 1.06rem; line-height: 1.62; color: var(--lr-ink); text-wrap: pretty; }
.lr-memo-meta { font-size: .75rem; line-height: 1.5; color: var(--lr-graphite); padding-left: 1.1rem; }
.lr-empty { margin-top: 1.4rem; padding: 2.4rem 1.5rem; border: 1px dashed var(--lr-rule); border-radius: 3px;
  text-align: center; font-family: var(--lr-serif); font-style: italic; font-size: 1.02rem; color: var(--lr-graphite); }
.lr-ficha dl { display: grid; grid-template-columns: auto 1fr; gap: .4rem .9rem; margin: 0; font-size: .86rem; }
.lr-ficha dt { color: var(--lr-graphite); }
.lr-ficha dd { margin: 0; color: var(--lr-ink); font-variant-numeric: tabular-nums; }
.lr-dot { display: inline-block; width: 7px; height: 7px; margin-right: .35rem; border-radius: 50%; vertical-align: 1px; }
.lr-dot.ok { background: var(--lr-ledger); }
.lr-dot.off { background: var(--lr-stamp); }
.lr-footnote { font-size: .75rem; line-height: 1.5; color: var(--lr-faint); }
</style>
"""


def _mtime(path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def cached_metrics(path: str, _mtime_key: float) -> dict:
    return load_metrics(path)


@st.cache_resource(show_spinner=False)
def cached_model(path: str, _mtime_key: float):
    return load_model(path)


@st.cache_data(ttl=15, show_spinner=False)
def cached_ollama_status() -> dict:
    return check_ollama()


def int_pt(value) -> str:
    return f"{value:,}".replace(",", ".") if isinstance(value, int) else "?"


def month_pt(iso_date: str | None) -> str:
    try:
        year, month = str(iso_date)[:7].split("-")
        return f"{MONTHS[int(month) - 1]}/{year}"
    except (ValueError, IndexError):
        return "?"


def html_block(markup: str):
    """HTML próprio numa linha só: o Markdown do Streamlit trataria recuos como bloco de código."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def styled_pt(frame: pd.DataFrame, precision: int = 4, formats: dict | None = None):
    """Exibe números em pt-BR (vírgula decimal, ponto de milhar) sem perder a ordenação numérica."""
    style = frame.style.format(precision=precision, decimal=",", thousands=".", na_rep="—")
    if formats:
        formats = {col: fmt for col, fmt in formats.items() if col in frame.columns}
        style = style.format(formats, subset=list(formats), decimal=",", thousands=".", na_rep="—")
    return style


def metric_table(values: dict, references: dict | None = None) -> pd.DataFrame:
    rows = []
    for key, (label, meaning) in METRIC_INFO.items():
        if key in values and values[key] is not None:
            rows.append({"Métrica": label, "Valor": values[key], "O que significa": meaning})
    if references:
        rows.append({"Métrica": "AP de um classificador aleatório", "Valor": references.get("average_precision_aleatorio"),
                     "O que significa": "Referência para comparar a AP do modelo."})
        rows.append({"Métrica": "Brier prevendo sempre a taxa média", "Valor": references.get("brier_prevalencia_constante"),
                     "O que significa": "Referência para comparar o Brier do modelo."})
    return pd.DataFrame(rows)


def show_figure(path_in_reports: str):
    path = config.REPORT_DIR / path_in_reports
    if path.is_file():
        st.image(str(path), caption=FIGURE_CAPTIONS.get(path_in_reports, ""), width="stretch")


def numeric_widget(col: str):
    stats = schema.get("numericos", {}).get(col, {})
    is_int = col in INTEGER_FIELDS
    lower = 300 if col == "fico_range_low" else 0
    upper = 850 if col == "fico_range_low" else max(stats.get("max", 1_000_000), stats.get("mediana", 0) * 2, 1)
    default = stats.get("mediana", lower)
    cast = int if is_int else float
    return st.number_input(
        FORM_LABELS[col], min_value=cast(lower), max_value=cast(math.ceil(upper)),
        value=cast(round(default)) if is_int else float(default), step=cast(STEPS[col]),
        format=FORMATS.get(col), help=FIELD_HELP[col], key=f"in_{col}",
    )


def category_widget(col: str):
    options = list(schema.get("categoricos", {}).get(col, [])) or list(config.CATEGORY_LABELS.get(col, {}))
    if schema.get("ausentes_categoricos", {}).get(col, 0) > 0:
        options.append(MISSING_LABEL)
    return st.selectbox(
        FORM_LABELS[col], options, index=0, help=FIELD_HELP[col], key=f"in_{col}",
        format_func=lambda v: MISSING_LABEL if v == MISSING_LABEL else category_label(col, v),
    )


def risk_ruler(probability: float, threshold: float, base_rate: float | None) -> str:
    """Régua 0–50% (ou 0–100% se a probabilidade passar de 50%) com limiar, taxa média e a proposta."""
    top = 0.5 if probability <= 0.5 else 1.0
    step = 0.1 if top == 0.5 else 0.2
    kind = "co" if probability >= threshold else "fp"

    def x(value: float) -> float:
        return round(min(max(value / top, 0.0), 1.0) * 100, 2)

    ticks = "".join(f'<span style="left:{x(i * step)}%">{round(i * step * 100)}%</span>'
                    for i in range(round(top / step) + 1))
    avg = f'<div class="lr-avg" style="left:{x(base_rate)}%"></div>' if base_rate is not None else ""
    avg_legend = (f'<span><i class="lr-sw-avg"></i>taxa média de Charged Off no teste ({percent_pt(base_rate)})</span>'
                  if base_rate is not None else "")
    label = (f"Régua de risco: probabilidade {percent_pt(probability)}, limiar {percent_pt(threshold)}"
             + (f", taxa média {percent_pt(base_rate)}" if base_rate is not None else ""))
    return f"""
    <div class="lr-ruler" role="img" aria-label="{label}">
      <div class="lr-track">
        <div class="lr-zone" style="left:{x(threshold)}%"></div>
        {avg}
        <div class="lr-cut" style="left:{x(threshold)}%"><span>limiar {percent_pt(threshold)}</span></div>
        <div class="lr-mark {kind}" style="left:{x(probability)}%"></div>
      </div>
      <div class="lr-ticks">{ticks}</div>
      <div class="lr-legend">
        <span><i class="lr-sw-mark {kind}"></i>esta proposta</span>
        <span><i class="lr-sw-zone"></i>faixa classificada como Charged Off</span>
        {avg_legend}
      </div>
    </div>"""


# --------------------------------------------------------------------------------------
st.html(STYLE)
html_block("""
<div class="lr-masthead">
  <div>
    <div class="lr-eyebrow">CARCDSI · IFSP Caraguatatuba · projeto final</div>
    <div class="lr-title" role="heading" aria-level="1">LendingRisk AI</div>
  </div>
  <div class="lr-byline">Hyan da Cunha Ferreira Inácio<br>Luan Ibanhez de Almeida</div>
</div>
""")

try:
    metrics = cached_metrics(str(config.METRICS_PATH), _mtime(config.METRICS_PATH))
    model = cached_model(str(config.MODEL_PATH), _mtime(config.MODEL_PATH))
except ArtefatosInvalidos as exc:
    st.error(f"Modelo indisponível: {exc}")
    st.markdown("Para gerar o modelo, coloque o dataset em `data/` e execute, na raiz do projeto:")
    st.code("python -m src.training --data data/accepted_2007_to_2018Q4.csv")
    st.stop()

threshold = float(metrics["limiar"])
schema = metrics.get("esquema_entrada", {})
test_metrics = metrics["metricas_teste"]
partitions = metrics.get("particoes", {})
base_rate = partitions.get("teste", {}).get("taxa_charged_off")
for warning in version_warnings(metrics):
    st.warning(warning)

with st.sidebar:
    status = cached_ollama_status()
    train_part, test_part = partitions.get("treino", {}), partitions.get("teste", {})
    llm_state = ('<span class="lr-dot ok"></span>Ollama ativo' if status["ok"]
                 else '<span class="lr-dot off"></span>indisponível')
    html_block(f"""
    <div class="lr-ficha">
      <div class="lr-eyebrow">Ficha técnica</div>
      <dl>
        <dt>Classificador</dt><dd>{html.escape(metrics['modelo_escolhido'])}</dd>
        <dt>Seleção</dt><dd>maior AP na validação</dd>
        <dt>Limiar</dt><dd>{percent_pt(threshold)} (máximo F2)</dd>
        <dt>Treino</dt><dd>{month_pt(train_part.get('inicio'))} a {month_pt(train_part.get('fim'))}</dd>
        <dt>Teste</dt><dd>{month_pt(test_part.get('inicio'))} a {month_pt(test_part.get('fim'))}</dd>
        <dt>LLM</dt><dd>{html.escape(config.LLM_MODEL)}<br>{llm_state}</dd>
      </dl>
    </div>
    """)
    if not status["ok"]:
        st.caption(status["mensagem"])
    if st.button("Verificar Ollama de novo", type="tertiary"):
        cached_ollama_status.clear()
        st.rerun()
    html_block('<div class="lr-footnote">Protótipo acadêmico. Não use para aprovar ou recusar crédito de verdade.</div>')

tab_simulate, tab_performance, tab_data = st.tabs(
    ["Simular empréstimo", "Desempenho dos modelos", "Dados e metodologia"]
)

# ---------------------------------- Simulação -----------------------------------------
with tab_simulate:
    proposal, verdict = st.columns([1, 1], gap="large")
    with proposal:
        html_block('<div class="lr-eyebrow">Proposta</div>'
                   '<div class="lr-lead">Dados do pedido, nos moldes do cadastro do Lending Club. '
                   'Os campos começam na mediana do treino.</div>')
        with st.form("loan_form", border=False):
            values = {}
            for pair in FORM_ROWS:  # uma linha por par: no celular, a ordem lógica é mantida
                for target, col in zip(st.columns(2), pair):
                    with target:
                        values[col] = numeric_widget(col) if col in NUMERIC_FEATURES else category_widget(col)
            submitted = st.form_submit_button("Estimar risco", type="primary")

    with verdict:
        if submitted:
            try:
                record, input_warnings = validate_record(values, schema)
                probability = predict_probability(model, record)
                st.session_state["analysis"] = {
                    "id": time.time(), "record": record, "probability": probability,
                    "warnings": input_warnings, "display": format_record_pt(record),
                }
                st.session_state.pop("explanation", None)
            except ValueError as exc:
                st.session_state.pop("analysis", None)
                st.error(f"Não foi possível calcular: {exc}")
            except Exception as exc:  # erro inesperado do modelo: mostrar mensagem útil, não traceback
                st.session_state.pop("analysis", None)
                st.error(f"Erro inesperado ao calcular a previsão: {exc}. Verifique se o modelo foi treinado "
                         "com a versão atual do código.")

        analysis = st.session_state.get("analysis")
        if not analysis:
            html_block('<div class="lr-eyebrow">Parecer</div>'
                       '<div class="lr-empty">Preencha a proposta e clique em <b>Estimar risco</b>.<br>'
                       'O parecer do classificador aparece aqui.</div>')
        else:
            probability = analysis["probability"]
            above = probability >= threshold
            kind, label = ("co", "Charged Off") if above else ("fp", "Fully Paid")
            reading = (f"A probabilidade ficou acima do limiar de {percent_pt(threshold)}. O perfil se parece "
                       "com o de empréstimos que terminaram como perda." if above else
                       f"A probabilidade ficou abaixo do limiar de {percent_pt(threshold)}. Pelo critério "
                       "adotado, não há sinal de risco elevado.")
            html_block(f"""
            <div class="lr-eyebrow">Parecer do classificador · {html.escape(metrics['modelo_escolhido'])}</div>
            <div class="lr-hero-label">Probabilidade estimada de Charged Off</div>
            <div class="lr-hero">{percent_pt(probability)}</div>
            <div class="lr-verdict">Classe prevista <span class="lr-stamp {kind}">{label}</span></div>
            <div class="lr-text">{reading}</div>
            {risk_ruler(probability, threshold, base_rate)}
            """)
            for warning in analysis["warnings"]:
                st.warning(warning)
            if analysis["record"].get("term") == "60 months" and partitions.get("teste", {}).get("proporcao_60_meses") == 0:
                html_block('<div class="lr-aside"><b>Prazo de 60 meses.</b> Esses empréstimos só aparecem no '
                           'treino (safras até 2013). O teste tem apenas empréstimos de 36 meses, então o '
                           'desempenho do modelo para 60 meses não foi avaliado.</div>')
            html_block(f"""
            <div class="lr-aside">No teste reservado, este limiar encontrou {percent_pt(test_metrics['recall'])}
            dos Charged Off reais, mas só {percent_pt(test_metrics['precision'])} dos casos sinalizados eram de
            fato Charged Off. Ele foi escolhido para deixar passar poucos casos de risco, à custa de muitos
            alarmes falsos. A probabilidade é uma estimativa, não uma garantia.</div>
            """)
            with st.expander("Dados enviados ao modelo"):
                st.table(pd.DataFrame({"Campo": list(analysis["display"]),
                                       "Valor": list(analysis["display"].values())}).set_index("Campo"))

            importance = metrics.get("importancia_global", [])
            top_features = [FEATURE_LABELS.get(row["atributo"], row["atributo"]) for row in importance[:5]]
            if top_features:
                term_note = (" O prazo fica com importância zero porque a validação só tem empréstimos de 36 meses."
                             if partitions.get("validacao", {}).get("proporcao_60_meses") == 0 else "")
                items = "".join(f"<li>{html.escape(name)}</li>" for name in top_features)
                html_block(f"""
                <div class="lr-section">
                  <div class="lr-eyebrow">O que o modelo mais usa, em geral</div>
                  <ol class="lr-rank">{items}</ol>
                  <div class="lr-text">Importância global, medida na validação. Não diz o que pesou nesta
                  proposta específica.{term_note}</div>
                </div>
                """)

            html_block(f"""
            <div class="lr-section">
              <div class="lr-eyebrow">Explicação em texto</div>
              <div class="lr-text">Escrita pela LLM local ({html.escape(config.LLM_MODEL)}, via Ollama) a partir
              do resultado acima. Ela não altera o cálculo.</div>
            </div>
            """)
            if st.button("Explicar com LLM local (Ollama)"):
                with st.spinner("Gerando a explicação na CPU. Pode levar até um minuto."):
                    started = time.time()
                    try:
                        result = explain_checked(probability, threshold, analysis["display"], top_features,
                                                 model_name=metrics["modelo_escolhido"])
                        st.session_state["explanation"] = {
                            "analysis_id": analysis["id"], "text": result["texto"], "error": None,
                            "warnings": result["avisos"], "attempts": result["tentativas"],
                            "seconds": round(time.time() - started, 1),
                        }
                    except LLMIndisponivel as exc:
                        st.session_state["explanation"] = {"analysis_id": analysis["id"], "error": str(exc)}
            explanation = st.session_state.get("explanation")
            if explanation and explanation["analysis_id"] == analysis["id"]:
                if explanation["error"]:
                    st.warning(f"LLM indisponível: {explanation['error']} A previsão acima continua válida.")
                else:
                    attempts = explanation.get("attempts", 1)
                    seconds = f"{explanation['seconds']:.1f}".replace(".", ",")
                    retry = " · 2ª tentativa (a 1ª falhou na verificação automática)" if attempts > 1 else ""
                    text = html.escape(explanation["text"]).replace("\n", "<br>")
                    html_block(f'<div class="lr-memo">{text}</div>'
                               f'<div class="lr-memo-meta">{html.escape(config.LLM_MODEL)} · gerado em {seconds} s'
                               f'{retry}. {CLOSING_SENTENCE}</div>')
                    if explanation["warnings"]:
                        st.warning("A verificação automática encontrou possíveis inconsistências no texto da LLM "
                                   "(os valores corretos são os do classificador, acima):\n\n- "
                                   + "\n- ".join(explanation["warnings"]))

# --------------------------------- Desempenho -----------------------------------------
with tab_performance:
    st.markdown(f"Modelo escolhido: **{metrics['modelo_escolhido']}**, o de maior Average Precision na "
                f"validação. Limiar: **{percent_pt(threshold)}**, o que maximiza o F2 na validação. "
                "O conjunto de teste foi usado uma única vez, depois dessas escolhas.")
    st.subheader("Métricas no teste reservado")
    st.dataframe(styled_pt(metric_table(test_metrics, metrics.get("referencias_teste"))), hide_index=True,
                 width="stretch")
    matrix = metrics.get("matriz_confusao_teste")
    if matrix:
        st.markdown("**Matriz de confusão (teste)**")
        st.table(pd.DataFrame(matrix, index=["Real: Fully Paid", "Real: Charged Off"],
                              columns=["Previsto: Fully Paid", "Previsto: Charged Off"]).map(int_pt))

    st.subheader("Comparação dos três modelos na validação")
    comparison = pd.DataFrame(metrics["comparacao_validacao"])
    rename = {"modelo": "Modelo", "limiar": "Limiar (F2)", "tempo_treino_s": "Treino (s)",
              **{k: v[0] for k, v in METRIC_INFO.items()}}
    st.dataframe(styled_pt(comparison.rename(columns=rename), formats={"Limiar (F2)": "{:.2f}", "Treino (s)": "{:.1f}"}),
                 hide_index=True, width="stretch")

    figures_list = metrics.get("figuras", [])
    if figures_list:
        st.subheader("Gráficos")
        left, right = st.columns(2)
        for i, figure in enumerate(figures_list):
            with (left if i % 2 == 0 else right):
                show_figure(figure)
    for name, title in [("estabilidade_temporal.csv", "Tabela: estabilidade por mês de emissão"),
                        ("analise_subgrupos_teste.csv", "Tabela: análise por subgrupos no teste"),
                        ("importancia_atributos.csv", "Tabela: importância global por permutação")]:
        path = config.REPORT_DIR / name
        if path.is_file():
            with st.expander(title):
                st.dataframe(styled_pt(pd.read_csv(path)), hide_index=True, width="stretch")

    post_dir = config.REPORT_DIR / "analises_pos_teste"
    sensitivity_path, variants_path = post_dir / "sensibilidade_limiar.csv", post_dir / "comparacao_v1_v2.csv"
    if sensitivity_path.is_file() or variants_path.is_file():
        st.subheader("Análises complementares")
        st.caption("Feitas depois do uso do teste reservado. São descritivas: não mudam o modelo, o limiar "
                   "nem as métricas oficiais acima. Detalhes em docs/experimentos.md, seção 5.")
    if sensitivity_path.is_file():
        st.markdown("**E se o limiar fosse outro?** Os limiares alternativos foram definidos só com a "
                    "validação; a tabela mostra o efeito no teste.")
        sensitivity = pd.read_csv(sensitivity_path).loc[lambda df: df["particao"] == "teste"]
        st.table(pd.DataFrame({
            "Critério": sensitivity["criterio"], "Limiar": sensitivity["limiar"].map(percent_pt),
            "Sinaliza": sensitivity["taxa_sinalizada"].map(percent_pt),
            "Recall": sensitivity["recall"].map(percent_pt),
            "Precision": sensitivity["precision"].map(percent_pt),
            "Acurácia": sensitivity["acuracia"].map(percent_pt),
        }).set_index("Critério"))
    if variants_path.is_file():
        st.markdown("**Versão 2: tempo de emprego \"não informado\" como categoria própria.** Na versão "
                    "oficial, o campo vazio recebe a resposta mais comum do treino (\"10 anos ou mais\").")
        variants = pd.read_csv(variants_path).set_index("versao")
        decimal = lambda v: f"{v:.4f}".replace(".", ",")  # noqa: E731
        rows = [("Modelo escolhido", "modelo_escolhido", str), ("Limiar", "limiar", percent_pt),
                ("AP (teste)", "teste_average_precision", decimal), ("ROC-AUC (teste)", "teste_roc_auc", decimal),
                ("Brier (teste)", "teste_brier", decimal), ("ECE (teste)", "teste_ece", decimal),
                ("Recall (teste)", "teste_recall", percent_pt), ("Precision (teste)", "teste_precision", percent_pt),
                ("Sinaliza (teste)", "teste_taxa_positiva_prevista", percent_pt),
                ("Não informado: Charged Off real", "emp_length_ausente_taxa_charged_off_real", percent_pt),
                ("Não informado: risco médio previsto", "emp_length_ausente_score_medio_previsto", percent_pt)]
        labels = {"v1_oficial_moda": "v1 oficial (moda)", "v2_ausente_como_categoria": "v2 (categoria própria)"}
        st.table(pd.DataFrame({
            labels.get(version, version): {label: (fmt(variants.at[version, col]) if col in variants
                                                   and pd.notna(variants.at[version, col]) else "—")
                                           for label, col, fmt in rows}
            for version in variants.index
        }))

# ------------------------------- Dados e metodologia ----------------------------------
with tab_data:
    dataset = metrics.get("dataset", {})
    st.subheader("Fonte e preparação dos dados")
    period = dataset.get("periodo_emissao_arquivo") or ["?", "?"]
    st.markdown(
        f"- **Arquivo:** `{dataset.get('arquivo', '?')}` (Lending Club, empréstimos aceitos).\n"
        f"- **SHA-256:** `{dataset.get('sha256', 'não calculado')}`\n"
        f"- **Linhas lidas:** {int_pt(dataset.get('linhas_lidas'))}; com desfecho final: "
        f"{int_pt(dataset.get('linhas_com_desfecho_final'))}; usadas no experimento: "
        f"{int_pt(dataset.get('linhas_utilizadas'))}.\n"
        "- **Alvo:** 0 = Fully Paid (quitado), 1 = Charged Off (baixado como perda). Os demais status "
        "(Current, Late, Default etc.) ficam fora por não terem desfecho final.\n"
        f"- **Período de emissão no arquivo:** {month_pt(period[0])} a {month_pt(period[1])}"
    )
    maturity = dataset.get("filtro_maturidade", {})
    if maturity.get("ativo"):
        st.markdown(f"- **Filtro de maturidade:** só entram empréstimos cujo prazo contratual terminou até "
                    f"{month_pt(maturity.get('data_referencia'))} ({int_pt(maturity.get('linhas_removidas'))} "
                    "linhas sorteadas removidas). Isso evita que o período recente contenha apenas quem terminou cedo.")
    sampling = dataset.get("amostragem", {})
    if sampling:
        fraction, cap = sampling.get("fracao"), sampling.get("max_linhas")
        fraction_text = ("100% (todas as linhas elegíveis)" if fraction == 1
                         else percent_pt(fraction) if isinstance(fraction, (int, float)) else "?")
        cap_text = f"teto de {int_pt(cap)} linhas" if cap else "sem teto de linhas"
        st.markdown(f"- **Amostragem:** sorteio uniforme por linha (Bernoulli), fração {fraction_text}, "
                    f"{cap_text}, semente {sampling.get('semente')}.")
    if partitions:
        st.markdown("**Divisão temporal por data de emissão (`issue_d`)**")
        st.table(pd.DataFrame([
            {"Partição": PARTITION_LABELS.get(name, name), "Linhas": int_pt(part.get("linhas")),
             "Início": month_pt(part.get("inicio")), "Fim": month_pt(part.get("fim")),
             "Taxa de Charged Off": percent_pt(part["taxa_charged_off"]),
             "Proporção 60 meses": percent_pt(part["proporcao_60_meses"])}
            for name, part in partitions.items()
        ]).set_index("Partição"))
    audit_figure = config.REPORT_DIR / "figures" / "auditoria_coortes.png"
    if audit_figure.is_file():
        st.image(str(audit_figure), caption="Auditoria do arquivo completo: proporção de empréstimos finalizados "
                 "e taxa de Charged Off por ano de emissão e prazo.", width="stretch")

    st.subheader("Atributos usados")
    st.markdown("Somente informações conhecidas no momento do pedido. Colunas posteriores à concessão "
                "(pagamentos, recuperações, FICO mais recente) foram excluídas para evitar vazamento de dados.")
    st.table(pd.DataFrame({"Atributo": list(FEATURE_LABELS), "Descrição": list(FEATURE_LABELS.values())})
             .set_index("Atributo"))

    st.subheader("Limitações")
    st.markdown(
        "- Dados históricos dos EUA: não representam o mercado brasileiro nem o atual.\n"
        "- `Charged Off` é um desfecho específico (baixa como perda), não qualquer atraso.\n"
        "- Só entram empréstimos com desfecho final e prazo já encerrado (filtro de maturidade); "
        "as safras de 2016–2018 ficam de fora e o teste cobre apenas os últimos meses de 2015.\n"
        "- O teste contém apenas empréstimos de 36 meses; o desempenho para 60 meses não foi avaliado.\n"
        "- A probabilidade pode estar descalibrada em períodos diferentes do treino.\n"
        "- A importância de atributos é global e não explica casos individuais; não há explicação causal.\n"
        "- A explicação da LLM pode conter erros; os números válidos são sempre os do classificador.\n"
        "- Uso exclusivamente acadêmico: não deve ser usado para decisões reais de crédito."
    )
    environment = metrics.get("ambiente")
    if environment:
        with st.expander("Ambiente do treinamento (reprodutibilidade)"):
            st.json(environment)
