"""Interface de demonstração do LendingRisk AI (projeto final CARCDSI, IFSP Caraguatatuba).

Executar a partir da raiz do projeto:  python -m streamlit run app.py

Navegação lateral, proposta e resumo lado a lado, parecer detalhado em largura
completa. Componentes em src/ui.py e estilos em assets/styles.css.
"""
from __future__ import annotations

import html
import math
import time

import pandas as pd
import streamlit as st

from src import config, ui
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


html_block = ui.markup


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
        value=st.session_state.get("proposal_values", {}).get(col, cast(round(default)) if is_int else float(default)), step=cast(STEPS[col]),
        format=FORMATS.get(col), help=FIELD_HELP[col], key=f"in_{col}", on_change=proposal_changed, args=(col,),
    )


def category_widget(col: str):
    options = list(schema.get("categoricos", {}).get(col, [])) or list(config.CATEGORY_LABELS.get(col, {}))
    if schema.get("ausentes_categoricos", {}).get(col, 0) > 0:
        options.append(MISSING_LABEL)
    return st.selectbox(
        FORM_LABELS[col], options, index=options.index(st.session_state.get("proposal_values", {}).get(col, options[0])),
        help=FIELD_HELP[col], key=f"in_{col}", on_change=proposal_changed, args=(col,),
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


def proposal_changed(column: str) -> None:
    """Preserva os campos entre páginas e invalida o parecer ao editar a proposta."""
    st.session_state.setdefault("proposal_values", {})[column] = st.session_state[f"in_{column}"]
    if "analysis" in st.session_state:
        st.session_state["proposal_edited"] = True
    st.session_state.pop("analysis", None)
    st.session_state.pop("explanation", None)


def go_to(page: str) -> None:
    st.session_state["page"] = page


def refresh_connection() -> None:
    cached_ollama_status.clear()
    st.session_state["refresh_ollama_requested"] = True


ui.load_styles()
try:
    metrics = cached_metrics(str(config.METRICS_PATH), _mtime(config.METRICS_PATH))
    model = cached_model(str(config.MODEL_PATH), _mtime(config.MODEL_PATH))
except ArtefatosInvalidos as exc:
    ui.brand()
    st.error(f"Modelo indisponível: {exc}")
    st.markdown("Para gerar o modelo, coloque o dataset em `data/lending_club/` e execute, na raiz do projeto:")
    st.code("python -m src.training --data data/lending_club/accepted_2007_to_2018Q4.csv")
    st.stop()

threshold = float(metrics["limiar"])
schema = metrics.get("esquema_entrada", {})
test_metrics = metrics["metricas_teste"]
partitions = metrics.get("particoes", {})
base_rate = partitions.get("teste", {}).get("taxa_charged_off")
dataset = metrics.get("dataset", {})
PAGES = ["Simular empréstimo", "Desempenho dos modelos", "Dados e metodologia"]
page = st.session_state.setdefault("page", PAGES[0])

with st.sidebar:
    with st.container(key="sidebar-shell"):
        ui.brand()
        html_block('<div class="lr-eyebrow">Plataforma</div>')
        ui.navigation(PAGES, page, go_to)
        with st.container(key="sidebar-bottom"):
            train_part, test_part = partitions.get("treino", {}), partitions.get("teste", {})
            html_block(f"""<div class="lr-model-card"><div class="lr-eyebrow">Modelo selecionado</div>
                <strong>{html.escape(metrics['modelo_escolhido'])}</strong>
                <dl><dt>Limiar de decisão</dt><dd>{percent_pt(threshold)}</dd><dt>Critério</dt><dd>Maior AP</dd>
                <dt>Treino</dt><dd>{month_pt(train_part.get('inicio'))} – {month_pt(train_part.get('fim'))}</dd>
                <dt>Teste</dt><dd>{month_pt(test_part.get('inicio'))} – {month_pt(test_part.get('fim'))}</dd></dl></div>""")
            with st.container(key="ollama-status"):
                if st.session_state.pop("refresh_ollama_requested", False):
                    with st.spinner("Verificando conexão…"):
                        status = cached_ollama_status()
                else:
                    status = cached_ollama_status()
                ui.connection_status(status)
                st.button("Verificar conexão", key="refresh-ollama", type="tertiary",
                          help="Verificar conexão", on_click=refresh_connection)
            ui.sidebar_credits()

llm_state = ('<span class="lr-dot ok"></span>Ollama ativo' if status["ok"]
             else '<span class="lr-dot off"></span>indisponível')
for warning in version_warnings(metrics):
    st.warning(warning)

if page == PAGES[0]:
    title, action = st.columns([4, 1], vertical_alignment="top")
    with title:
        ui.heading("Simulador de risco de crédito", "Explore o comportamento de um classificador treinado com dados históricos do Lending Club. "
                   "Preencha a proposta para estimar a probabilidade de Charged Off.", "SIMULADOR")
    with action:
        st.button("Ver metodologia", icon=":material/menu_book:", on_click=go_to, args=(PAGES[2],))
    ui.indicators([
        ("Empréstimos no experimento", int_pt(dataset.get("linhas_utilizadas")), "Base histórica elegível", "card"),
        ("Taxa de Charged Off", percent_pt(dataset["taxa_charged_off"]) if "taxa_charged_off" in dataset else "—", "Base do experimento", "chart"),
        ("Modelo utilizado", metrics["modelo_escolhido"], "Maior AP na validação", "bolt"),
        ("Limiar de classificação", percent_pt(threshold), "Máximo F2 na validação", "sliders"),
    ])
    proposal, verdict = st.columns([1.65, 1], gap="medium")
    with proposal, st.container(border=True, key="proposal"):
        ui.section("Dados da solicitação", "Os campos começam na mediana do treino. Preencha as informações para sua simulação.", "card")
        values = {}
        groups = [
            ("Informações do empréstimo", "bank", [("loan_amnt", "term"), ("purpose",)]),
            ("Capacidade financeira", "card", [("annual_inc", "dti"), ("fico_range_low", "revol_util")]),
            ("Perfil e histórico", "people", [("inq_last_6mths", "home_ownership"), ("emp_length",)]),
        ]
        for group_title, symbol, rows in groups:
            html_block('<div class="lr-form-section"></div>')
            ui.section(group_title, symbol=symbol)
            for row in rows:
                for target, col in zip(st.columns(len(row)), row):
                    with target:
                        values[col] = numeric_widget(col) if col in NUMERIC_FEATURES else category_widget(col)
        st.session_state["proposal_values"] = values.copy()
        st.caption("Valores em USD · A previsão é produzida pelo classificador real.")
        submitted = st.button("Estimar risco", type="primary", icon=":material/arrow_forward:", width="stretch")
    if submitted:
        try:
            record, input_warnings = validate_record(values, schema)
            with st.spinner("Calculando a previsão do classificador…"):
                probability = predict_probability(model, record)
            st.session_state["analysis"] = {
                "id": time.time(), "record": record, "probability": probability,
                "warnings": input_warnings, "display": format_record_pt(record),
            }
            st.session_state.pop("explanation", None)
            st.session_state.pop("proposal_edited", None)
        except ValueError as exc:
            st.session_state.pop("analysis", None)
            st.error(f"Não foi possível calcular: {exc}")
        except Exception as exc:
            st.session_state.pop("analysis", None)
            st.error(f"Erro inesperado ao calcular a previsão: {exc}. Verifique se o modelo foi treinado "
                     "com a versão atual do código.")
    analysis = st.session_state.get("analysis")
    if analysis:
        probability = analysis["probability"]
        above = probability >= threshold
        kind, label = ("co", "Charged Off") if above else ("fp", "Fully Paid")
        relation = "igual" if probability == threshold else "acima" if above else "abaixo"
        reading = (f"A probabilidade ficou {relation} ao limiar de {percent_pt(threshold)}. "
                   + ("O perfil se parece com o de empréstimos que terminaram como perda." if above else
                      "Pelo critério adotado, não há sinal de risco elevado. Isso não garante o pagamento."))
    with verdict:
        with st.container(border=True, key="summary"):
            ui.section("Parecer de risco")
            if not analysis:
                html_block(f"""<div class="lr-empty"><span class="lr-icon">{ui.icon()}</span>
                    <h3>Aguardando simulação</h3><p>Preencha a proposta e clique em <b>Estimar risco</b>.
                    O resultado do classificador aparecerá aqui.</p></div>""")
                if st.session_state.get("proposal_edited"):
                    st.info("Proposta alterada. Execute uma nova análise para atualizar o parecer.")
            else:
                html_block(f"""<div class="lr-hero-label">Probabilidade estimada de Charged Off</div>
                    <div class="lr-hero">{percent_pt(probability)}</div>
                    <div class="lr-verdict">Classe prevista <span class="lr-stamp {kind}">{label}</span></div>
                    <div class="lr-text">{reading}</div>""")
                html_block(risk_ruler(probability, threshold, base_rate))
                st.markdown("[Ver parecer completo ↓](#parecer-detalhado)")
            html_block('<div class="lr-aside"><b>Como interpretar?</b><br>O limiar foi escolhido para maximizar '
                       'F2 na validação. Charged Off é baixa como perda, não qualquer atraso ou recusa automática.</div>')
        ui.academic_notice()
        with st.container(border=True, key="classifier"):
            ui.section("Informações do classificador", symbol="shield")
            html_block(f"""<div class="lr-ficha"><dl><dt>Algoritmo</dt><dd>{html.escape(metrics['modelo_escolhido'])}</dd>
                <dt>Seleção</dt><dd>Maior AP na validação</dd><dt>Limiar</dt><dd>{percent_pt(threshold)} · F2</dd>
                <dt>LLM local</dt><dd>{llm_state}</dd></dl></div>""")
    # Irmão do layout de duas colunas: o parecer ocupa a largura completa.
    if analysis:
        with st.container(key="detailed-verdict"):
            html_block('<div id="parecer-detalhado"></div>')
            ui.heading("Parecer do classificador", "Leitura detalhada da previsão, dos dados enviados e das limitações.",
                       f"RESULTADO DA ANÁLISE / {metrics['modelo_escolhido']}")
            for warning in analysis["warnings"]:
                st.warning(warning)
            if analysis["record"].get("term") == "60 months" and partitions.get("teste", {}).get("proporcao_60_meses") == 0:
                st.warning("Prazo de 60 meses: esses empréstimos só aparecem no treino (safras até 2013). "
                           "O teste tem apenas empréstimos de 36 meses; o desempenho para 60 meses não foi avaliado.")
            prob_col, class_col = st.columns([1.6, 1])
            with prob_col, st.container(border=True, key="probability-card"):
                ui.section("Probabilidade estimada", "A posição da estimativa em relação ao limiar e à taxa média do teste.")
                ui.indicators([
                    ("Esta proposta", percent_pt(probability), "Probabilidade de Charged Off", "chart"),
                    ("Limiar de decisão", percent_pt(threshold), "Maior F2 na validação", "sliders"),
                    ("Taxa média no teste", percent_pt(base_rate) if base_rate is not None else "—", "Frequência observada de perdas", "people"),
                ])
                html_block(risk_ruler(probability, threshold, base_rate))
                st.caption("Escala até 50% (100% se a proposta superar 50%). A taxa média do teste é uma estatística, não um limiar adicional.")
            with class_col, st.container(border=True, key="interpretation-card"):
                ui.section("Classificação e interpretação", "Resultado da previsão e contexto de desempenho do classificador.", "shield")
                html_block(f'<div class="lr-verdict {kind}"><b>Classe prevista: {label}</b><p>{reading}</p></div>')
                ui.indicators([
                    ("Recall no teste", percent_pt(test_metrics["recall"]), "Perdas reais identificadas", "chart"),
                    ("Precisão no teste", percent_pt(test_metrics["precision"]), "Sinalizações que eram perdas reais", "shield"),
                ])
                html_block(f'<div class="lr-text">No teste reservado, este limiar encontrou {percent_pt(test_metrics["recall"])} '
                           f'dos Charged Off reais, mas só {percent_pt(test_metrics["precision"])} dos casos sinalizados eram de fato Charged Off. '
                           'Ele prioriza encontrar perdas, à custa de muitos alarmes falsos. A probabilidade é uma estimativa, não uma garantia.</div>')
            data_col, importance_col = st.columns([1.35, 1])
            with data_col, st.container(border=True, key="record-card"):
                ui.section("Dados enviados ao modelo", "Os dez atributos desta proposta, utilizados nesta previsão.", "card")
                items = ''.join(f'<div><dt>{html.escape(name)}</dt><dd>{html.escape(value)}</dd></div>' for name, value in analysis["display"].items())
                html_block(f'<dl class="lr-record">{items}</dl>')
            importance = metrics.get("importancia_global", [])
            top_features = [FEATURE_LABELS.get(row["atributo"], row["atributo"]) for row in importance[:5]]
            with importance_col, st.container(border=True, key="importance-card"):
                ui.section("Atributos mais importantes no geral", "Importância global por permutação, medida na validação.")
                items = ''.join(f'<li>{html.escape(name)}</li>' for name in top_features)
                html_block(f'<ol class="lr-rank">{items}</ol>')
                term_note = (" O prazo fica com importância zero porque a validação só tem empréstimos de 36 meses."
                             if partitions.get("validacao", {}).get("proporcao_60_meses") == 0 else "")
                html_block(f'<div class="lr-text">Importância global não diz o que pesou nesta proposta específica '
                           f'e não implica causalidade.{term_note}</div>')
            with st.container(border=True, key="llm-card"):
                ui.section("Explicação em texto · LLM local", "A LLM explica o resultado e não altera o cálculo do classificador.", "info")
                st.caption(f"{config.LLM_MODEL} · {config.OLLAMA_URL}")
                if not status["ok"]:
                    st.info(f"Ollama indisponível: {status['mensagem']} A previsão do classificador continua disponível.")
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
        ui.academic_notice()

elif page == PAGES[1]:
    title, action = st.columns([4, 1])
    with title:
        ui.heading("Desempenho dos modelos", "Resultados da validação e do teste reservado, com a comparação dos três classificadores.", "DESEMPENHO")
    with action:
        st.button("Ver metodologia", icon=":material/menu_book:", on_click=go_to, args=(PAGES[2],))
    ui.indicators([
        ("Modelo selecionado", metrics["modelo_escolhido"], "Maior AP na validação", "bolt"),
        ("Limiar de classificação", percent_pt(threshold), "Máximo F2 na validação", "sliders"),
        ("Recall no teste", percent_pt(test_metrics["recall"]), "Charged Off identificados", "chart"),
        ("Precisão no teste", percent_pt(test_metrics["precision"]), "Casos sinalizados corretos", "shield"),
        ("Base de teste", int_pt(partitions.get("teste", {}).get("linhas")), "Empréstimos avaliados", "card"),
    ])
    with st.container(border=True, key="model-summary"):
        ui.section("Resumo do modelo escolhido", symbol="shield")
        st.markdown(f"Modelo escolhido: **{metrics['modelo_escolhido']}**, o de maior Average Precision na "
                    f"validação. Limiar: **{percent_pt(threshold)}**, o que maximiza o F2 na validação. "
                    "O conjunto de teste foi usado uma única vez, depois dessas escolhas.")
        st.caption("O limiar prioriza encontrar mais perdas reais, mesmo com muitos falsos positivos. "
                   "A classificação não é uma aprovação ou recusa de crédito.")
    met_col, matrix_col, comp_col = st.columns([1.2, 1, 1.3])
    with met_col, st.container(border=True, key="test-metrics"):
        ui.section("Métricas no teste reservado")
        st.dataframe(styled_pt(metric_table(test_metrics, metrics.get("referencias_teste"))), hide_index=True, width="stretch")
    with matrix_col, st.container(border=True, key="confusion"):
        ui.section("Matriz de confusão (teste)", "Linhas: classe real · Colunas: classe prevista", "card")
        matrix = metrics.get("matriz_confusao_teste")
        if matrix:
            st.table(pd.DataFrame(matrix, index=["Fully Paid", "Charged Off"], columns=["Fully Paid", "Charged Off"]).map(int_pt))
            tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
            st.caption(f"Acertos: {int_pt(tn + tp)} · Erros: {int_pt(fp + fn)}")
    with comp_col, st.container(border=True, key="comparison"):
        ui.section("Comparação dos três modelos na validação")
        comparison = pd.DataFrame(metrics["comparacao_validacao"])
        rename = {"modelo": "Modelo", "limiar": "Limiar (F2)", "tempo_treino_s": "Treino (s)", **{k: v[0] for k, v in METRIC_INFO.items()}}
        st.dataframe(styled_pt(comparison.rename(columns=rename), formats={"Limiar (F2)": "{:.2f}", "Treino (s)": "{:.1f}"}), hide_index=True, width="stretch")
    figures_list = metrics.get("figuras", [])
    if figures_list:
        st.subheader("Gráficos")
        st.caption("Figuras originais do experimento. Use a ampliação da imagem para inspecionar eixos e legendas.")
        for start in range(0, len(figures_list), 3):
            columns = st.columns(3)
            for target, figure in zip(columns, figures_list[start:start + 3]):
                with target, st.container(border=True, key=f"figure-{start}-{figure.split('/')[-1]}"):
                    ui.section(FIGURE_CAPTIONS.get(figure, figure).split(":")[0].split("(")[0].rstrip(". "))
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

    ui.academic_notice()

elif page == PAGES[2]:
    title, action = st.columns([4, 1])
    with title:
        ui.heading("Dados e metodologia", "Entenda a origem dos dados, os critérios de seleção, a divisão temporal e as limitações do experimento.", "DADOS E METODOLOGIA")
    with action:
        st.button("Ver desempenho", icon=":material/show_chart:", on_click=go_to, args=(PAGES[1],))
    ui.indicators([
        ("Linhas lidas", int_pt(dataset.get("linhas_lidas")), "Arquivo original do Lending Club", "book"),
        ("Com desfecho final", int_pt(dataset.get("linhas_com_desfecho_final")), "Fully Paid ou Charged Off", "shield"),
        ("Usadas no experimento", int_pt(dataset.get("linhas_utilizadas")), "Após o filtro de maturidade", "card"),
        ("Atributos utilizados", str(len(metrics["atributos"])), "Conhecidos no momento do pedido", "chart"),
    ])
    source_col, eligibility_col = st.columns([1.1, 1])
    with source_col, st.container(border=True, key="source-card"):
        ui.section("Fonte e preparação dos dados", "Lending Club · empréstimos aceitos · Estados Unidos", "book")
        st.code(dataset.get("arquivo", "?"), language=None, wrap_lines=True)
        period = dataset.get("periodo_emissao_arquivo") or ["?", "?"]
        html_block(f'<div class="lr-text"><b>Período de emissão:</b> {month_pt(period[0])} a {month_pt(period[1])}</div>')
        st.markdown("**Alvo:** 0 = Fully Paid (quitado), 1 = Charged Off (baixado como perda). "
                    "Os demais status (Current, Late, Default etc.) ficam fora por não terem desfecho final.")
        html_block(f'<p class="lr-hash">SHA-256: {html.escape(dataset.get("sha256", "não calculado"))}</p>')
    with eligibility_col, st.container(border=True, key="eligibility-card"):
        ui.section("Critérios de elegibilidade", "Cuidados para reduzir vieses e vazamento de informações", "shield")
        st.markdown("**01 · Desfecho conhecido** — Somente Fully Paid e Charged Off; demais estados não indicam resultado final.")
        maturity = dataset.get("filtro_maturidade", {})
        if maturity.get("ativo"):
            st.markdown(f"**02 · Filtro de maturidade** — Só entram empréstimos cujo prazo contratual terminou até "
                        f"{month_pt(maturity.get('data_referencia'))}. {int_pt(maturity.get('linhas_removidas'))} "
                        "linhas sorteadas removidas. Isso evita que o período recente contenha apenas quem terminou cedo.")
        sampling = dataset.get("amostragem", {})
        if sampling:
            fraction, cap = sampling.get("fracao"), sampling.get("max_linhas")
            fraction_text = ("100% (todas as linhas elegíveis)" if fraction == 1 else
                             percent_pt(fraction) if isinstance(fraction, (int, float)) else "?")
            cap_text = f"teto de {int_pt(cap)} linhas" if cap else "sem teto de linhas"
            st.markdown(f"**03 · Amostragem reproduzível** — Sorteio uniforme por linha (Bernoulli), fração {fraction_text}, "
                        f"{cap_text}, semente {sampling.get('semente')}.")
        st.markdown("**04 · Sem vazamento de dados** — Informações posteriores à concessão (pagamentos, recuperações e FICO recente) foram excluídas.")
    if partitions:
        with st.container(border=True, key="temporal-card"):
            ui.section("Divisão temporal por data de emissão", "Treino, validação e teste reservado · campo issue_d", "shield")
            cards = ''.join(f'<article><b>{i:02} · {PARTITION_LABELS.get(name, name)}</b>'
                            f'<strong>{int_pt(part.get("linhas"))}</strong><small>{month_pt(part.get("inicio"))} — '
                            f'{month_pt(part.get("fim"))}</small></article>' for i, (name, part) in enumerate(partitions.items(), 1))
            html_block(f'<div class="lr-timeline">{cards}</div>')
            st.table(pd.DataFrame([
                {"Partição": PARTITION_LABELS.get(name, name), "Linhas": int_pt(part.get("linhas")),
                 "Início": month_pt(part.get("inicio")), "Fim": month_pt(part.get("fim")),
                 "Taxa de Charged Off": percent_pt(part["taxa_charged_off"]), "Proporção 60 meses": percent_pt(part["proporcao_60_meses"])}
                for name, part in partitions.items()
            ]).set_index("Partição"))
            html_block(f'<div class="lr-aside">Seleção por maior AP e limiar {percent_pt(threshold)} por maior F2, '
                       'ambos na validação. O teste foi avaliado uma única vez, depois dessas escolhas.</div>')
    features_col, audit_col = st.columns([1.1, 1])
    with features_col, st.container(border=True, key="features-card"):
        ui.section("Atributos utilizados pelo modelo", "Dez informações disponíveis no momento da proposta", "card")
        st.table(pd.DataFrame({"Atributo": list(FEATURE_LABELS), "Descrição": list(FEATURE_LABELS.values())}).set_index("Atributo"))
    with audit_col:
        with st.container(border=True, key="audit-card"):
            ui.section("Auditoria da base completa", "Maturidade dos empréstimos por ano de emissão e prazo")
            audit_figure = config.REPORT_DIR / "figures" / "auditoria_coortes.png"
            if audit_figure.is_file():
                st.image(str(audit_figure), caption="Auditoria do arquivo completo: proporção de empréstimos finalizados "
                         "e taxa de Charged Off por ano de emissão e prazo.", width="stretch")
            else:
                st.warning("Gráfico de auditoria indisponível nos artefatos carregados.")
        with st.container(border=True, key="target-card"):
            ui.section("Interpretação da variável-alvo", "O que o classificador realmente prevê", "info")
            st.markdown("**0 · Fully Paid:** empréstimo quitado. **1 · Charged Off:** baixa como perda.")
            html_block('<div class="lr-aside">Charged Off não equivale a qualquer atraso. '
                       'As probabilidades estimadas não são garantias de um resultado real.</div>')
    with st.container(border=True, key="method-flow"):
        ui.section("Do dado à explicação", "Fluxo implementado no projeto")
        steps = ["Preparação e filtragem da base", "Seleção e tratamento dos atributos", "Separação temporal",
                 "Treinamento dos três modelos", "Seleção por AP na validação", "Limiar por F2 na validação",
                 "Avaliação no teste reservado", "Explicação pela LLM local"]
        cards = ''.join(f'<article><b>{i:02}</b><p>{name}</p></article>' for i, name in enumerate(steps, 1))
        html_block(f'<div class="lr-flow">{cards}</div>')
    with st.container(border=True, key="limitations-card"):
        ui.section("Limitações e considerações", "Restrições científicas relevantes para a leitura dos resultados", "info")
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
        with st.container(border=True, key="environment-card"):
            ui.section("Ambiente de treinamento · Reprodutibilidade", "Versões e configurações registradas na execução do experimento", "book")
            cards = ''.join(f'<article><small>{html.escape(str(name))}</small><strong>{html.escape(str(value))}</strong></article>'
                            for name, value in environment.items() if not isinstance(value, (dict, list)))
            html_block(f'<div class="lr-environment">{cards}</div>')
            with st.expander("Ambiente do treinamento (reprodutibilidade)"):
                st.json(environment)
    ui.academic_notice()

ui.footer()
