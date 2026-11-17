"""Interface de demonstração do LendingRisk AI (projeto final CARCDSI — IFSP Caraguatatuba).

Executar a partir da raiz do projeto:  python -m streamlit run app.py
"""
from __future__ import annotations

import math
import time

import pandas as pd
import streamlit as st

from src import config
from src.config import CATEGORICAL_FEATURES, FEATURE_LABELS, NUMERIC_FEATURES
from src.inference import (
    MISSING_LABEL, ArtefatosInvalidos, category_label, format_record_pt, load_metrics,
    load_model, percent_pt, predict_probability, validate_record, version_warnings,
)
from src.llm import (
    CLOSING_SENTENCE, LLMIndisponivel, check_ollama, explain_checked,
)

st.set_page_config(page_title="LendingRisk AI", page_icon="📊", layout="wide")

METRIC_INFO = {
    "average_precision": ("Average Precision (AP)", "Resumo da curva precisão × recall. Um modelo aleatório teria AP ≈ taxa de Charged Off."),
    "roc_auc": ("ROC-AUC", "Chance de um Charged Off receber score maior que um Fully Paid (0,5 = aleatório)."),
    "brier": ("Brier score", "Erro quadrático médio da probabilidade (menor é melhor)."),
    "log_loss": ("Log loss", "Penaliza probabilidades confiantes e erradas (menor é melhor)."),
    "ece": ("ECE — erro de calibração", "Diferença média entre probabilidade prevista e taxa observada (menor é melhor)."),
    "recall": ("Recall", "Fração dos Charged Off reais que o modelo sinalizou acima do limiar."),
    "precision": ("Precision", "Fração dos sinalizados que de fato terminaram em Charged Off."),
    "f1": ("F1", "Média harmônica de precision e recall."),
    "f2": ("F2", "Como o F1, mas dá mais peso ao recall (critério do limiar)."),
    "acuracia": ("Acurácia", "Acertos totais. Enganosa com classes desbalanceadas."),
    "taxa_positiva_real": ("Taxa real de Charged Off", "Proporção de Charged Off no conjunto."),
    "taxa_positiva_prevista": ("Taxa sinalizada pelo modelo", "Proporção de casos acima do limiar."),
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
FIGURE_CAPTIONS = {
    "figures/curvas_pr_validacao.png": "Curvas precisão × recall dos três modelos na validação (critério de seleção: AP).",
    "figures/curvas_roc_validacao.png": "Curvas ROC dos três modelos na validação.",
    "figures/calibracao_validacao.png": "Calibração na validação: pontos sobre a diagonal = probabilidade bem calibrada.",
    "figures/limiar_validacao.png": "Precision, recall e F2 por limiar na validação; a linha vertical é o limiar escolhido.",
    "figures/matriz_confusao_teste.png": "Matriz de confusão no teste reservado (avaliado uma única vez).",
    "figures/distribuicao_scores_teste.png": "Distribuição das probabilidades no teste, separada pela classe real.",
    "figures/importancia_atributos.png": "Importância GLOBAL por permutação (validação) — não explica casos individuais.",
    "figures/estabilidade_temporal.png": "AP e taxa real de Charged Off por mês de emissão (validação e teste).",
    "figures/calibracao_teste.png": "Calibração do modelo escolhido no teste.",
}


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


# --------------------------------------------------------------------------------------
st.title("LendingRisk AI")
st.caption("Sistema acadêmico de análise de risco de empréstimos • CARCDSI — IFSP Caraguatatuba • "
           "Hyan da Cunha Ferreira Inácio e Luan Ibanhez de Almeida")

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
for warning in version_warnings(metrics):
    st.warning(warning)

with st.sidebar:
    st.header("Modelo em uso")
    st.markdown(f"**{metrics['modelo_escolhido']}**, escolhido pela maior AP na validação.")
    st.markdown(f"Limiar de classificação: **{percent_pt(threshold)}** (máximo F2 na validação).")
    if partitions:
        st.markdown(f"Dados de treino: {partitions['treino']['inicio'][:7]} a {partitions['treino']['fim'][:7]}; "
                    f"teste: {partitions['teste']['inicio'][:7]} a {partitions['teste']['fim'][:7]}.")
    st.divider()
    st.header("LLM local")
    status = cached_ollama_status()
    if status["ok"]:
        st.success(status["mensagem"], icon="✅")
    else:
        st.warning(status["mensagem"], icon="⚠️")
    if st.button("Verificar Ollama novamente"):
        cached_ollama_status.clear()
        st.rerun()
    st.divider()
    st.caption("Protótipo acadêmico. Não use para aprovar ou recusar crédito de verdade.")

tab_simulate, tab_performance, tab_data = st.tabs(
    ["Simular empréstimo", "Desempenho dos modelos", "Dados e metodologia"]
)

# ---------------------------------- Simulação -----------------------------------------
with tab_simulate:
    st.markdown("Informe as características de um pedido de empréstimo. O classificador estima a "
                "probabilidade de o empréstimo terminar como **Charged Off** (baixado como perda), "
                "segundo o padrão histórico do Lending Club.")
    with st.form("loan_form"):
        col_a, col_b = st.columns(2)
        values = {}
        for i, col in enumerate(NUMERIC_FEATURES):
            stats = schema.get("numericos", {}).get(col, {})
            target = col_a if i % 2 == 0 else col_b
            is_int = col in INTEGER_FIELDS
            lower = 0 if col != "fico_range_low" else 300
            upper = max(stats.get("max", 1_000_000), stats.get("mediana", 0) * 2, 1)
            if col == "fico_range_low":
                upper = 850
            default = stats.get("mediana", lower)
            cast = int if is_int else float
            values[col] = target.number_input(
                FEATURE_LABELS[col], min_value=cast(lower), max_value=cast(math.ceil(upper)),
                value=cast(round(default)) if is_int else float(default), step=cast(STEPS[col]),
                help=FIELD_HELP[col], key=f"in_{col}",
            )
        for i, col in enumerate(CATEGORICAL_FEATURES):
            options = list(schema.get("categoricos", {}).get(col, []))
            if not options:
                options = list(config.CATEGORY_LABELS.get(col, {}))
            if schema.get("ausentes_categoricos", {}).get(col, 0) > 0:
                options.append(MISSING_LABEL)
            target = col_a if i % 2 == 0 else col_b
            values[col] = target.selectbox(
                FEATURE_LABELS[col], options, index=0, help=FIELD_HELP[col], key=f"in_{col}",
                format_func=lambda v, c=col: MISSING_LABEL if v == MISSING_LABEL else category_label(c, v),
            )
        submitted = st.form_submit_button("Estimar risco", type="primary")

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
    if analysis:
        probability = analysis["probability"]
        above = probability >= threshold
        st.subheader("Resultado do classificador")
        for warning in analysis["warnings"]:
            st.warning(warning)
        test_share_60 = partitions.get("teste", {}).get("proporcao_60_meses")
        if analysis["record"].get("term") == "60 months" and test_share_60 == 0:
            st.info("Empréstimos de **60 meses** só aparecem no treino (safras até 2013, com prazo já "
                    "encerrado). O teste contém apenas empréstimos de 36 meses, então o desempenho do "
                    "modelo para 60 meses **não foi avaliado**.", icon="ℹ️")
        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilidade estimada de Charged Off", percent_pt(probability))
        m2.metric("Limiar de classificação", percent_pt(threshold))
        if partitions:
            m3.metric("Taxa média de Charged Off no período de teste", percent_pt(partitions["teste"]["taxa_charged_off"]))
        if above:
            st.error("**Classe prevista: Charged Off** — a probabilidade ficou **acima** do limiar. "
                     "O perfil se parece com o de empréstimos que terminaram como perda.", icon="🔺")
        else:
            st.success("**Classe prevista: Fully Paid** — a probabilidade ficou **abaixo** do limiar.", icon="🔹")
        st.caption(
            f"Com este limiar, no teste reservado o modelo sinalizou {percent_pt(test_metrics['recall'])} dos "
            f"Charged Off reais (recall), e {percent_pt(test_metrics['precision'])} dos casos sinalizados eram "
            "de fato Charged Off (precision). O limiar privilegia encontrar casos de risco, aceitando muitos "
            "alarmes falsos. A probabilidade é uma estimativa do modelo — veja a calibração na aba "
            "'Desempenho dos modelos' — e não uma garantia."
        )
        with st.expander("Dados enviados ao modelo"):
            st.table(pd.DataFrame({"Campo": list(analysis["display"]), "Valor": list(analysis["display"].values())}))

        importance = metrics.get("importancia_global", [])
        top_features = [FEATURE_LABELS.get(row["atributo"], row["atributo"]) for row in importance[:5]]
        if top_features:
            st.markdown("**Atributos mais importantes para o modelo em geral** (importância global): "
                        + ", ".join(top_features) + ".")
            st.caption("Importância global indica o que o modelo mais usa em média. Ela NÃO indica quais "
                       "atributos causaram ou pesaram mais nesta previsão individual.")
            if partitions.get("validacao", {}).get("proporcao_60_meses") == 0:
                st.caption("O prazo aparece com importância zero porque a validação (onde a importância é "
                           "medida) só contém empréstimos de 36 meses; isso não prova que o prazo seja irrelevante.")

        st.subheader("Explicação em linguagem natural")
        st.caption(f"Gerada localmente pelo modelo {config.LLM_MODEL} via Ollama. A LLM recebe a probabilidade "
                   "já calculada e apenas a explica; ela não altera o resultado acima.")
        if st.button("Explicar com LLM local (Ollama)"):
            with st.spinner("Gerando explicação local (na CPU pode levar até um minuto)..."):
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
                st.info(explanation["text"])
                attempts = explanation.get("attempts", 1)
                st.caption(f"{CLOSING_SENTENCE} Texto gerado em {explanation['seconds']} s"
                           + (f" ({attempts} tentativas: a primeira falhou na verificação automática)." if attempts > 1 else "."))
                if explanation["warnings"]:
                    st.warning("A verificação automática encontrou possíveis inconsistências no texto da LLM "
                               "(os valores corretos são os do classificador, acima):\n\n- "
                               + "\n- ".join(explanation["warnings"]))

# --------------------------------- Desempenho -----------------------------------------
with tab_performance:
    st.markdown(f"Modelo escolhido: **{metrics['modelo_escolhido']}** — maior Average Precision na "
                f"validação. Limiar: **{percent_pt(threshold)}** (máximo F2 na validação). "
                "O conjunto de teste foi usado uma única vez, depois dessas escolhas.")
    st.subheader("Métricas no teste reservado")
    st.dataframe(metric_table(test_metrics, metrics.get("referencias_teste")), hide_index=True, width="stretch")
    matrix = metrics.get("matriz_confusao_teste")
    if matrix:
        st.markdown("**Matriz de confusão (teste)**")
        st.table(pd.DataFrame(matrix, index=["Real: Fully Paid", "Real: Charged Off"],
                              columns=["Previsto: Fully Paid", "Previsto: Charged Off"]))

    st.subheader("Comparação dos três modelos na validação")
    comparison = pd.DataFrame(metrics["comparacao_validacao"])
    rename = {"modelo": "Modelo", "limiar": "Limiar (F2)", "tempo_treino_s": "Treino (s)",
              **{k: v[0] for k, v in METRIC_INFO.items()}}
    st.dataframe(comparison.rename(columns=rename), hide_index=True, width="stretch")

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
                st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")

# ------------------------------- Dados e metodologia ----------------------------------
with tab_data:
    dataset = metrics.get("dataset", {})
    st.subheader("Fonte e preparação dos dados")
    period = dataset.get("periodo_emissao_arquivo") or ["?", "?"]
    st.markdown(
        f"- **Arquivo:** `{dataset.get('arquivo', '?')}` — Lending Club, empréstimos aceitos.\n"
        f"- **SHA-256:** `{dataset.get('sha256', 'não calculado')}`\n"
        f"- **Linhas lidas:** {int_pt(dataset.get('linhas_lidas'))}; com desfecho final: "
        f"{int_pt(dataset.get('linhas_com_desfecho_final'))}; usadas no experimento: "
        f"{int_pt(dataset.get('linhas_utilizadas'))}.\n"
        "- **Alvo:** 0 = Fully Paid (quitado), 1 = Charged Off (baixado como perda). Os demais status "
        "(Current, Late, Default etc.) ficam fora por não terem desfecho final.\n"
        f"- **Período de emissão no arquivo:** {period[0]} a {period[1]}"
    )
    maturity = dataset.get("filtro_maturidade", {})
    if maturity.get("ativo"):
        st.markdown(f"- **Filtro de maturidade:** só entram empréstimos cujo prazo contratual terminou até "
                    f"{maturity.get('data_referencia')} ({int_pt(maturity.get('linhas_removidas'))} linhas "
                    "sorteadas removidas). Isso evita que o período recente contenha apenas quem terminou cedo.")
    sampling = dataset.get("amostragem", {})
    if sampling:
        st.markdown(f"- **Amostragem:** {sampling.get('metodo')}, fração {sampling.get('fracao')}, "
                    f"teto {sampling.get('max_linhas')} linhas, semente {sampling.get('semente')}.")
    if partitions:
        st.markdown("**Divisão temporal por data de emissão (`issue_d`)**")
        st.table(pd.DataFrame(partitions).T.rename(columns={
            "linhas": "Linhas", "inicio": "Início", "fim": "Fim", "taxa_charged_off": "Taxa de Charged Off",
            "proporcao_60_meses": "Proporção 60 meses"}))
    audit_figure = config.REPORT_DIR / "figures" / "auditoria_coortes.png"
    if audit_figure.is_file():
        st.image(str(audit_figure), caption="Auditoria do arquivo completo: proporção de empréstimos finalizados "
                 "e taxa de Charged Off por ano de emissão e prazo.", width="stretch")

    st.subheader("Atributos usados")
    st.markdown("Somente informações conhecidas no momento do pedido. Colunas posteriores à concessão "
                "(pagamentos, recuperações, FICO mais recente) foram excluídas para evitar vazamento de dados.")
    st.table(pd.DataFrame({"Atributo": list(FEATURE_LABELS), "Descrição": list(FEATURE_LABELS.values())}))

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
