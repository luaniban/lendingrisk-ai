"""Interface de demonstração do projeto final CARCDSI."""
import json

import joblib
import pandas as pd
import streamlit as st

from src.config import FEATURES, METRICS_PATH, MODEL_PATH, REPORT_DIR
from src.llm import explain_with_ollama

st.set_page_config(page_title="LendingRisk AI", page_icon="📊", layout="centered")
st.title("LendingRisk AI")
st.caption("Projeto acadêmico • Hyan da Cunha Ferreira Inácio e Luan Ibanhez de Almeida")
st.write("Analise o risco estimado de não pagamento de um empréstimo usando Machine Learning.")

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("Modelo ainda não treinado. Execute o treinamento indicado no README.md.")
    st.code("python -m src.training --data data/accepted_2007_to_2018Q4.csv.gz")
    st.stop()

@st.cache_resource
def load_model():
    # Carregue somente um modelo treinado por vocês (joblib não é seguro para arquivos não confiáveis).
    return joblib.load(MODEL_PATH)

pipeline = load_model()
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
threshold = float(metrics["limiar"])

with st.expander("Resultados do treinamento", expanded=False):
    st.write("Modelo escolhido:", metrics["modelo_escolhido"])
    st.write("Métricas no conjunto de teste (separado da validação):")
    st.json(metrics["metricas_teste"])
    st.write("Comparação dos modelos na validação:")
    st.dataframe(pd.DataFrame(metrics["comparacao_validacao"]), hide_index=True)

st.subheader("Dados do empréstimo")
with st.form("loan_form"):
    amount = st.number_input("Valor solicitado (USD)", min_value=500.0, max_value=100_000.0, value=10_000.0, step=500.0)
    income = st.number_input("Renda anual declarada (USD)", min_value=0.0, max_value=10_000_000.0, value=50_000.0, step=1_000.0)
    dti = st.number_input("Relação entre dívidas e renda (dti, conforme o dataset)", min_value=0.0, max_value=100.0, value=18.0, step=1.0)
    fico = st.number_input("FICO (limite inferior do intervalo)", min_value=300, max_value=850, value=680, step=5)
    inquiries = st.number_input("Consultas de crédito nos últimos 6 meses", min_value=0, max_value=50, value=1, step=1)
    utilization = st.number_input("Uso do crédito rotativo (%)", min_value=0.0, max_value=200.0, value=45.0, step=1.0)
    term = st.selectbox("Prazo", ["36 months", "60 months"])
    purpose = st.selectbox("Finalidade", [
        "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
        "car", "medical", "small_business", "moving", "vacation", "other",
    ])
    home = st.selectbox("Moradia", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    employment = st.selectbox("Tempo de emprego", [
        "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
        "6 years", "7 years", "8 years", "9 years", "10+ years",
    ])
    submitted = st.form_submit_button("Estimar risco", type="primary")

if submitted:
    record = {
        "loan_amnt": amount, "annual_inc": income, "dti": dti,
        "fico_range_low": fico, "inq_last_6mths": inquiries, "revol_util": utilization,
        "term": term, "purpose": purpose, "home_ownership": home, "emp_length": employment,
    }
    probability = float(pipeline.predict_proba(pd.DataFrame([record])[FEATURES])[0, 1])
    st.session_state["last_analysis"] = {"record": record, "probability": probability}

if "last_analysis" in st.session_state:
    analysis = st.session_state["last_analysis"]
    record = analysis["record"]
    probability = analysis["probability"]
    st.subheader("Resultado do modelo")
    st.metric("Probabilidade estimada de não pagamento", f"{probability * 100:.1f}%")
    st.write(f"Limiar escolhido na validação: **{threshold * 100:.0f}%**")
    st.write("**Classificação:** " + ("Acima do limiar de risco" if probability >= threshold else "Abaixo do limiar de risco"))
    st.caption("Este score se refere ao desfecho histórico 'Charged Off' no dataset estudado. Não é uma decisão de crédito nem garantia do resultado.")

    importances = REPORT_DIR / "importancia_atributos.csv"
    top_features = []
    if importances.exists():
        top_features = pd.read_csv(importances)["atributo"].head(5).tolist()
        st.write("Atributos de maior importância **global** na validação:", ", ".join(top_features))
        st.caption("Importância global não significa que esses atributos causaram esta previsão individual.")

    if st.button("Explicar com LLM local (Ollama)"):
        with st.spinner("Gerando explicação local..."):
            try:
                explanation = explain_with_ollama(probability, threshold, record, top_features)
                st.info(explanation)
            except Exception as e:
                st.warning(f"LLM indisponível: {e}. Confira o Ollama e se o modelo llama3.2:3b foi baixado.")
