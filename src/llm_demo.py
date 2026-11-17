"""Demonstração documentada da LLM local com o modelo treinado.

Uso: python -m src.llm_demo

Usa PERFIS HIPOTÉTICOS de entrada (não são mutuários reais) para cobrir casos de
risco diferentes. Para cada perfil:
1. o classificador real calcula a probabilidade;
2. o Ollama gera a explicação;
3. a verificação de consistência procura números inventados ou classe trocada.

Grava reports/llm_demonstracao.json e reports/llm_demonstracao.md.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import FEATURE_LABELS, LLM_MODEL, METRICS_PATH, MODEL_PATH, OLLAMA_URL, REPORT_DIR
from src.inference import format_record_pt, load_metrics, load_model, percent_pt, predict_probability, validate_record
from src.llm import LLMIndisponivel, SYSTEM_PROMPT, build_prompt, explain_checked

PROFILES = {
    "A — FICO alto, DTI baixo, 36 meses": {
        "loan_amnt": 8000.0, "annual_inc": 120000.0, "dti": 8.0, "fico_range_low": 780,
        "inq_last_6mths": 0, "revol_util": 15.0, "term": "36 months",
        "purpose": "credit_card", "home_ownership": "MORTGAGE", "emp_length": "10+ years",
    },
    "B — perfil mediano, 36 meses": {
        "loan_amnt": 12000.0, "annual_inc": 65000.0, "dti": 17.6, "fico_range_low": 690,
        "inq_last_6mths": 0, "revol_util": 52.0, "term": "36 months",
        "purpose": "debt_consolidation", "home_ownership": "RENT", "emp_length": "3 years",
    },
    "C — FICO no mínimo usual, DTI alto, 60 meses": {
        "loan_amnt": 30000.0, "annual_inc": 40000.0, "dti": 35.0, "fico_range_low": 660,
        "inq_last_6mths": 3, "revol_util": 90.0, "term": "60 months",
        "purpose": "small_business", "home_ownership": "RENT", "emp_length": "< 1 year",
    },
    "D — tempo de emprego não informado": {
        "loan_amnt": 15000.0, "annual_inc": 55000.0, "dti": 22.0, "fico_range_low": 700,
        "inq_last_6mths": 1, "revol_util": 60.0, "term": "36 months",
        "purpose": "home_improvement", "home_ownership": "OWN", "emp_length": "Não informado",
    },
}


def ollama_metadata(url: str, model: str) -> dict:
    info = {"url": url, "modelo": model}
    try:
        info["versao_ollama"] = requests.get(f"{url}/api/version", timeout=3).json().get("version")
        for item in requests.get(f"{url}/api/tags", timeout=3).json().get("models", []):
            if item.get("name") == model:
                info["digest"] = item.get("digest")
                info["tamanho_bytes"] = item.get("size")
                info["detalhes"] = item.get("details")
    except (requests.RequestException, ValueError) as exc:
        info["erro"] = str(exc)
    return info


def run(report_dir: Path = REPORT_DIR) -> dict:
    metrics = load_metrics(METRICS_PATH)
    model = load_model(MODEL_PATH)
    threshold = float(metrics["limiar"])
    top = [FEATURE_LABELS.get(r["atributo"], r["atributo"]) for r in metrics.get("importancia_global", [])[:5]]
    results = []
    for name, raw in PROFILES.items():
        record, warnings = validate_record(raw, metrics.get("esquema_entrada"))
        probability = predict_probability(model, record)
        display = format_record_pt(record)
        prompt = build_prompt(probability, threshold, display, top, metrics["modelo_escolhido"])
        item = {"perfil": name, "entrada": display, "avisos_entrada": warnings,
                "probabilidade": round(probability, 6), "limiar": threshold,
                "classe": "Charged Off" if probability >= threshold else "Fully Paid"}
        started = time.time()
        try:
            result = explain_checked(probability, threshold, display, top, model_name=metrics["modelo_escolhido"])
            item.update({"explicacao": result["texto"], "erro": None, "verificacao": result["avisos"],
                         "tentativas": result["tentativas"], "prompt_usuario": prompt})
        except LLMIndisponivel as exc:
            item.update({"explicacao": None, "erro": str(exc), "verificacao": []})
        item["segundos"] = round(time.time() - started, 1)
        # a probabilidade não muda depois da LLM (a LLM não tem acesso ao modelo)
        item["probabilidade_recalculada_igual"] = predict_probability(model, record) == probability
        results.append(item)
        print(f"{name}: {percent_pt(probability)} -> {item['classe']} ({item['segundos']} s; "
              f"tentativas: {item.get('tentativas')}); avisos: {item['verificacao']}; erro: {item['erro']}", flush=True)

    payload = {
        "aviso": "Perfis HIPOTÉTICOS de entrada (não são mutuários reais). Probabilidades calculadas pelo "
                 "modelo treinado; textos gerados pela LLM local.",
        "modelo_classificador": metrics["modelo_escolhido"], "limiar": threshold,
        "importancia_global_top5": top, "ollama": ollama_metadata(OLLAMA_URL, LLM_MODEL),
        "prompt_sistema": SYSTEM_PROMPT,
        "resultados": results, "criado_em_utc": datetime.now(timezone.utc).isoformat(),
    }
    (report_dir / "llm_demonstracao.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    lines = [
        "# Demonstração da LLM local (Ollama)", "",
        f"> {payload['aviso']}", "",
        f"- Classificador: **{payload['modelo_classificador']}**, limiar {percent_pt(threshold)}",
        f"- LLM: `{LLM_MODEL}` via Ollama {payload['ollama'].get('versao_ollama', '?')} "
        f"(digest `{str(payload['ollama'].get('digest', '?'))[:12]}`), temperatura 0,1, execução local em CPU",
        f"- Gerado em {payload['criado_em_utc']}",
        "- A interface exibe, abaixo de cada explicação: \"Esta é uma estimativa acadêmica e não substitui "
        "a análise humana.\"", "",
        "<details><summary>Prompt de sistema usado</summary>", "", "```text", SYSTEM_PROMPT, "```", "</details>", "",
    ]
    for item in results:
        lines += [f"## Perfil {item['perfil']}", "",
                  "| Campo | Valor |", "|---|---|",
                  *[f"| {k} | {v} |" for k, v in item["entrada"].items()], "",
                  f"**Classificador:** probabilidade {percent_pt(item['probabilidade'])} → classe "
                  f"**{item['classe']}** (limiar {percent_pt(threshold)}).", ""]
        if item["erro"]:
            lines += [f"**LLM indisponível:** {item['erro']}", ""]
        else:
            lines += ["**Explicação gerada pela LLM** "
                      f"({item['segundos']} s, {item.get('tentativas', 1)} tentativa(s)):", "", *[f"> {line}" if line else ">" for line in
                                                        item["explicacao"].splitlines()], ""]
            lines += ["**Verificação automática:** " + (
                "nenhuma inconsistência encontrada." if not item["verificacao"]
                else "; ".join(item["verificacao"])), ""]
    (report_dir / "llm_demonstracao.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main(argv=None):
    argparse.ArgumentParser(description="Demonstração da LLM local com perfis hipotéticos").parse_args(argv)
    run()


if __name__ == "__main__":
    main()
