"""LLM local (Ollama) apenas para verbalizar o resultado; NÃO calcula nem modifica o risco.

A probabilidade, o limiar e a classe vêm do classificador. A LLM recebe esses
valores prontos e só redige o texto. Após a resposta, `check_consistency`
procura contradições simples (percentuais inventados, classe invertida).
"""
from __future__ import annotations

import re

import requests

from src.config import LLM_MODEL, OLLAMA_URL

CONNECT_TIMEOUT_S = 3
READ_TIMEOUT_S = 180  # CPU sem GPU: a primeira chamada inclui carregar o modelo na memória
MAX_TOKENS = 320
GENERATION_OPTIONS = {"temperature": 0.1, "top_p": 0.9, "num_predict": MAX_TOKENS, "seed": 42}
CLOSING_SENTENCE = "Esta é uma estimativa acadêmica e não substitui a análise humana."

SYSTEM_PROMPT = f"""Você explica, em português do Brasil, o resultado de um protótipo acadêmico de \
análise de risco de empréstimos treinado com dados históricos do Lending Club (EUA, 2007–2018).
Regras obrigatórias:
1. Use SOMENTE os números e fatos fornecidos. Não calcule, não estime e não altere a probabilidade, o limiar ou a classificação.
2. Cite a probabilidade e o limiar exatamente como foram escritos.
3. Não afirme que algum dado do cliente aumentou ou reduziu o risco DESTE caso: a importância informada é GLOBAL (média do modelo) e não explica casos individuais.
4. Não invente estatísticas, causas, regras de crédito, comparações ou percentuais.
5. Não recomende aprovar, recusar ou alterar o empréstimo.
6. Escreva no máximo 120 palavras, em linguagem simples, sem listas e sem títulos.
7. Termine exatamente com a frase: "{CLOSING_SENTENCE}\""""


class LLMIndisponivel(RuntimeError):
    """Ollama fora do ar, modelo não baixado, tempo esgotado ou resposta inválida."""


def _pct(value: float) -> str:
    return f"{value * 100:.1f}".replace(".", ",") + "%"


def classification_text(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "ACIMA do limiar — classe prevista: Charged Off (empréstimo com perfil de maior risco de perda)"
    return "ABAIXO do limiar — classe prevista: Fully Paid (empréstimo sem sinal de risco elevado pelo critério adotado)"


def build_prompt(probability: float, threshold: float, inputs: dict, global_features: list[str],
                 model_name: str | None = None) -> str:
    """inputs: rótulos em português -> valores já formatados (ver inference.format_record_pt)."""
    dados = "\n".join(f"- {key}: {value}" for key, value in inputs.items())
    importance = ", ".join(global_features[:5]) or "indisponível"
    classifier = f" ({model_name})" if model_name else ""
    return f"""RESULTADO CALCULADO PELO CLASSIFICADOR DE MACHINE LEARNING{classifier} — não altere:
- Probabilidade estimada de Charged Off (empréstimo baixado como perda): {_pct(probability)}
- Limiar de classificação definido na validação: {_pct(threshold)}
- Classificação: {classification_text(probability, threshold)}

DADOS INFORMADOS PELO USUÁRIO:
{dados}

ATRIBUTOS MAIS IMPORTANTES DO MODELO EM GERAL (importância global, NÃO deste caso): {importance}

Explique esse resultado para uma pessoa leiga."""


def explain_with_ollama(probability: float, threshold: float, inputs: dict, global_features: list[str],
                        model: str = LLM_MODEL, url: str = OLLAMA_URL, model_name: str | None = None,
                        timeout: tuple = (CONNECT_TIMEOUT_S, READ_TIMEOUT_S)) -> str:
    prompt = build_prompt(probability, threshold, inputs, global_features, model_name)
    try:
        response = requests.post(
            f"{url}/api/generate",
            json={"model": model, "system": SYSTEM_PROMPT, "prompt": prompt, "stream": False,
                  "options": GENERATION_OPTIONS, "keep_alive": "10m"},
            timeout=timeout,
        )
    except requests.exceptions.ConnectTimeout as exc:
        raise LLMIndisponivel(f"Tempo esgotado ao conectar ao Ollama em {url}.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise LLMIndisponivel(
            f"Não foi possível conectar ao Ollama em {url}. Verifique se o serviço está em execução."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise LLMIndisponivel(
            f"O Ollama não respondeu em {timeout[1]} s. Na primeira chamada o modelo é carregado; tente de novo."
        ) from exc
    if response.status_code == 404:
        raise LLMIndisponivel(f"Modelo '{model}' não encontrado no Ollama. Execute: ollama pull {model}")
    if response.status_code >= 400:
        raise LLMIndisponivel(f"O Ollama respondeu com erro HTTP {response.status_code}: {response.text[:200]}")
    try:
        output = str(response.json().get("response", "")).strip()
    except ValueError as exc:
        raise LLMIndisponivel("O Ollama devolveu uma resposta que não é JSON válido.") from exc
    if not output:
        raise LLMIndisponivel("O Ollama retornou uma resposta vazia.")
    return output


def check_ollama(url: str = OLLAMA_URL, model: str = LLM_MODEL, timeout: float = 3) -> dict:
    """Verifica se o servidor responde e se o modelo foi baixado (sem gerar texto)."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=timeout)
        response.raise_for_status()
        names = [m.get("name", "") for m in response.json().get("models", [])]
    except (requests.exceptions.RequestException, ValueError):
        return {"ok": False, "mensagem": f"Ollama não está acessível em {url}.", "modelos": []}
    if model not in names and f"{model}:latest" not in names:
        return {"ok": False, "mensagem": f"Ollama ativo, mas o modelo '{model}' não foi baixado "
                f"(ollama pull {model}).", "modelos": names}
    return {"ok": True, "mensagem": f"Ollama ativo com o modelo '{model}'.", "modelos": names}


_PERCENT = re.compile(r"(\d{1,3}(?:[.,]\d{1,2})?)\s?%")


def _percent_values(text: str) -> list[float]:
    return [float(v.replace(",", ".")) for v in _PERCENT.findall(text)]


def check_consistency(text: str, probability: float, threshold: float, prompt: str = "") -> list[str]:
    """Verificação heurística: a resposta não pode contradizer os números fornecidos."""
    warnings = []
    found = _percent_values(text)
    allowed = _percent_values(prompt) + [probability * 100, threshold * 100]
    if not any(abs(v - probability * 100) <= 0.6 for v in found):
        warnings.append(f"A explicação não cita a probabilidade calculada pelo modelo ({_pct(probability)}).")
    invented = sorted({v for v in found if all(abs(v - a) > 0.6 for a in allowed)})
    if invented:
        listed = ", ".join(f"{v:g}%".replace(".", ",") for v in invented)
        warnings.append(f"A explicação cita percentuais que não foram fornecidos ({listed}); desconsidere-os.")
    lowered = text.lower()
    if probability >= threshold and re.search(r"\babaixo do limiar", lowered):
        warnings.append("A explicação fala em 'abaixo do limiar', mas o modelo classificou ACIMA do limiar.")
    if probability < threshold and re.search(r"\bacima do limiar", lowered):
        warnings.append("A explicação fala em 'acima do limiar', mas o modelo classificou ABAIXO do limiar.")
    return warnings
