"""LLM local (Ollama) apenas para verbalizar o resultado; NÃO calcula nem modifica o risco.

A probabilidade, o limiar e a classe vêm do classificador. A LLM recebe esses
valores prontos e só redige o texto. Após a resposta, `check_consistency`
procura contradições simples (percentuais inventados, classe invertida).
"""
from __future__ import annotations

import json
import re

import requests

from src.config import LLM_MODEL, OLLAMA_URL

CONNECT_TIMEOUT_S = 3
READ_TIMEOUT_S = 180  # CPU sem GPU: a primeira chamada inclui carregar o modelo na memória
MAX_TOKENS = 400      # folga para os 3 campos JSON; se estourar, check_consistency avisa
GENERATION_OPTIONS = {"temperature": 0.1, "top_p": 0.9, "num_predict": MAX_TOKENS, "seed": 42}
# Exibido pela interface junto da explicação (não depende da LLM escrever).
CLOSING_SENTENCE = "Esta é uma estimativa acadêmica e não substitui a análise humana."

# Histórico do prompt (evidências em reports/llm_demonstracao_v*.md):
# - v1: texto livre "até 120 palavras" -> ~200 palavras, listas, texto cortado e leitura
#   errada da importância global.
# - v2: "EXATAMENTE 3 frases" -> o llama3.2:3b ignorou a estrutura (~150 palavras, cortes).
# - v3: saída estruturada do Ollama (JSON Schema), uma frase por campo; o código monta o
#   parágrafo. Resolveu tamanho/cortes, mas o modelo omitia o limiar e fazia atribuição
#   causal ("seguro devido à situação de moradia").
# - v4: v3 + dois exemplos curtos (few-shot). Melhorou muito, mas o modelo ainda errou
#   uma comparação numérica ("10,9% está abaixo do limiar de 9,0%") e qualificou valores
#   do cliente ("score FICO baixo").
# - v5 (atual): v4 + comparação já resolvida no prompt ("10,9% está acima do limiar de 9,0%"),
#   proibição de qualificar valores do cliente e, em erro crítico detectado pela
#   verificação, uma segunda geração (explain_checked) — fica a resposta com menos avisos.
RESPONSE_FIELDS = ("probabilidade", "limiar", "atributos")
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {field: {"type": "string"} for field in RESPONSE_FIELDS},
    "required": list(RESPONSE_FIELDS),
}
SYSTEM_PROMPT = """Você explica, em português do Brasil e para uma pessoa leiga, o resultado de um protótipo \
acadêmico de análise de risco de empréstimos treinado com dados históricos do Lending Club (EUA, 2007–2015).
Responda SOMENTE com um JSON com três campos, cada um com UMA frase curta (no máximo 30 palavras):
- "probabilidade": informe a probabilidade estimada de Charged Off (empréstimo baixado como perda) e o que ela significa, sem exagerar.
- "limiar": compare a probabilidade com o limiar e diga a classificação resultante.
- "atributos": cite os atributos que o modelo mais usa EM GERAL e diga que isso não explica este caso específico.
Regras obrigatórias:
- Use somente os números fornecidos, escritos exatamente como aparecem. Não calcule, não estime e não altere nenhum valor.
- Não diga que algum dado deste cliente aumentou, reduziu ou causou o risco: esse efeito individual não foi medido.
- Não use palavras absolutas como "certamente", "extremamente seguro" ou "será perdido": é uma estimativa.
- Não invente estatísticas, causas, regras de crédito ou comparações. Não repita a lista de dados do cliente.
- Não recomende aprovar, recusar ou alterar o empréstimo.
- No campo "atributos", não comente nem qualifique os valores deste cliente (não diga que a renda é alta, o score é baixo etc.).
- No campo "limiar", use a comparação fornecida (acima/abaixo) exatamente como está.
Exemplos de formato (os números dos exemplos NÃO são deste caso; use sempre os números fornecidos):
{"probabilidade": "O modelo estimou em 23,4% a chance de este empréstimo terminar como Charged Off, isto é, ser baixado como perda.", \
"limiar": "Como 23,4% está acima do limiar de 12,0%, o empréstimo foi classificado como Charged Off, um perfil de maior risco pelo critério adotado.", \
"atributos": "Em geral, o modelo dá mais peso a atributos como prazo e situação de moradia, mas essa importância é global e não explica este caso específico."}
{"probabilidade": "O modelo estimou em 5,2% a chance de este empréstimo terminar como Charged Off, isto é, ser baixado como perda.", \
"limiar": "Como 5,2% está abaixo do limiar de 12,0%, o empréstimo foi classificado como Fully Paid, sem sinal de risco elevado pelo critério adotado.", \
"atributos": "Em geral, o modelo dá mais peso a atributos como prazo e situação de moradia, mas essa importância é global e não explica este caso específico."}"""


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
- Comparação (já conferida; copie): {_pct(probability)} está {"acima" if probability >= threshold else "abaixo"} \
do limiar de {_pct(threshold)}
- Classificação: {classification_text(probability, threshold)}

DADOS INFORMADOS PELO USUÁRIO (apenas contexto; não repita esta lista):
{dados}

ATRIBUTOS MAIS IMPORTANTES DO MODELO EM GERAL (importância global, NÃO deste caso): {importance}

Explique esse resultado para uma pessoa leiga."""


def explain_with_ollama(probability: float, threshold: float, inputs: dict, global_features: list[str],
                        model: str = LLM_MODEL, url: str = OLLAMA_URL, model_name: str | None = None,
                        timeout: tuple = (CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
                        options: dict | None = None) -> str:
    prompt = build_prompt(probability, threshold, inputs, global_features, model_name)
    generation = {**GENERATION_OPTIONS, **(options or {})}
    try:
        response = requests.post(
            f"{url}/api/generate",
            json={"model": model, "system": SYSTEM_PROMPT, "prompt": prompt, "stream": False,
                  "format": RESPONSE_SCHEMA, "options": generation, "keep_alive": "10m"},
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
    return compose_text(output)


# Avisos que indicam erro factual (e não só de estilo): disparam uma segunda geração.
CRITICAL_MARKERS = ("não cita a probabilidade", "não cita o limiar", "não foram fornecidos",
                    "mas o modelo classificou", "cortado")
RETRY_OPTIONS = {"temperature": 0.4, "seed": 7}


def is_critical(warnings: list[str]) -> bool:
    return any(marker in warning for warning in warnings for marker in CRITICAL_MARKERS)


def explain_checked(probability: float, threshold: float, inputs: dict, global_features: list[str],
                    model_name: str | None = None, max_attempts: int = 2, **kwargs) -> dict:
    """Gera a explicação, verifica e, se houver erro crítico, tenta mais uma vez.

    Devolve {"texto", "avisos", "tentativas"} com a resposta que teve menos avisos.
    A probabilidade/limiar/classe exibidos ao usuário continuam vindo do classificador.
    """
    prompt = build_prompt(probability, threshold, inputs, global_features, model_name)
    best, attempts = None, 0
    for attempt in range(max_attempts):
        attempts += 1
        text = explain_with_ollama(probability, threshold, inputs, global_features, model_name=model_name,
                                   options=RETRY_OPTIONS if attempt else None, **kwargs)
        warnings = check_consistency(text, probability, threshold, prompt)
        if best is None or len(warnings) < len(best["avisos"]):
            best = {"texto": text, "avisos": warnings}
        if not is_critical(warnings):
            break
    return {**best, "tentativas": attempts}


def compose_text(output: str) -> str:
    """Junta os 3 campos do JSON em um parágrafo. Se não for JSON (ex.: Ollama antigo sem
    saída estruturada, ou resposta cortada), devolve o texto bruto para a verificação apontar."""
    try:
        data = json.loads(output)
    except ValueError:
        return output
    if not isinstance(data, dict):
        return output
    sentences = []
    for field in RESPONSE_FIELDS:
        sentence = str(data.get(field, "")).strip()
        if sentence:
            sentences.append(sentence if sentence.endswith((".", "!", "?")) else sentence + ".")
    return " ".join(sentences) or output


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
_CAUSAL = re.compile(
    r"\b(aument|reduz|diminu|contribu|caus|influenci|impact)\w*\b[^.]{0,40}\b(risco|probabilidade|score)\b"
    r"|\bnão (são|é|foram|foi) relevantes?\b"
    r"|\b(devido (a|à|ao|às|aos)|por causa d|graças a|em razão d|em função d)"
    r"|\bcontribu\w*\s+para\b"
)
_ABSOLUTE = re.compile(r"\b(certamente|com certeza|garant\w*|não há risco|sem (nenhum )?risco|extremamente|"
                       r"será (perdido|pago|quitado)|vai (pagar|quitar|perder))\b")


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
    if not text.rstrip().endswith((".", "!", "?", "…", '"', "»")):
        warnings.append("O texto parece ter sido cortado pelo limite de tamanho da resposta.")
    if not any(abs(v - threshold * 100) <= 0.6 for v in found):
        warnings.append(f"A explicação não cita o limiar de classificação ({_pct(threshold)}).")
    if _ABSOLUTE.search(lowered):
        warnings.append("O texto usa termos absolutos (ex.: 'sem risco', 'certamente'); a saída do modelo é "
                        "apenas uma estimativa.")
    if _CAUSAL.search(lowered):
        warnings.append("O texto pode estar atribuindo causa a algum atributo deste caso, o que o modelo "
                        "não mede; considere apenas os números do classificador.")
    if probability >= threshold and re.search(r"\babaixo do limiar", lowered):
        warnings.append("A explicação fala em 'abaixo do limiar', mas o modelo classificou ACIMA do limiar.")
    if probability < threshold and re.search(r"\bacima do limiar", lowered):
        warnings.append("A explicação fala em 'acima do limiar', mas o modelo classificou ABAIXO do limiar.")
    return warnings
