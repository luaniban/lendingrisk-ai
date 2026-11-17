"""LLM local apenas para verbalizar resultados; não calcula nem modifica o risco."""
import requests


def explain_with_ollama(probability, threshold, inputs, global_features, model="llama3.2:3b"):
    dados = "\n".join(f"- {key}: {value}" for key, value in inputs.items())
    importance = ", ".join(global_features[:5]) or "indisponível"
    prompt = f"""Você está ajudando a explicar um protótipo acadêmico de análise de crédito.
Escreva em português brasileiro, com até 110 palavras, usando linguagem simples.
O CLASSIFICADOR DE MACHINE LEARNING calculou, e você não pode alterar:
Probabilidade estimada de não pagamento: {probability:.1%}.
Limiar de classificação: {threshold:.1%}.
Classe prevista: {"risco acima do limiar" if probability >= threshold else "risco abaixo do limiar"}.
Dados informados:\n{dados}
Atributos globalmente relevantes na validação: {importance}.
IMPORTANTE: a importância é global, NÃO é explicação causal nem individual; não invente
justificativas, estatísticas, fatos ou razões diretas não fornecidas. Não afirme que determinado
atributo aumentou ou diminuiu esta previsão, porque esse efeito específico não foi medido.
Não recomende aprovação ou recusa automática de crédito. Termine alertando que é apenas
uma estimativa acadêmica sujeita a revisão humana."""
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
        timeout=180,
    )
    response.raise_for_status()
    output = response.json().get("response", "").strip()
    if not output:
        raise RuntimeError("O Ollama retornou uma resposta vazia.")
    return output
