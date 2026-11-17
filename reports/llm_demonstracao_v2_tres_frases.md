# Demonstração da LLM local (Ollama)

> Perfis HIPOTÉTICOS de entrada (não são mutuários reais). Probabilidades calculadas pelo modelo treinado; textos gerados pela LLM local.

- Classificador: **XGBoost**, limiar 9,0%
- LLM: `llama3.2:3b` via Ollama 0.40.16 (digest `a80c4f17acd5`), temperatura 0,1, execução local em CPU
- Gerado em 2026-11-17T12:51:16.598619+00:00
- A interface exibe, abaixo de cada explicação: "Esta é uma estimativa acadêmica e não substitui a análise humana."

<details><summary>Prompt de sistema usado</summary>

```text
Você explica, em português do Brasil, o resultado de um protótipo acadêmico de análise de risco de empréstimos treinado com dados históricos do Lending Club (EUA, 2007–2015).
Responda com EXATAMENTE 3 frases curtas, em um único parágrafo, sem listas e sem títulos (no máximo 90 palavras):
1ª frase: informe a probabilidade estimada de Charged Off e diga, em palavras simples, o que ela significa.
2ª frase: compare a probabilidade com o limiar e diga a classificação resultante.
3ª frase: cite os atributos que o modelo mais usa EM GERAL e deixe claro que isso não explica este caso específico.
Regras obrigatórias:
- Use somente os números fornecidos, escritos exatamente como aparecem. Não calcule, não estime e não altere nenhum valor.
- Não diga que algum dado deste cliente aumentou, reduziu ou causou o risco: esse efeito individual não foi medido.
- Não invente estatísticas, causas, regras de crédito ou comparações. Não repita a lista de dados do cliente.
- Não recomende aprovar, recusar ou alterar o empréstimo.
```
</details>

## Perfil A — FICO alto, DTI baixo, 36 meses

| Campo | Valor |
|---|---|
| Valor solicitado (USD) | US$ 8.000 |
| Renda anual declarada (USD) | US$ 120.000 |
| Relação dívida/renda — DTI (%) | 8,0% |
| Score FICO (limite inferior da faixa) | 780 |
| Consultas de crédito nos últimos 6 meses | 0 |
| Uso do crédito rotativo (%) | 15,0% |
| Prazo | 36 meses |
| Finalidade | Quitar cartão de crédito |
| Situação de moradia | Imóvel financiado (hipoteca) |
| Tempo de emprego | 10 anos ou mais |

**Classificador:** probabilidade 1,4% → classe **Fully Paid** (limiar 9,0%).

**Explicação gerada pela LLM** (24.4 s):

> O modelo de análise de risco de empréstimos usou uma técnica de aprendizado de máquina chamada XGBoost para avaliar o risco de um empréstimo de US$ 8.000. O resultado foi que a probabilidade de o empréstimo ser "Charged Off" (ou seja, ser considerado perigoso e ser baixado como perda) é de apenas 1,4%. Isso significa que o modelo considera que o empréstimo é extremamente seguro.
>
> O modelo também definiu um limiar de classificação de 9,0%, que é a probabilidade acima da qual o empréstimo é considerado de alto risco. Nesse caso, o modelo classificou o empréstimo como "Fully Paid", o que significa que o empréstimo é considerado seguro e não há sinal de risco elevado.
>
> É importante notar que o modelo considera vários fatores, incluindo o Score FICO, a renda anual, as consultas de crédito nos últimos 6 meses, o valor do empréstimo e a finalidade do empréstimo. No ent

**Verificação automática:** O texto parece ter sido cortado pelo limite de tamanho da resposta.

## Perfil B — perfil mediano, 36 meses

| Campo | Valor |
|---|---|
| Valor solicitado (USD) | US$ 12.000 |
| Renda anual declarada (USD) | US$ 65.000 |
| Relação dívida/renda — DTI (%) | 17,6% |
| Score FICO (limite inferior da faixa) | 690 |
| Consultas de crédito nos últimos 6 meses | 0 |
| Uso do crédito rotativo (%) | 52,0% |
| Prazo | 36 meses |
| Finalidade | Consolidação de dívidas |
| Situação de moradia | Aluguel |
| Tempo de emprego | 3 anos |

**Classificador:** probabilidade 10,9% → classe **Charged Off** (limiar 9,0%).

**Explicação gerada pela LLM** (12.5 s):

> O modelo de análise de risco de empréstimos previu que esse empréstimo tem uma probabilidade de 10,9% de ser "Charged Off", o que significa que o empréstimo pode ser considerado perdedor. Isso ocorre porque a probabilidade é maior do que o limiar de 9,0%, que é o valor acima do qual o modelo considera que o empréstimo é de alto risco. Portanto, o modelo prevê que o empréstimo é de alto risco de perda.

**Verificação automática:** nenhuma inconsistência encontrada.

## Perfil C — FICO no mínimo usual, DTI alto, 60 meses

| Campo | Valor |
|---|---|
| Valor solicitado (USD) | US$ 30.000 |
| Renda anual declarada (USD) | US$ 40.000 |
| Relação dívida/renda — DTI (%) | 35,0% |
| Score FICO (limite inferior da faixa) | 660 |
| Consultas de crédito nos últimos 6 meses | 3 |
| Uso do crédito rotativo (%) | 90,0% |
| Prazo | 60 meses |
| Finalidade | Pequeno negócio |
| Situação de moradia | Aluguel |
| Tempo de emprego | Menos de 1 ano |

**Classificador:** probabilidade 68,1% → classe **Charged Off** (limiar 9,0%).

**Explicação gerada pela LLM** (25.2 s):

> O modelo de análise de risco de empréstimos usou uma técnica de aprendizado de máquina chamada XGBoost para prever a probabilidade de um empréstimo ser "Charged Off", o que significa que o empréstimo será considerado perdedor. O resultado foi que a probabilidade estimada de um empréstimo ser "Charged Off" é de 68,1%.
>
> Isso significa que, de acordo com o modelo, há uma grande chance de que esse empréstimo seja considerado perdedor. A classificação é "ACIMA do limiar", o que significa que o modelo prevê que o empréstimo tem um perfil de maior risco de perda.
>
> É importante notar que isso não significa que o empréstimo será automaticamente aprovado ou rejeitado. O modelo está apenas fornecendo uma previsão baseada nos dados disponíveis. Além disso, é importante lembrar que o modelo mais usa atributos como Score FICO, Renda anual declarada, Consultas de crédito nos últimos 6 meses, Valor solicitado e Finalidade, mas isso não explica o resultado específico

**Verificação automática:** O texto parece ter sido cortado pelo limite de tamanho da resposta.

## Perfil D — tempo de emprego não informado

| Campo | Valor |
|---|---|
| Valor solicitado (USD) | US$ 15.000 |
| Renda anual declarada (USD) | US$ 55.000 |
| Relação dívida/renda — DTI (%) | 22,0% |
| Score FICO (limite inferior da faixa) | 700 |
| Consultas de crédito nos últimos 6 meses | 1 |
| Uso do crédito rotativo (%) | 60,0% |
| Prazo | 36 meses |
| Finalidade | Reforma da casa |
| Situação de moradia | Imóvel próprio quitado |
| Tempo de emprego | Não informado |

**Classificador:** probabilidade 14,6% → classe **Charged Off** (limiar 9,0%).

**Explicação gerada pela LLM** (24.2 s):

> O modelo de análise de risco de empréstimos usou uma técnica de aprendizado de máquina chamada XGBoost para avaliar o risco de um empréstimo de US$ 15.000. O resultado foi que a probabilidade estimada de que o empréstimo seja "Charged Off" (ou seja, que o devedor não pague o empréstimo) é de 14,6%. Isso significa que, com base nos dados fornecidos, o modelo considera que o empréstimo é de alto risco de perda.
>
> A classificação do modelo é "ACIMA do limiar", o que significa que o empréstimo foi classificado como tendo um risco de perda maior do que o estabelecido pelo limiar de 9,0%. Nesse caso, o modelo prevê que o empréstimo seja "Charged Off".
>
> É importante notar que o modelo usa vários atributos, como o Score FICO, a renda anual declarada, as consultas de crédito nos últimos 6 meses, o valor solicitado e a finalidade do empréstimo, para avaliar o

**Verificação automática:** O texto parece ter sido cortado pelo limite de tamanho da resposta.
