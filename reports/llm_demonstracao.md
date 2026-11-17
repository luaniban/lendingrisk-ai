# Demonstração da LLM local (Ollama)

> Perfis HIPOTÉTICOS de entrada (não são mutuários reais). Probabilidades calculadas pelo modelo treinado; textos gerados pela LLM local.

- Classificador: **XGBoost**, limiar 9,0%
- LLM: `llama3.2:3b` via Ollama 0.40.16 (digest `a80c4f17acd5`), temperatura 0,1, execução local em CPU
- Gerado em 2026-11-17T12:59:29.890435+00:00
- A interface exibe, abaixo de cada explicação: "Esta é uma estimativa acadêmica e não substitui a análise humana."

<details><summary>Prompt de sistema usado</summary>

```text
Você explica, em português do Brasil e para uma pessoa leiga, o resultado de um protótipo acadêmico de análise de risco de empréstimos treinado com dados históricos do Lending Club (EUA, 2007–2015).
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
{"probabilidade": "O modelo estimou em 23,4% a chance de este empréstimo terminar como Charged Off, isto é, ser baixado como perda.", "limiar": "Como 23,4% está acima do limiar de 12,0%, o empréstimo foi classificado como Charged Off, um perfil de maior risco pelo critério adotado.", "atributos": "Em geral, o modelo dá mais peso a atributos como prazo e situação de moradia, mas essa importância é global e não explica este caso específico."}
{"probabilidade": "O modelo estimou em 5,2% a chance de este empréstimo terminar como Charged Off, isto é, ser baixado como perda.", "limiar": "Como 5,2% está abaixo do limiar de 12,0%, o empréstimo foi classificado como Fully Paid, sem sinal de risco elevado pelo critério adotado.", "atributos": "Em geral, o modelo dá mais peso a atributos como prazo e situação de moradia, mas essa importância é global e não explica este caso específico."}
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

**Explicação gerada pela LLM** (18.5 s, 1 tentativa(s)):

> O modelo estimou em 1,4% a chance de este empréstimo terminar como Charged Off, isto é, ser baixado como perda. Como 1,4% está abaixo do limiar de 9,0%, o empréstimo foi classificado como Fully Paid, sem sinal de risco elevado pelo critério adotado. Em geral, o modelo dá mais peso a atributos como Score FICO, Renda anual declarada e Consultas de crédito nos últimos 6 meses, mas essa importância é global e não explica este caso específico.

**Verificação automática:** nenhuma inconsistência encontrada.

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

**Explicação gerada pela LLM** (16.3 s, 1 tentativa(s)):

> O modelo estimou em 10,9% a chance de este empréstimo terminar como Charged Off, isto é, ser baixado como perda. Como 10,9% está acima do limiar de 9,0%, o empréstimo foi classificado como Charged Off, um perfil de maior risco pelo critério adotado. Em geral, o modelo dá mais peso a atributos como Score FICO, Renda anual declarada e Valor solicitado, mas essa importância é global e não explica este caso específico.

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

**Explicação gerada pela LLM** (19.6 s, 1 tentativa(s)):

> O modelo estimou em 68,1% a chance de este empréstimo terminar como Charged Off, o que significa que há uma grande probabilidade de o empréstimo ser baixado como perda. Como 68,1% está acima do limiar de 9,0%, o empréstimo foi classificado como Charged Off, indicando um perfil de maior risco de perda. Em geral, o modelo considera atributos como Score FICO, renda anual, consultas de crédito e valor solicitado, mas essas características não explicam por que este empréstimo foi classificado como de maior risco.

**Verificação automática:** nenhuma inconsistência encontrada.

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

**Explicação gerada pela LLM** (20.8 s, 1 tentativa(s)):

> O modelo estimou em 14,6% a chance de este empréstimo terminar como Charged Off, o que significa que há uma probabilidade moderada de o empréstimo ser baixado como perda. Como 14,6% está acima do limiar de 9,0%, o empréstimo foi classificado como Charged Off, indicando um perfil de maior risco de perda. Em geral, o modelo considera os atributos como Score FICO, renda anual, consultas de crédito, valor solicitado e finalidade, mas esses fatores não explicam a classificação deste caso específico.

**Verificação automática:** nenhuma inconsistência encontrada.
