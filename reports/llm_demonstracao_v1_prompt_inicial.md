# Demonstração da LLM local (Ollama)

> Perfis HIPOTÉTICOS de entrada (não são mutuários reais). Probabilidades calculadas pelo modelo treinado; textos gerados pela LLM local.

- Classificador: **XGBoost**, limiar 9,0%
- LLM: `llama3.2:3b` via Ollama 0.40.16 (digest `a80c4f17acd5`), temperatura 0,1, execução local em CPU
- Gerado em 2026-11-17T12:48:40.411219+00:00

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

**Explicação gerada pela LLM** (32.5 s):

> O modelo de análise de risco de empréstimos usou um algoritmo chamado XGBoost para avaliar o risco de um empréstimo de US$ 8.000. O resultado foi que o empréstimo foi classificado como "Fully Paid", o que significa que o modelo acredita que o empréstimo não apresenta risco elevado.
>
> O modelo considerou vários fatores, como a renda anual declarada (US$ 120.000), a relação dívida/renda (8,0%), o score FICO (780), o uso do crédito rotativo (15,0%) e a finalidade do empréstimo (quitá- cartão de crédito). Além disso, o modelo também considerou que o empréstimo foi solicitado por uma pessoa que tem um tempo de emprego de 10 anos ou mais e que mora em um imóvel financiado.
>
> O modelo estima que a probabilidade de o empréstimo ser "Charged Off" (baixado como perda) é de 1,4%, o que é muito baixo. O limiar de classificação definido é de 9,0%, o que significa que o modelo considera que o empréstimo está abaixo desse limiar e, portanto, não apresenta risco elevado.
>
> Esta é uma estimativa acadêmica e não substitui a análise humana.

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

**Explicação gerada pela LLM** (29.0 s):

> O modelo de análise de risco de empréstimos usou uma técnica de aprendizado de máquina chamada XGBoost para avaliar o risco de um empréstimo de US$ 12.000. O resultado foi que o empréstimo apresenta uma probabilidade de 10,9% de ser considerado "Charged Off" (empréstimo com perfil de maior risco de perda).
>
> Isso significa que, com base nos dados fornecidos, o modelo considerou que o empréstimo apresentava um risco moderado. No entanto, o limiar de classificação definido foi de 9,0%, o que significa que o empréstimo foi classificado como "ACIMA do limiar" e, portanto, considerado de maior risco.
>
> Os atributos mais importantes do modelo em geral são:
>
> * Score FICO (limite inferior da faixa): 690
> * Renda anual declarada (USD): US$ 65.000
> * Consultas de crédito nos últimos 6 meses: 0
> * Valor solicitado (USD): US$ 12.000
> * Finalidade: Consolidação de dívidas
>
> Esses atributos foram considerados importantes para avaliar o risco do empréstimo. No entanto, é importante notar que esses atributos não explicam o resultado específico deste caso, mas sim fornecem uma ideia geral do que o modelo considerou importante ao

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

**Explicação gerada pela LLM** (25.1 s):

> O modelo de análise de risco de empréstimos usou uma técnica de aprendizado de máquina chamada XGBoost para avaliar o risco de um empréstimo de US$ 30.000. O resultado foi que o modelo prevê que o empréstimo tenha uma probabilidade de 68,1% de ser "Charged Off" (empréstimo baixado como perda).
>
> Isso significa que, de acordo com o modelo, o empréstimo é considerado de alto risco de perda. O modelo também definiu um limiar de classificação de 9,0%, o que significa que qualquer empréstimo com uma probabilidade acima desse valor é considerado de alto risco.
>
> O modelo considerou vários fatores, incluindo a renda anual declarada, o score FICO, o número de consultas de crédito nos últimos 6 meses, o valor solicitado, a finalidade do empréstimo e a situação de moradia. No entanto, é importante notar que esses fatores são apenas uma representação geral do risco e não explicam o caso específico desse empréstimo.
>
> Esta é uma estimativa acadêmica e não substitui a análise humana.

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

**Explicação gerada pela LLM** (29.8 s):

> O modelo de análise de risco de empréstimos usou uma técnica de aprendizado de máquina chamada XGBoost para avaliar o risco de um empréstimo de US$ 15.000. O resultado foi que o empréstimo apresenta uma probabilidade de 14,6% de ser considerado "Charged Off" (perda), o que significa que o empréstimo tem um perfil de maior risco de não ser pago.
>
> O modelo considerou vários fatores, incluindo o valor do empréstimo, a renda anual do solicitante, a relação dívida/renda, o score FICO, o número de consultas de crédito nos últimos 6 meses, o uso do crédito rotativo e o prazo do empréstimo. No entanto, é importante notar que esses fatores não são relevantes para esse caso específico, pois o modelo está apenas fornecendo uma estimativa geral.
>
> O modelo também definiu um limiar de classificação de 9,0%, o que significa que se o empréstimo for acima desse valor, ele será considerado "Charged Off". Nesse caso, o empréstimo está acima do limiar, o que significa que o modelo está indicando que o empréstimo tem um perfil de maior risco de não ser pago.
>
> Esta é uma estimativa acadêmica e não substitui a análise humana.

**Verificação automática:** nenhuma inconsistência encontrada.
