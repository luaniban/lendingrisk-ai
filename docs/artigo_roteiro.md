# Roteiro de artigo científico (preencher após executar no dataset real)

## 1. Introdução
- Problema: estimar probabilidade de não pagamento a partir de características iniciais do empréstimo.
- Objetivo e relevância; delimitação como protótipo acadêmico.

## 2. Referencial teórico
- Classificação binária; desbalanceamento de classes; vazamento de dados; LLMs como explicação textual.

## 3. Metodologia
- Descrever fonte, período, número de registros e versão do Lending Club Dataset.
- Target: 0 = Fully Paid, 1 = Charged Off; justificar exclusão de Current, Default sem desfecho final etc.
- Listar features de `src/config.py` e decisões de exclusão por leakage.
- Pré-processamento e cortes temporais por `issue_d`.
- Modelos: Regressão Logística, Random Forest, XGBoost; parâmetros em `src/training.py`.
- Critério de seleção PR-AUC (validação) e escolha do threshold por F2 (validação).
- Métricas complementares, matriz de confusão e teste reservado.

## 4. Resultados e análise
- **NÃO INVENTAR VALORES**: copiar resultados reais de `reports/metricas.json` e `reports/comparacao_modelos.csv`.
- Incluir tabela de modelos, PR-AUC, recall, precision, F1, ROC-AUC, Brier, acurácia.
- Discutir falsos positivos, falsos negativos e limitações.
- Mostrar exemplo de saída da interface com LLM local, após executar a aplicação.

## 5. Conclusão
- O que foi implementado e quais limitações foram percebidas.
- Próximos passos: calibração, validação adicional, revisão de atributos, mitigação de viés.

## Referências
- Materiais da disciplina CARCDSI; dataset; documentações das bibliotecas utilizadas.
