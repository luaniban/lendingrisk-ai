# Registro de experimentos e decisões metodológicas — LendingRisk AI

Este arquivo registra, **em ordem cronológica**, as decisões metodológicas e os
experimentos executados. Os números citados aqui são copiados dos arquivos em
`reports/` gerados pelas execuções reais. Nada aqui é estimado à mão.

---

## 1. Protocolo (definido em 17/11/2026, ANTES de treinar com o dataset real)

O protocolo foi fixado e versionado no Git antes de qualquer treino com dados
reais, para que as escolhas não fossem influenciadas pelo resultado do teste.

### 1.1 Dados
- **Arquivo:** `accepted_2007_to_2018Q4.csv` (Lending Club, empréstimos aceitos).
- **Fonte primária:** Kaggle `wordsforthewise/lending-club` (CC0). Como o Kaggle exige
  login, o arquivo foi obtido de um espelho público no Hugging Face
  (`codesignal/lending-club-loan-accepted`, CC0, revisão `c435da0`), cujo SHA-256
  (`3eae03c2…f36a`) é idêntico ao de outros dois espelhos independentes. Ver
  `python -m src.download`.
- **Alvo:** `0 = Fully Paid`, `1 = Charged Off`. Os demais status (`Current`, `Late`,
  `In Grace Period`, `Default` e `Does not meet the credit policy. Status:…`) ficam
  fora, porque não têm desfecho final ou seguem outra política de crédito.
- **Features (10):** `loan_amnt`, `annual_inc`, `dti`, `fico_range_low`,
  `inq_last_6mths`, `revol_util`, `term`, `purpose`, `home_ownership`, `emp_length`.
  Todas são conhecidas no momento do pedido. Nenhuma coluna pós-concessão entra no
  modelo (há teste automatizado em `tests/test_data.py`).

### 1.2 Correções feitas no starter (antes do treino real)
| Problema do starter | Correção |
|---|---|
| Piso de 1.000 linhas por bloco: blocos com poucos finalizados eram sorteados acima de 15%, o que distorcia a distribuição por período | Amostragem Bernoulli uniforme: um sorteio por linha do arquivo, independente de bloco e de filtro |
| `average_precision_score` rotulado como `pr_auc` | Renomeado para `average_precision` (AP) |
| Datas lidas com `%b-%Y` (dependem do locale) | Mapa explícito de meses em inglês e aceite de ISO |
| Sem metadados de reprodutibilidade | SHA-256, versões, semente, contagens por etapa e partições com taxa de classe |
| Permutação com 600 linhas × 2 repetições (ruidosa) | 5.000 linhas × 5 repetições |
| Faixas e categorias do formulário fixas no código | Derivadas do conjunto de treino e salvas em `metricas.json` |

### 1.3 Viés de maturidade e regra de filtro
Empréstimos emitidos perto do fim do arquivo (2016–2018) ainda estavam, em sua
maioria, `Current`. Entre eles, os que já têm desfecho final são justamente os que
terminaram **cedo**: quitação antecipada ou baixa precoce. Uma divisão temporal
sem tratamento colocaria essa população atípica no teste.

**Regra adotada:** manter apenas empréstimos cujo prazo contratual já terminou na
data de referência do arquivo (`issue_d + term <= maior issue_d do arquivo`).
A auditoria (`python -m src.audit`) mede o efeito da regra *antes* do treino, sem
usar modelos. A regra pode ser desativada com `--no-maturity-filter`, o que
reproduz o comportamento original.

### 1.4 Divisão, seleção e teste
- **Divisão temporal por `issue_d`:** cerca de 70% treino, 15% validação e 15% teste.
  Um mesmo mês nunca aparece em duas partições.
- **Pré-processamento** (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) dentro do
  `Pipeline`, ajustado só no treino. Há teste que verifica a mediana e a média do
  scaler contra o treino.
- **Três modelos** com os hiperparâmetros iniciais do starter: Regressão Logística,
  Random Forest e XGBoost. Não houve busca de hiperparâmetros.
- **Seleção do modelo:** maior **AP na validação**.
- **Limiar:** maior **F2 na validação** (grade de 0,01 a 0,99). O F2 privilegia o
  recall, pois deixar de sinalizar um Charged Off foi considerado mais custoso que
  um falso alarme. Essa escolha produz muitos falsos positivos, e isso será discutido.
- **Teste:** avaliado **uma única vez**, com o modelo e o limiar já escolhidos.
- **Análises descritivas, que não alteram as escolhas:** calibração (tabela por decis e
  ECE), estabilidade por trimestre, subgrupos por atributo categórico e
  distribuição dos scores.
- **Tamanho dos dados:** usar **toda a população elegível** (`--fraction 1 --max-rows 0`).
  A amostra de 15% com teto de 150 mil linhas do starter era uma escolha para
  economizar memória, não uma exigência metodológica. Com mais dados, as estimativas
  do teste são menos ruidosas. Só se a execução completa não couber na máquina
  (memória ou tempo) a fração será reduzida, e o motivo será registrado. A decisão
  depende apenas do custo computacional, nunca do resultado.
- **Execução oficial:** uma única execução de treino com teste, com os parâmetros
  registrados na seção 3.

---

## 2. Auditoria do dataset real (17/11/2026, sem treinar modelos)

Comando: `python -m src.audit --data data/accepted_2007_to_2018Q4.csv`. A execução levou
30 s e usou no máximo 1,26 GB de RAM. Saídas: `reports/auditoria_dataset.json`,
`reports/auditoria_coortes.csv`, `reports/figures/auditoria_coortes.png` e o log em
`reports/logs/auditoria_2026-11-17.log`.

### 2.1 Arquivo
- `accepted_2007_to_2018Q4.csv`: 1.675.133.810 bytes, SHA-256
  `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a`.
- 151 colunas e 2.260.701 linhas lidas. Entre elas estão 2 linhas de rodapé
  ("Total amount funded in policy code…") e 33 linhas sem status/data, que são descartadas.
- Emissões (`issue_d`) de 2007-06 a 2018-12. Todas as 10 features existem no arquivo e
  nenhuma delas está na lista de colunas de vazamento. O arquivo contém 20 colunas
  pós-concessão (ex.: `total_pymnt`, `recoveries`, `last_fico_range_high`), que **não**
  são lidas.

### 2.2 Status (`loan_status`)
| Status | Linhas |
|---|---:|
| Fully Paid | 1.076.751 |
| Current | 878.317 |
| Charged Off | 268.559 |
| Late (31-120 days) | 21.467 |
| In Grace Period | 8.436 |
| Late (16-30 days) | 4.349 |
| Does not meet the credit policy. Status:Fully Paid | 1.988 |
| Does not meet the credit policy. Status:Charged Off | 761 |
| Default | 40 |
| vazio (rodapé/linhas inválidas) | 33 |

Com desfecho final há 1.345.310 linhas, das quais **19,96%** são Charged Off. Os 2.749
casos "Does not meet the credit policy" (quase todos de 2007–2010) seguem outra
política de crédito e foram excluídos. Por isso as coortes de 2007–2010 aparecem
com menos de 100% "finalizado" em `auditoria_coortes.csv`.

### 2.3 Evidência do viés de maturidade
Data de referência: 2018-12-01 (maior `issue_d`).

| Grupo | Empréstimos | % com desfecho final | Charged Off entre finalizados |
|---|---:|---:|---:|
| Coortes maturadas (prazo já encerrado) | 676.459 | 99,57% | 14,82% |
| Coortes não maturadas | 1.584.209 | 42,40% | 25,12% |

Nas coortes recentes, quem já tem desfecho é desproporcionalmente Charged Off,
porque a baixa precoce aparece antes da quitação no prazo. Exemplo: 60 meses emitidos em
2015 têm só 67% finalizados, e entre eles 36,4% são Charged Off. Já 36 meses emitidos em
2015, praticamente todos finalizados, têm 14,9%.

**Efeito na divisão temporal** (todos os finalizados, sem amostragem):

| Política | Treino | Validação | Teste |
|---|---|---|---|
| Sem filtro (starter) | 938.537 linhas; 2007-06 a 2016-03; 18,66% CO | 197.556; 2016-04 a 2017-01; **24,89%** CO | 209.217; 2017-02 a 2018-12; 21,17% CO |
| **Com filtro de maturidade** | 470.714; 2007-06 a 2015-04; 14,81% CO | 95.405; 2015-05 a 2015-08; 14,81% CO | 107.434; 2015-09 a 2015-12; 14,84% CO |

Sem o filtro, a taxa de Charged Off muda de 18,7% no treino para 24,9% na validação.
Essa mudança vem da seleção de quem terminou cedo, não de uma mudança real no risco.
Com o filtro, as três partições têm a mesma taxa (~14,8%). **Decisão: manter o
filtro de maturidade**, como previsto no protocolo.

**Limitações dessa decisão, registradas antes do treino:**
1. Validação e teste cobrem só 4 meses cada (2015-05 a 2015-08 e 2015-09 a 2015-12).
   Por isso a estabilidade passa a ser analisada por **mês**, e não por trimestre.
2. Validação e teste contêm **apenas empréstimos de 36 meses**. Nenhum empréstimo de 60
   meses emitido depois de 2013-12 tinha prazo encerrado. O modelo aprende o efeito de
   60 meses com coortes de 2010–2013 (11,7% do treino), mas esse efeito **não é
   avaliado** no teste. A interface deve deixar isso claro.
3. A população analisada são empréstimos de 2007–2015 com prazo encerrado. Os resultados
   não representam as safras de 2016–2018.

### 2.4 Qualidade dos dados (finalizados)
- Ausentes: `emp_length` 5,84%, `revol_util` 0,064%, `dti` 0,028%; demais 0%.
- Valores extremos mantidos (o modelo imputa ausentes, mas não corta extremos):
  `dti` de −1 a 999; `revol_util` até 892,3; `annual_inc` de 0 a 10,99 milhões.
  Os percentis de 1% e 99% são usados pela interface para avisar quando uma entrada
  foge da faixa usual.
- `home_ownership` tem categorias raras: ANY (286), OTHER (144) e NONE (48).
- `loan_amnt` vai de US$ 500 a US$ 40.000, e `fico_range_low` de 625 a 845.

---

## 3. Execução oficial de treinamento (17/11/2026)

Comando: `python -m src.training --data data/accepted_2007_to_2018Q4.csv`, com os padrões
fração 1,0, sem teto, filtro de maturidade ativo e semente 42.
Ambiente: Python 3.12.16, scikit-learn 1.9.1, XGBoost 3.4.1, pandas 2.3.3 e numpy 2.5.5,
num notebook com Intel Tiger Lake (8 threads) e 15 GiB de RAM, sem GPU.
A execução levou **2 min 28 s** e usou no máximo **1,17 GB** de RAM (a segunda execução: 2 min 45 s e 1,18 GB). Logs em
`reports/logs/treino_oficial_2026-11-17*.log`.

**Reprodutibilidade:** o treino foi executado duas vezes com o mesmo comando. A segunda
execução só refez os gráficos, com ajustes de formatação. As métricas das duas são
idênticas (ver `reports/historico_execucoes.csv`).

### 3.1 Dados usados
| Etapa | Linhas |
|---|---:|
| Lidas do arquivo | 2.260.701 |
| Com desfecho final (Fully Paid / Charged Off) e data válida | 1.345.310 |
| Removidas pelo filtro de maturidade | 671.757 |
| **Usadas no experimento** | **673.553** (14,82% Charged Off) |

| Partição | Período (`issue_d`) | Linhas | Charged Off | 60 meses |
|---|---|---:|---:|---:|
| Treino | 2007-06 a 2015-04 | 470.714 | 14,81% | 11,68% |
| Validação | 2015-05 a 2015-08 | 95.405 | 14,81% | 0% |
| Teste | 2015-09 a 2015-12 | 107.434 | 14,84% | 0% |

### 3.2 Comparação na validação (critério de seleção: AP)
Fonte: `reports/comparacao_modelos.csv`. AP de um classificador aleatório = 0,148.

| Modelo | AP | ROC-AUC | Brier | Log loss | ECE | Limiar F2 | Recall | Precision | Treino (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística | 0,2331 | 0,6529 | 0,1218 | 0,4025 | 0,0129 | 0,10 | 0,860 | 0,180 | 3,6 |
| Random Forest | 0,2323 | 0,6545 | 0,1219 | 0,4013 | 0,0158 | 0,10 | 0,876 | 0,179 | 105,5 |
| **XGBoost** | **0,2368** | **0,6583** | 0,1214 | 0,3997 | 0,0125 | 0,09 | 0,887 | 0,178 | 6,0 |

**Modelo escolhido: XGBoost**, com limiar **0,09**. A vantagem sobre os outros é pequena
(+0,004 de AP sobre a Regressão Logística). Com só 10 atributos, os três modelos
capturam quase o mesmo sinal, e a Regressão Logística seria uma alternativa quase
equivalente e mais interpretável.

### 3.3 Avaliação final no teste (executada uma vez)
Fonte: `reports/metricas.json` → `metricas_teste`.

| Métrica | XGBoost (teste) | Referência trivial |
|---|---:|---:|
| Average Precision (AP) | **0,2425** | 0,1484 (aleatório) |
| ROC-AUC | **0,6623** | 0,5 |
| Brier | 0,1214 | 0,1264 (prever sempre 14,84%) |
| Log loss | 0,3994 | — |
| ECE (10 faixas) | 0,0155 | — |
| Recall | 0,8805 | — |
| Precision | 0,1798 | 0,1484 |
| F1 / F2 | 0,2986 / 0,4948 | — |
| Acurácia | 0,3862 | 0,8516 (sempre Fully Paid) |
| Taxa sinalizada como Charged Off | 72,68% | — |

Matriz de confusão (teste, 107.434 empréstimos):

| | Previsto Fully Paid | Previsto Charged Off |
|---|---:|---:|
| **Real Fully Paid** | 27.450 | 64.043 |
| **Real Charged Off** | 1.905 | 14.036 |

### 3.4 Interpretação
- **Desempenho modesto, mas acima do acaso e estável.** AP de 0,24 contra 0,15 do
  aleatório, e ROC-AUC de 0,66. Validação e teste dão valores próximos (0,237 e 0,243), e
  a AP mensal varia pouco (`estabilidade_temporal.csv`). O limite vem dos 10 atributos
  escolhidos no Entregável 2. Variáveis como `grade` e `int_rate`, também conhecidas na
  concessão, ficaram de fora porque já são a avaliação de risco do próprio Lending Club.
- **O limiar por F2 é agressivo.** Ele sinaliza 73% dos empréstimos para capturar 88% dos
  Charged Off. Para cada Charged Off detectado há 4,6 falsos alarmes (precision de 18%).
  Por isso a acurácia (38,6%) fica abaixo da de "sempre Fully Paid" (85,2%), que não
  detecta nenhum caso. Na prática, a saída serve para triagem e revisão humana, não para
  decisão. A curva em `figures/limiar_validacao.png` mostra o trade-off. Um limiar maior
  reduziria os alarmes ao custo de recall. Essa troca **não** foi feita, para manter o
  protocolo e não ajustar nada olhando o teste.
- **A calibração é boa, com leve subestimação.** O Brier é melhor que o da referência
  constante e o ECE é de 1,5 p.p. No teste, a probabilidade média prevista é 13,3%
  contra 14,8% observado. Os decis mais altos ficam cerca de 2 p.p. abaixo do observado
  (`calibracao_teste.csv`). As probabilidades **não** foram recalibradas.
- **Importância global (permutação na validação):** FICO > renda anual > consultas em 6
  meses > valor > finalidade > DTI ≈ moradia. `term` aparece com 0 porque a validação só
  tem 36 meses: embaralhar uma coluna constante não muda nada, o que não prova
  irrelevância. `emp_length` e `revol_util` ficam perto de zero ou levemente negativos,
  dentro do ruído.

### 3.5 Análise de erros por subgrupo (teste, descritiva)
Fonte: `reports/analise_subgrupos_teste.csv`.
- **Moradia:** a taxa de falso positivo é de 58,6% para MORTGAGE e de 82,4% para RENT.
  Isso acompanha a taxa real de Charged Off (12,0% e 17,6%), mas mostra que o limiar
  único sinaliza a grande maioria dos locatários. É um ponto de atenção ética: o modelo
  reproduz diferenças históricas entre grupos.
- **Finalidade:** `small_business` tem 23,3% de Charged Off real e 95,9% de falso positivo;
  `credit_card` tem 12,2% e 57,9%.
- **`emp_length` ausente:** a taxa real é de **21,7%**, mas o score médio é de 16,3%. O
  modelo subestima esse grupo. A causa provável é a imputação pela moda (`10+ years`),
  justamente a categoria de menor risco, o que apaga o sinal de que "não informado" é
  mais arriscado. **Melhoria futura:** tratar ausência como categoria própria
  (`SimpleImputer(strategy="constant")`). Essa mudança **não** foi aplicada ao modelo
  oficial, porque foi notada no teste e aplicá-la agora contaminaria a avaliação.

### 3.6 Limitações e próximos passos
1. A avaliação cobre só 36 meses e emissões de 2015 (efeito do filtro de maturidade).
2. Não houve busca de hiperparâmetros. Um próximo passo seria uma busca na validação,
   ou validação cruzada temporal dentro do treino.
3. Com o filtro, as probabilidades refletem as safras de 2007–2015 e podem não valer para
   outras épocas.
4. Faltam testes de justiça com métricas formais. A análise de subgrupos é só descritiva.
5. Para uma próxima versão: tratar ausentes como categoria, avaliar `grade`/`int_rate`
   com aprovação da dupla e recalibrar com dados separados.

---

## 4. Integração e teste da LLM local (17/11/2026)

**Ambiente:** Ollama 0.40.16, instalado sem sudo como serviço do usuário e escutando
só em 127.0.0.1. Modelo `llama3.2:3b`: digest `a80c4f17acd5…`, Q4_K_M, 2,0 GB.
Executa na CPU (Intel Tiger Lake, sem GPU). Temperatura 0,1 e semente 42.

**Garantias de projeto:**
- A LLM recebe a probabilidade, o limiar e a classe **já calculados** e não tem acesso
  ao modelo.
- O app mostra os números vindos do classificador, e não da LLM.
- A demonstração confere que a probabilidade recalculada depois da LLM é idêntica.
- Se o Ollama falhar, o app continua funcionando e mostra o motivo. Isso é testado com
  mocks em `tests/test_llm.py` e `tests/test_app.py`.

**Verificação automática (`check_consistency`, heurística):** aponta probabilidade ou
limiar não citados, percentuais não fornecidos, classe contraditória ("abaixo" quando
está acima), texto cortado, possível atribuição causal ("aumentou o risco", "devido a…",
"contribuiu para") e termos absolutos ("sem risco", "extremamente"). Se houver erro
factual, `explain_checked` gera uma segunda resposta (temperatura 0,4) e fica com a que
tiver menos avisos. Os avisos aparecem na interface.

**Evolução do prompt, com 4 perfis HIPOTÉTICOS** (não são mutuários reais).
As probabilidades são do modelo oficial: A 1,4%, B 10,9%, C 68,1% e D 14,6%; limiar 9,0%.

| Versão | Estratégia | Resultado observado | Evidência |
|---|---|---|---|
| v1 | Texto livre, "até 120 palavras" | Números certos, mas cerca de 200 palavras, lista, 1 texto cortado e leitura errada da importância ("fatores não são relevantes para esse caso") | `reports/llm_demonstracao_v1_prompt_inicial.md` |
| v2 | "EXATAMENTE 3 frases" | O modelo ignorou a estrutura: ~150 palavras e 3 de 4 textos cortados (o verificador detectou os 3) | `reports/llm_demonstracao_v2_tres_frases.md` |
| v3 | Saída estruturada (JSON Schema, 3 campos) | Tamanho resolvido, mas omitiu o limiar e fez atribuição causal ("seguro devido à situação de moradia") | teste manual (log da sessão) |
| v4 | v3 + 2 exemplos (few-shot) | Erro de comparação no perfil B ("10,9% está abaixo do limiar de 9,0%") e termo absoluto no A. **O verificador detectou os dois.** | teste manual (log da sessão) |
| **v5 (atual)** | v4 + comparação pronta no prompt, proibição de qualificar os valores do cliente, exemplos com atributos diferentes dos reais e segunda geração em erro crítico | **4/4 corretos** (números, comparação e classe), 0 avisos, 1 tentativa cada, 72–84 palavras, 16–21 s por explicação | `reports/llm_demonstracao.md` e `.json`, `reports/logs/llm_demo_v5_2026-11-17.log` |

Os exemplos do prompt v5 usam atributos de propósito diferentes dos reais ("prazo e
situação de moradia"). Assim dá para conferir que a LLM cita os atributos realmente
fornecidos (FICO, renda, consultas, valor, finalidade) e não copia o exemplo.

**Limitações:**
- A verificação é heurística. Juízos sobre o caso ("renda alta", "score baixo")
  nem sempre são detectados.
- O ganho de confiabilidade veio de restringir a forma: a v5 é quase um preenchimento
  guiado. Com um modelo de 3B em CPU, essa foi a troca escolhida entre fluência e
  fidelidade.
- Cada explicação leva de 16 a 21 s em CPU. A primeira chamada inclui carregar o modelo,
  cerca de 30 s.
