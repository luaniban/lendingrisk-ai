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

## 2. Auditoria do dataset real

*(preenchida após `python -m src.audit` — ver seção abaixo)*

---

## 3. Execuções de treinamento

*(preenchida após o treino real — números copiados de `reports/metricas.json`)*
