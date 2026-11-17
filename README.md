# LendingRisk AI

Protótipo acadêmico para analisar o risco de não pagamento de empréstimos com o **Lending Club Dataset**.

**Dupla:** Hyan da Cunha Ferreira Inácio — CG3034704; Luan Ibanhez de Almeida — CG3034755.

**Disciplina:** Ciência de Dados e Sistemas Inteligentes (CARCDSI) — IFSP Caraguatatuba.

## Objetivo

Comparar três modelos supervisionados (Regressão Logística, Random Forest e XGBoost) para classificar empréstimos já encerrados em `Fully Paid` (0) ou `Charged Off` (1). A interface apresenta uma probabilidade estimada de não pagamento e pode explicá-la com uma LLM local. **Não serve para aprovação automática de crédito.**

## Requisitos

- Python 3.11 ou 3.12 (recomendado)
- Git
- Ollama opcional, para a explicação local com LLM (`llama3.2:3b`)
- Dataset do Lending Club (`accepted_2007_to_2018Q4.csv.gz` ou `.csv`)

**Dataset:** https://www.kaggle.com/datasets/wordsforthewise/lending-club

Se a versão fornecida pelo professor tiver as mesmas colunas, ela também pode ser usada. Coloque **somente o arquivo de empréstimos ACEITOS** na pasta `data/`. Os arquivos grandes em `data/` não são versionados.

## Instalação (Linux / Pop!_OS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se `py -3.12` não existir, use a versão do Python instalada, de preferência 3.11/3.12.

## Treinamento real

Para `.csv.gz` (não precisa descompactar):

```bash
python -m src.training --data data/accepted_2007_to_2018Q4.csv.gz
```

Ou para `.csv`:

```bash
python -m src.training --data data/accepted_2007_to_2018Q4.csv
```

O script lê colunas específicas em blocos para economizar memória, amostra registros de toda a base, mantém apenas resultados encerrados, limpa valores ausentes dentro de `Pipeline` e separa dados **cronologicamente** usando `issue_d` (treino/validação/teste).

Os modelos são comparados na **validação** usando PR-AUC como métrica principal; o limiar é escolhido na validação priorizando recall (F2). O conjunto de teste fica reservado para o modelo selecionado. Não treinar nem ajustar com o teste.

Resultados gerados em `reports/`:

- `comparacao_modelos.csv`: PR-AUC, recall, precision, F1, ROC-AUC, Brier e acurácia na validação;
- `importancia_atributos.csv`: importância **global** por permutação na validação;
- `matriz_confusao_teste.csv`: resultados do modelo final no teste;
- `metricas.json`: resumo de dados, partições, modelos e avaliação final.

O modelo treinado localmente fica em `models/modelo_risco.joblib` (não é enviado ao GitHub).

**Observação:** cada versão do Lending Club pode ter nomes diferentes. O script avisa se faltarem colunas. Os números só são científicos depois de executar no dataset real e revisar os resultados.

## Interface

```bash
python -m streamlit run app.py
```

O navegador abrirá uma interface para informar características do empréstimo e obter o risco estimado.

## LLM local com Ollama

Baixe e instale: https://ollama.com/download

```bash
ollama pull llama3.2:3b
```

O Ollama precisa estar em execução no computador local. No Linux, se o serviço não estiver ativo, execute `ollama serve` em outro terminal (se já estiver ativo, não abra uma segunda instância). A interface faz uma solicitação HTTP local em `127.0.0.1:11434` apenas ao clicar em **Explicar com LLM local**.

A LLM **não prevê o resultado**: apenas comunica a probabilidade retornada pelo modelo. Os atributos apresentados têm importância **global**, não uma explicação causal individual.

## Estrutura

```text
lendingrisk-ai/
├── app.py
├── src/
│   ├── config.py
│   ├── training.py
│   └── llm.py
├── data/                 # adicionar dataset localmente
├── models/               # modelo treinado localmente
├── reports/              # métricas e gráficos/resultados
├── docs/                 # roteiros do artigo e do pitch
├── tests/                # testes básicos
├── requirements.txt
└── README.md
```

## Limites e cuidados

- Só incluir atributos conhecidos no momento da concessão. Colunas de pagamentos posteriores, recuperação e o `last_fico_range_high` causariam vazamento de informação.
- Como os resultados de empréstimos de diferentes datas amadurecem de forma desigual, o conjunto escolhido não representa todos os pedidos nem todos os possíveis desfechos.
- Treinamento usa amostra representativa aleatória do arquivo, e não todo o arquivo; registrar parâmetros e tamanhos na documentação.
- A avaliação é acadêmica e não substitui avaliação humana. Antes de uso real, seriam necessários testes de calibração, viés, estabilidade e monitoramento.
- Nunca carregar arquivos `.joblib` de fontes não confiáveis.
