# LendingRisk AI — Contexto completo e instruções para Claude Code / Codex

> **LEIA ESTE ARQUIVO INTEIRO ANTES DE ALTERAR O PROJETO.**
>
> Este documento é um briefing de implementação para um agente de programação. Ele acompanha o pacote inicial `lendingrisk-ai-starter.zip`. **A base já possui código, mas não é uma versão final pronta para entrega.** Não assuma que modelos foram treinados com dados reais nem que o resultado científico foi validado.
>
> **Prioridade:** desenvolver e testar o sistema em Python; **deixar artigo e pitch para a etapa final**, depois dos experimentos reais. Trabalhar de maneira incremental, preservar o que funciona, registrar decisões e apresentar evidências de execução.

## 1. Identificação e contexto acadêmico

- **Instituição:** Instituto Federal de São Paulo (IFSP), câmpus Caraguatatuba.
- **Curso:** Tecnologia em Análise e Desenvolvimento de Sistemas.
- **Disciplina:** Ciência de Dados e Sistemas Inteligentes (CARCDSI).
- **Dupla:**
  - **Hyan da Cunha Ferreira Inácio — CG3034704**.
  - **Luan Ibanhez de Almeida — CG3034755**.
- **Projeto:** **LendingRisk AI — Sistema Inteligente de Análise de Risco de Empréstimos**.
- **Dataset escolhido:** Lending Club Dataset.
- **Repositório sugerido:** `lendingrisk-ai` (GitHub público, contendo o código e as instruções, sem incluir o dataset pesado).

### Exigência do projeto final informada pelo professor

O projeto deve integrar **pré-processamento de datasets, técnicas de Machine Learning e Large Language Models (LLMs)**. O desenvolvimento deve ocorrer em **Python**, e o código deve ser disponibilizado em **repositório público no GitHub**, de maneira organizada e documentada.

As entregas finais previstas no documento da disciplina são:

1. **Pitch em vídeo**, de **3 a 5 minutos**: problema, solução, técnicas e resultados.
2. **Artigo científico**: problema, metodologia, desenvolvimento, experimentos e discussão dos resultados.
3. **Repositório final no GitHub**: código Python, `requirements.txt` e instruções que permitam entender, configurar e reproduzir o projeto; o *final release* deve ser informado no EcAD conforme orientação do professor.

**Cronograma visto no EcAD em 17/11/2026:**

- **19/11/2026:** “Projeto Final”;
- **26/11/2026:** “Projeto Final”;
- **02/12/2026:** “Entrega do Projeto Final”.

**Atenção:** o print não especifica se 19/11 e 26/11 são aulas, acompanhamento ou apresentações. Adotar **02/12 como prazo indicado no ambiente**, mas confirmar com o professor possíveis exigências anteriores. Almejar demonstração funcional até 26/11.

## 2. Histórico das entregas anteriores: NÃO MUDAR O TEMA

### Entregável 1 — definição do sistema

A dupla definiu o uso do **Lending Club Dataset**, com o target derivado de `loan_status`, para separar empréstimos pagos de empréstimos que terminaram com perda/não pagamento. A ideia foi usar dados históricos para estimar o risco de um empréstimo e uma LLM para apresentar a análise em linguagem compreensível.

No texto da primeira entrega, a expressão usada foi “risco de descumprimento”. Na implementação, mantenha os nomes técnicos das classes (`Fully Paid` e `Charged Off`), mesmo que a interface use termos simples em português. Evite interpretar os rótulos como uma representação perfeita de qualquer atraso: o target mede **o desfecho histórico específico**, não qualquer parcela paga fora do prazo.

### Entregável 2 — modelos e métricas

A proposta metodológica já definiu:

- **Classificação binária supervisionada**: `0 = Fully Paid` (empréstimo quitado); `1 = Charged Off` (empréstimo baixado como perda). Estados sem desfecho consolidado ficam fora da classificação binária; validar cuidadosamente qualquer ampliação dessa regra.
- **Três modelos:** Regressão Logística (*baseline*), Random Forest e XGBoost.
- **Métrica principal proposta:** PR-AUC / *Average Precision* (AP), importante quando há desbalanceamento.
- **Outras métricas:** recall, precision, F1, ROC-AUC, Brier score e acurácia; analisar também a matriz de confusão.
- **Tratamento dos dados:** imputação, codificação de variáveis categóricas e padronização quando cabível, sem vazamento de informação.
- **Separação:** treinamento, validação e teste independentes; preferencialmente **temporal**, utilizando `issue_d` e cuidando da maturidade dos empréstimos.
- **Escolha do modelo:** comparar capacidade preditiva, recall/precision, estabilidade, calibração e interpretação; reservar o conjunto de teste para uma avaliação final.
- **Uso da LLM:** explicação textual do score calculado pelo classificador; **a LLM não calcula, substitui ou altera a probabilidade**.

**Compatibilidade com o starter:** a implementação atual seleciona o melhor modelo pela **Average Precision** da validação e escolhe o limiar de classificação maximizando **F2** também na validação. A metodologia do trabalho deve deixar essa regra explícita e justificar suas limitações; aprimorar calibração/estabilidade se houver tempo e evidências.

## 3. Funcionamento esperado do produto

Fluxo:

```text
Lending Club CSV/CSV.GZ
   -> leitura e auditoria do dataset
   -> filtro de rótulos finais e pré-processamento sem leakage
   -> divisão temporal treino/validação/teste
   -> treino de Regressão Logística, Random Forest e XGBoost
   -> comparação das métricas e seleção do modelo
   -> avaliação FINAL no teste reservado
   -> modelo + métricas + gráficos + relatório
   -> interface Streamlit para novos exemplos
   -> previsão de probabilidade e classe
   -> LLM local (Ollama) para explicar o resultado SEM inventar causas
```

O usuário da interface informa características de um empréstimo, e o classificador devolve uma **probabilidade estimada de `Charged Off`** e a classificação perante um limiar estabelecido na validação. A LLM local recebe essa saída e redige uma explicação em português. **Não utilizar o sistema para decisões automáticas reais de aprovação de crédito.**

A interface deve ser simples, clara, coerente com as variáveis efetivamente usadas pelo modelo, e apresentável numa demonstração acadêmica.

## 4. O que já existe no ZIP inicial — auditoria dos arquivos

O conteúdo foi conferido diretamente no `lendingrisk-ai-starter.zip` em 17/11/2026:

```text
lendingrisk-ai-starter/
├── app.py                       # interface Streamlit inicial
├── src/
│   ├── __init__.py
│   ├── config.py                # features, target, caminhos e rótulos
│   ├── training.py              # leitura, preparação, treino, seleção, avaliação
│   └── llm.py                   # chamada à API local do Ollama
├── tests/
│   └── test_basics.py           # testes unitários iniciais
├── docs/
│   ├── artigo_roteiro.md        # ESBOÇO; não é o artigo final
│   └── pitch_roteiro.md         # ESBOÇO; não é o pitch final
├── data/.gitkeep                # NÃO contém o dataset
├── models/.gitkeep              # NÃO contém modelo treinado
├── reports/.gitkeep             # NÃO contém resultados reais
├── requirements.txt
├── .gitignore
└── README.md
```

### Estado verdadeiro do código

**Presente no starter:**

- Leitura do CSV/CSV.GZ em blocos e amostragem configurável (padrão: fração `0.15`, máximo `150000` registros).
- Filtro dos rótulos `Fully Paid` e `Charged Off` e conversão para 0/1.
- Conversão de atributos numéricos, limpeza básica, data de emissão e divisão temporal de treino/validação/teste.
- `Pipeline` de pré-processamento com `SimpleImputer`, `StandardScaler` e `OneHotEncoder`.
- Três classificadores com hiperparâmetros iniciais.
- Métricas de avaliação, seleção por Average Precision na validação e limiar por F2.
- Geração de arquivos de métricas e importância global dos atributos por permutação.
- Interface inicial em Streamlit com formulário e chamada `predict_proba`.
- Função que solicita explicação à LLM por API local do Ollama (`http://127.0.0.1:11434/api/generate`).
- Três testes unitários **básicos** no `tests/test_basics.py`.

**Ainda NÃO está demonstrado no starter:**

- Treinamento com o **Lending Club real**;
- métricas reais, modelo final treinado, gráficos finais e análise de erros;
- execução ponta a ponta da interface com o dataset real;
- funcionamento da LLM local na máquina de quem vai apresentar;
- análise robusta de estabilidade, calibração, subgrupos e possíveis vieses;
- testes suficientes para afirmar que a aplicação está pronta;
- artigo científico preenchido com resultados e pitch gravado.

**Não inventar resultados de treino nem afirmar que algo foi executado sem log/evidência.** Usar dados sintéticos apenas para testar código, com identificação explícita de que são sintéticos.

## 5. Instalação para quem vai executar o projeto

### 5.1. Ferramentas

- Git: <https://git-scm.com/downloads>
- Python **3.11 ou 3.12** (preferência do starter): <https://www.python.org/downloads/>
- VS Code (opcional): <https://code.visualstudio.com/>
- Ollama para LLM local: <https://ollama.com/download>
- Acesso ao Kaggle para obter os dados: <https://www.kaggle.com/datasets/wordsforthewise/lending-club>

A máquina precisa de espaço disponível para baixar e processar uma base grande; o CSV compactado tem centenas de MB e descompactado pode ocupar mais de 1 GB. O código trabalha em blocos e amostra dados para reduzir custo de memória, mas ainda é necessário testar de acordo com o hardware local.

### 5.2. Extrair e abrir a pasta

1. Extrair `lendingrisk-ai-starter.zip`.
2. Entrar na pasta `lendingrisk-ai-starter/` (ou renomeá-la para `lendingrisk-ai/`).
3. Abrir um terminal **na raiz**, onde estão `requirements.txt`, `app.py` e a pasta `src/`.
4. Deixar este documento `.md` na raiz do repositório ou em `docs/` para o agente consultar.

Linux:

```bash
unzip lendingrisk-ai-starter.zip
cd lendingrisk-ai-starter
```

Windows: extrair pelo Explorador de Arquivos e abrir a pasta no VS Code/terminal.

### 5.3. Ambiente Python — Linux / Pop!_OS / Ubuntu

Se faltar `venv`, instalar o pacote adequado à distribuição. Em Debian/Ubuntu/Pop!_OS, normalmente:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
```

Na raiz do projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Se o Python padrão da máquina não for 3.11/3.12, criar o ambiente com a versão correta já instalada, por exemplo `python3.12 -m venv .venv`. **Não alterar Python global ou remover instalações existentes sem autorização.**

### 5.4. Ambiente Python — Windows / PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Se o Windows não tiver Python 3.12, usar uma versão compatível instalada e documentar a escolha. Se a política do PowerShell impedir a ativação, é possível usar diretamente `.\.venv\Scripts\python.exe` nos comandos, sem alterar políticas globais.

### 5.5. Dataset real do Lending Club

- Página pública: <https://www.kaggle.com/datasets/wordsforthewise/lending-club>
- Arquivo esperado: **`accepted_2007_to_2018Q4.csv.gz`** (ou correspondente `.csv`).
- Colocar em `data/accepted_2007_to_2018Q4.csv.gz`.
- Se o download vier em um ZIP, extrair o **ZIP externo**, sem necessidade de descompactar o `.csv.gz`.
- **Não usar o CSV de pedidos rejeitados** como se fosse o de empréstimos aceitos.
- Pode-se usar uma versão fornecida pelo professor, **desde que** seu esquema de colunas e suas características sejam verificados; registrar a diferença de versão.
- O starter **não baixa automaticamente** esse dataset; não afirmar que a base já está no projeto.
- **Não commitar** dataset completo em GitHub. Evitar guardar dados sensíveis de mutuários em logs e capturas desnecessárias.

Antes de treinar, verificar se existem as colunas em `src/config.py` e medir quantidade total de linhas, classes e períodos disponíveis. Registrar a origem, nome do arquivo, período, critérios de filtragem e, se útil, hash SHA-256 para reprodutibilidade.

### 5.6. Treinar os modelos

Executar **da raiz do projeto**, com ambiente Python ativado:

```bash
python -m src.training --data data/accepted_2007_to_2018Q4.csv.gz
```

Se o dataset estiver em `.csv`:

```bash
python -m src.training --data data/accepted_2007_to_2018Q4.csv
```

É possível controlar a amostra, sem confundi-la com a base integral:

```bash
python -m src.training --data data/accepted_2007_to_2018Q4.csv.gz --fraction 0.15 --max-rows 150000
```

Saídas esperadas **após o treinamento real ter sucesso**:

```text
models/modelo_risco.joblib
reports/metricas.json
reports/comparacao_modelos.csv
reports/importancia_atributos.csv
reports/matriz_confusao_teste.csv
```

Interpretar cada métrica no conjunto correto (validação ou teste). **Não usar o conjunto de teste para escolher modelo, limiar ou hiperparâmetros.** Se o dataset não existir ou as colunas divergirem, diagnosticar o motivo: não criar números ou um CSV falso como substituto silencioso.

### 5.7. Executar a interface

Após existir modelo treinado e `reports/metricas.json`:

```bash
python -m streamlit run app.py
```

O Streamlit normalmente abre uma URL local no navegador, em geral `http://localhost:8501` (porta sujeita à disponibilidade e configuração). Testar a submissão do formulário, a probabilidade, a classificação e os relatórios exibidos.

### 5.8. Executar a LLM local com Ollama

Instalar o Ollama pelo site oficial. Então:

```bash
ollama pull llama3.2:3b
```

Verificar que o Ollama está executando. Em instalações onde o serviço ainda não está ativo, pode ser necessário, em outro terminal:

```bash
ollama serve
```

**Não iniciar uma segunda instância se o serviço já estiver rodando.** O código atual usa o endpoint local `127.0.0.1:11434` e o modelo `llama3.2:3b`. A LLM só deve ser acionada pelo botão **“Explicar com LLM local (Ollama)”**.

Para o trabalho final, a integração de LLM é **parte do escopo da disciplina**: o aplicativo pode continuar mostrando a previsão caso o Ollama falhe, mas a equipe precisa demonstrar que a integração realmente foi implementada e testada. Documentar eventual dependência de hardware ou impossibilidade verificada.

## 6. Features (atributos) e target existentes — manter consistentes

O arquivo `src/config.py` define:

**Target**:

```python
TARGET = "loan_status"
CLASS_MAP = {"Fully Paid": 0, "Charged Off": 1}
```

**Atributos numéricos**:

- `loan_amnt`: valor do empréstimo;
- `annual_inc`: renda anual declarada;
- `dti`: relação dívida/renda, conforme a definição da base;
- `fico_range_low`: limite inferior da faixa FICO;
- `inq_last_6mths`: número de consultas recentes de crédito;
- `revol_util`: utilização de crédito rotativo.

**Atributos categóricos**:

- `term`: prazo do empréstimo;
- `purpose`: finalidade;
- `home_ownership`: situação de moradia;
- `emp_length`: tempo de emprego.

**Coluna temporal para divisão (NÃO é feature de predição):** `issue_d`.

**Risco de vazamento:** não usar atributos que só são conhecidos **depois** da concessão/ao longo do pagamento, como `total_pymnt`, `recoveries`, `last_pymnt_d` ou `last_fico_range_high`. Não incluir o próprio `loan_status` na entrada do modelo. Verificar eventuais *proxies* de informações futuras caso sejam adicionadas novas features.

A interface deve enviar ao modelo **exatamente as features e formatos treinados**. Se a lista de atributos mudar, modificar `src/config.py`, tratamento, formulário, testes e documentação em conjunto.

## 7. Plano de execução para o agente — ORDEM DE PRIORIDADE

### Etapa P0 — Diagnóstico e execução fiel do que já existe (primeiro)

1. Ler `README.md`, `requirements.txt`, `src/config.py`, `src/training.py`, `src/llm.py`, `app.py` e `tests/test_basics.py`.
2. Descrever objetivamente o estado do projeto e possíveis problemas encontrados. **Não refazer tudo sem necessidade.**
3. Configurar um ambiente isolado, instalar dependências e executar `python -m unittest discover -s tests -v`.
4. Verificar versões, imports, sintaxe, compatibilidade entre arquivos e comandos.
5. Verificar se o CSV/CSV.GZ real está em `data/`. Se não estiver, registrar a pendência de maneira explícita. **Não fingir treino real.**
6. Checar se `app.py` consegue carregar modelos/artefatos sem divergência de schemas.
7. Corrigir bugs reais encontrados, com testes de regressão.

**Aceite P0:** instalação reproduzível, testes básicos executáveis, diagnóstico claro e ausência de erro óbvio de integração.

### Etapa P1 — Validar e aprimorar o Machine Learning (MAIOR PRIORIDADE)

1. Usar o **dataset real**, verificar colunas, dimensões, rótulos, percentuais de classes e datas.
2. Investigar validade dos rótulos e o efeito da exclusão de `Current` e outros estados (risco de viés por maturidade/seleção).
3. Inspecionar a estratégia de **amostragem por blocos** (`sample_fraction=0.15`, `max_rows=150000`): se há distorção de distribuição por período/classe e se os dados são suficientes para o teste temporal.
4. Examinar se cortes em `issue_d` produzem períodos distintos e amostras suficientes para as duas classes. Registrar datas exatas e proporções por partição.
5. Garantir que `Pipeline` é ajustado **somente no treino**; nada de calcular imputação/escala/encoding usando validação ou teste.
6. Treinar **Regressão Logística, Random Forest e XGBoost**; manter os três, pois constam no Entregável 2.
7. Comparar **Average Precision (AP), recall, precision, F1, ROC-AUC, Brier e acurácia**. `average_precision_score` mede **AP**; usar a nomenclatura correta e não confundir automaticamente com a área trapezoidal da curva PR.
8. Registrar critérios de seleção e escolha de limiar, atualmente AP para o modelo e F2 para o limiar, ambos na validação. Discutir trade-off de falsos positivos e falsos negativos.
9. Avaliar **uma vez** o modelo final no teste reservado. Fazer análise de erros e matriz de confusão.
10. Testar **calibração** da probabilidade, desbalanceamento e estabilidade temporal se os dados permitirem. Não anunciar uma probabilidade como rigorosamente calibrada sem comprovação.
11. Gerar tabelas e gráficos claros a partir dos **números reais**, preferencialmente salvando PNGs em `reports/figures/` e registrando os parâmetros.
12. Registrar semente aleatória, versão de dependências, amostragem, período do dataset, políticas de exclusão, features e quantidade de linhas por partição.

**Aceite P1:** treinamento real concluído, três modelos efetivamente comparados, validação/limiar documentados, modelo e relatórios produzidos e resultados de teste preservados corretamente.

### Etapa P2 — Melhorar a interface Streamlit

1. Preservar o propósito e facilitar a demonstração acadêmica.
2. Campos claros em português; explicar as unidades (`USD`, renda **anual**, percentuais) e o significado das categorias.
3. Validar valores e lidar com categorias desconhecidas ou configurações incompatíveis.
4. Apresentar **probabilidade estimada, classe perante o limiar, modelo escolhido, contexto e limitações**.
5. Disponibilizar comparação de modelos, métricas e gráficos em seção organizada (abas/expansores, se útil).
6. Evitar apresentar *importância global* como “causa da previsão deste usuário”; distinguir explicação global de local.
7. Tratar casos sem modelo, relatórios corrompidos, previsão inválida e problemas de sessão com mensagens úteis.
8. Verificar acessibilidade básica, responsividade e execução sem acesso a serviços externos (exceto download inicial dos componentes).

**Aceite P2:** alguém consegue abrir o app, inserir um exemplo, receber o resultado, compreender os limites e ver métricas reais sem erro.

### Etapa P3 — Integração e teste da LLM

1. Verificar Ollama e `llama3.2:3b` localmente.
2. Usar o score **realmente fornecido pelo classificador**, sem pedir à LLM que o recalcule.
3. Criar um prompt claro para explicar o resultado com linguagem simples e factual, sem afirmar causalidade individual não comprovada.
4. Definir limite razoável de resposta, temperatura baixa, timeouts, tratamento de exceções e fallback quando Ollama não estiver acessível.
5. Testar amostras com previsões diferentes e conferir se a explicação **não contradiz** probabilidade, classe, limiar e atributos fornecidos.
6. Incluir uma demonstração documentada da LLM funcionando; se o recurso não estiver disponível no ambiente do agente, informar a pendência de teste em máquina local.

**Aceite P3:** a LLM gera uma explicação clara e coerente, sem modificar o resultado do ML, e a aplicação continua funcional se a LLM falhar.

### Etapa P4 — Qualidade, colaboração, GitHub e reprodutibilidade

1. Ampliar testes unitários e acrescentar testes de integração com **fixtures sintéticas claramente identificadas** (sem confundir com experimentos reais).
2. Cobrir carregamento, dados ausentes, classes inválidas, datas inválidas, splits temporais, métricas, inferência e erros de Ollama.
3. Executar testes e documentar resultados; caso alguma dependência não esteja disponível, declarar o limite.
4. Tornar estrutura e scripts previsíveis; não ampliar a arquitetura sem benefício real. **Não migrar para Laravel/Next.js:** a disciplina solicita Python e o Streamlit é suficiente.
5. Revisar `.gitignore`: não versionar `.venv/`, dados brutos, segredos ou modelos gigantes. Avaliar se relatórios pequenos e figuras devem ficar no GitHub para auditoria.
6. Atualizar README com comandos reproduzíveis em Linux e Windows, link da fonte de dados, diagrama do fluxo, versões testadas, exemplos de execução e interpretação de resultados.
7. Criar/atualizar um registro de experimentos e decisões metodológicas. Evitar substituir experimentos anteriores sem rastreio.
8. Preparar projeto para duas pessoas colaborarem por Git, preferencialmente com branches e PRs pequenos.

**Aceite P4:** um terceiro consegue clonar o repositório, instalar requisitos, obter os dados pela fonte documentada, treinar, executar o app e reproduzir as métricas dentro do processo descrito.

### Etapa P5 — Artigo e pitch: DEIXAR PARA O FIM

**Não começar escrevendo resultados que ainda não existem.** Depois das etapas anteriores:

- Elaborar o artigo em **LaTeX no Overleaf**, utilizando o template SBC informado pelo aluno:
  <https://pt.overleaf.com/latex/templates/sbc-conferences-template-updated-sbc-template-dot-sty-v2017/pyhttxftxjqn>
- O PDF de exemplo do template SBC foi fornecido à equipe separadamente; ele é um **modelo de formatação**, não um artigo já pronto.
- Estrutura sugerida: título/autores; **resumo e abstract**; introdução; referencial teórico; metodologia/dados; experimentos/resultados; discussão/limitações; conclusão; referências.
- Pelo template, artigos em português incluem **Resumo e Abstract**, ambos na primeira página e, segundo a orientação do exemplo, com até dez linhas cada. **Não inventar limite total de páginas**: verificar eventuais regras adicionais do professor.
- Tabelas/gráficos devem derivar de `reports/` e trazer valores reais e interpretação; referências devem ser reais e verificáveis.
- Gravar **pitch de 3–5 minutos**, apresentando problema, método, três modelos, métricas reais, interface, integração LLM e limites.
- Revisar as exigências exatas e o modo de submissão no EcAD antes da entrega final.

**Aceite P5:** artigo consistente com experimentos e resultados realmente obtidos, vídeo dentro da duração e GitHub na versão final documentada.

## 8. Problemas metodológicos e de engenharia que o agente DEVE investigar

- **Rótulo não é um atraso genérico.** `Charged Off` é uma baixa como perda; não chamar toda parcela atrasada de `Charged Off`.
- **Risco de viés de seleção:** manter apenas empréstimos já finalizados altera a população analisada. Empréstimos recentes podem ter mais casos sem desfecho; isso afeta o corte temporal e a generalização. Explicitar e avaliar.
- **Temporalidade:** `issue_d` serve à divisão, mas a base contém informações históricas até 2018; não afirmar que previsões são atuais ou representativas do mercado brasileiro.
- **Leakage:** atributos pós-concessão não podem entrar como entrada do modelo. `loan_status` jamais entra no `X`; `issue_d` atualmente é só para o split.
- **Escopo e amostragem:** uma amostra de 15% limitada a 150 mil linhas é escolha técnica inicial, não obrigação científica. Avaliar impacto e registrar tamanho real.
- **Calibração:** `predict_proba` produz estimativa do modelo, não uma taxa necessariamente calibrada. Analisar se é necessário calibrar e como fazê-lo sem contaminar teste.
- **Threshold:** não confundir probabilidade estimada com rótulo previsto pelo limiar. O limiar pode ser diferente de 50%.
- **Importância global:** permutation importance sobre validação fala do comportamento médio; não permite dizer por que uma observação específica teve aquele score. Não criar explicações causais falsas nem inventar SHAP se não for implementado.
- **LLM:** respostas são textuais e podem errar; usar prompt restritivo e verificar inconsistências. Não enviar informações sensíveis a endpoints externos sem necessidade.
- **Dataset muito grande:** evitar leitura integral desnecessária que estoure memória. Não substituir os dados reais por aleatórios apenas para conseguir concluir a tarefa.
- **Arquivos `.joblib`:** carregar apenas artefatos produzidos localmente e de origem confiável.
- **Reprodutibilidade:** mudanças de bibliotecas, amostra e período afetam números; registrar dependências e condições de teste.

## 9. Estrutura final recomendada (evolução, não exigência de reescrita)

```text
lendingrisk-ai/
├── app.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── training.py
│   ├── llm.py
│   └── ...                    # novos módulos somente se necessários
├── tests/
│   └── ...                    # testes unitários e de integração
├── data/                      # arquivos brutos locais; .gitignore
├── models/                    # artefatos de treino locais; .gitignore
├── reports/
│   ├── metricas.json
│   ├── comparacao_modelos.csv
│   ├── importancia_atributos.csv
│   ├── matriz_confusao_teste.csv
│   └── figures/              # gráficos reais, depois do treino
├── docs/
│   ├── artigo_roteiro.md
│   ├── pitch_roteiro.md
│   ├── experimentos.md        # criar se útil
│   └── ...
├── requirements.txt
├── README.md
├── .gitignore
└── INSTRUCOES_CLAUDE_CODEX_LENDINGRISK_AI.md
```

Separar **código**, **dados brutos**, **modelo treinado**, **relatórios** e **documentação**. Preservar os diretórios já existentes e evitar adicionar framework/infraestrutura desnecessários.

## 10. Publicação no GitHub (quando a dupla decidir)

1. Criar repositório **público** no GitHub com nome sugerido `lendingrisk-ai`.
2. Se criar o repositório vazio pela interface, não gerar README/.gitignore automáticos, pois esses arquivos já estão no pacote.
3. Na pasta do projeto, configurar Git e enviar os arquivos:

```bash
git init -b main
git add .
git commit -m "chore: estrutura inicial do LendingRisk AI"
git remote add origin https://github.com/SEU-USUARIO/lendingrisk-ai.git
git push -u origin main
```

Substituir `SEU-USUARIO` pelo usuário real. Se o projeto já tiver `.git` ou um remoto configurado, **não executar comandos cegamente**: inspecionar `git status` e `git remote -v` antes.

Antes de `git add .`, conferir que não há CSV, segredos, credenciais ou modelos grandes entre os arquivos rastreados. A conta deve ter autenticação Git configurada. **Não presumir que o agente tem permissão para publicar ou alterar GitHub sem autorização.**

## 11. Definição de pronto (*Definition of Done*)

O projeto só deve ser chamado de **completo** quando todas as condições pertinentes forem atendidas:

- [ ] Repositório público documentado e com `requirements.txt` funcional.
- [ ] Dataset real identificado, baixado e corretamente interpretado.
- [ ] Target `0=Fully Paid`, `1=Charged Off` devidamente aplicado e descrito.
- [ ] Leakage analisado e evitado; separação temporal registrada.
- [ ] **Três modelos treinados de verdade** no Lending Club real.
- [ ] Métricas e matriz de confusão reais, sem números inventados.
- [ ] Melhor modelo e limiar escolhidos pela validação, teste final preservado.
- [ ] Arquivos de modelo e relatórios gerados e coerentes entre si.
- [ ] Interface executável e capaz de realizar previsão com novo formulário.
- [ ] LLM local funcionando e explicando a saída sem alterar o score.
- [ ] Testes automatizados e teste manual ponta a ponta registrados.
- [ ] README e documentação permitem reprodução por terceiros.
- [ ] Artigo SBC pronto **depois** dos experimentos.
- [ ] Pitch gravado com 3–5 minutos e evidências reais.
- [ ] Versão final publicada no GitHub e entrega feita conforme EcAD.

## 12. Como o agente deve trabalhar e responder ao usuário

**Diretrizes de execução:**

1. **Primeiro auditar, depois mexer.** Ler os arquivos existentes e apresentar um plano curto, com prioridades. Não descartar o starter à toa.
2. **Trabalhar efetivamente no repositório recebido.** Implementar melhorias reais em arquivos, não responder somente com uma arquitetura teórica.
3. **Executar comandos, testes e verificações quando possível.** Registrar comandos executados, saídas importantes e eventuais falhas.
4. **Diferenciar:** implementado / testado / não testado / bloqueado (ex.: falta do CSV ou Ollama). Jamais declarar que algo está pronto sem evidência.
5. **Não inventar métricas científicas, fontes, prints ou logs.** Se houver apenas dataset sintético para teste, deixar a limitação inequívoca.
6. **Não trocar o dataset, o tema ou os três modelos** sem um motivo documentado e aprovação da dupla.
7. **Documentar alterações** e manter interface e texto em português brasileiro natural. Preserve nomes de API, arquivos, métricas e variáveis em inglês quando apropriado.
8. **Não criar dependência de API paga.** Preferir o Ollama local já planejado.
9. **Não fazer publicação externa, exclusão de arquivos importantes ou mudanças destrutivas sem autorização.**
10. **Após cada etapa**, relatar: arquivos alterados, comandos/testes executados, evidências/artefatos obtidos, pendências e próximo passo concreto.
11. **Priorizar o núcleo científico (ML + dataset real) antes de polir o visual.** Artigo e vídeo só devem usar resultados verificados.

### Primeira instrução prática ao agente (comece por aqui)

> Analise os arquivos deste projeto, compare o código existente com este briefing e com o `README.md`, identifique os problemas e pendências mais importantes e execute a Etapa P0. Se o dataset real estiver disponível em `data/`, prossiga para a Etapa P1, treinando e avaliando os três modelos. Se não estiver, conclua o que for possível sem inventar resultados e informe exatamente como o aluno deve baixar/colocar o arquivo. Preserve a arquitetura Python + Streamlit + Ollama, corrija problemas reais, amplie os testes e registre o que foi executado. Não comece o artigo nem o pitch antes de termos experimentos reais.

---

## 13. Links de referência

- **Projeto acadêmico:** documento “Projeto Final — CARCDSI”, fornecido pelo professor.
- **Dataset Lending Club:** <https://www.kaggle.com/datasets/wordsforthewise/lending-club>
- **Template SBC no Overleaf:** <https://pt.overleaf.com/latex/templates/sbc-conferences-template-updated-sbc-template-dot-sty-v2017/pyhttxftxjqn>
- **Python:** <https://www.python.org/>
- **Scikit-learn:** <https://scikit-learn.org/stable/>
- **XGBoost:** <https://xgboost.readthedocs.io/>
- **Streamlit:** <https://docs.streamlit.io/>
- **Ollama:** <https://ollama.com/>
- **Git:** <https://git-scm.com/>

**Fim do briefing.** A referência prioritária para a execução é o **código realmente presente no ZIP**, combinado com as especificações acadêmicas e metodológicas descritas acima. Corrija discrepâncias; não presuma que o projeto já está finalizado.
