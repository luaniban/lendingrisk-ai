# Redesign LendingRisk AI — plano de implementação

**Objetivo:** aplicar as referências visuais às três páginas Streamlit, preservando toda a inferência e os artefatos científicos.

**Arquitetura:** `app.py` continua orquestrando widgets, inferência e Ollama. `src/ui.py` concentra componentes HTML de apresentação, com escape de conteúdo; `assets/styles.css` concentra estilos. O tema nativo permanece em `.streamlit/config.toml`.

**Escopo autorizado:** implementar diretamente neste checkout, conforme pedido. Não retreinar, não reavaliar o conjunto de teste, não alterar os módulos científicos, não publicar. Preservar a alteração preexistente em `PENDENCIAS.md`.

## Etapas

- [x] Auditar aplicação, artefatos e testes; registrar uma previsão real anterior à alteração.
- [x] Criar regressões para navegação, edição após previsão, persistência dos campos, parecer fora das colunas e integridade dos gráficos.
- [x] Aplicar identidade visual, sidebar e cabeçalho com componentes nativos de navegação e entrada.
- [x] Organizar o simulador em três grupos, resumo lateral e parecer detalhado em largura completa; preservar explicação e erros de Ollama.
- [x] Reorganizar desempenho em indicadores, tabelas, nove gráficos originais e tabelas complementares, incluindo as análises pós-teste existentes.
- [x] Organizar fonte, elegibilidade, divisão temporal, auditoria real, atributos, limitações, fluxo e ambiente.
- [x] Executar testes, comparar inferência real, verificar renderização desktop/mobile e documentar evidências e limites.

## Verificações essenciais

Modelo ausente deve continuar exibindo instruções. Ollama indisponível não impede inferência. Edição invalida parecer e texto da LLM. Navegação preserva proposta e análise ainda válida. Nenhum gráfico ou coluna analítica pode desaparecer. Tabelas conservam visualização e exportação nativas.

## Auditoria inicial

- Entrada: `app.py`; tema antigo em `.streamlit/config.toml`; inferência e validação em `src/inference.py`; Ollama em `src/llm.py`.
- Modelo oficial local, dataset aceito real e relatórios já presentes. Sem necessidade de treino neste trabalho de interface.
- Nove figuras de desempenho e `auditoria_coortes.png` presentes. Há análises pós-teste adicionais que também serão preservadas.
- A referência visual contém métricas ilustrativas divergentes: usar exclusivamente `reports/metricas.json`.
- Proposta real de referência antes da alteração: 11225 USD; renda 60320 USD; DTI 16,7; FICO 690; consultas 0; rotativo 56,5; 36 meses; consolidação; hipoteca; 10+ anos. Probabilidade bruta: `0.10669147968292236`.

## Evidências finais

- Antes do redesign: `python -m unittest discover -s tests -v` — **78 testes passaram**.
- Execução final: `python -m unittest discover -s tests -v` — **82 testes passaram** em 26,657 s, incluindo inferência sem Ollama, tratamento de erros, estado dos campos e posição do parecer. `git diff --check` também passou.
- Edge headless (Playwright, ferramenta de desenvolvimento): 1440, 900, 390 e 320 px. Navegação, previsão, posição/largura do parecer, preservação da análise ao navegar, invalidação após edição, nove figuras de desempenho, uma de auditoria, ausência de overflow horizontal da página e de erros JavaScript/console: todos passaram.
- Modelo real: probabilidade bruta antes e depois `0.10669147968292236`, exibida como **10,7%**, classe Charged Off, limiar 9,0%.
- Ollama real (`llama3.2:3b`): explicação em **31,5 s**, primeira tentativa, sem avisos da verificação heurística, probabilidade bruta idêntica antes/depois. Indisponibilidade e erros também cobertos por mocks nos testes automatizados.
- Ambiente da validação visual: Windows, Python 3.12, Streamlit 1.67.0, Microsoft Edge 157.0.4341.60. O ambiente de treinamento exibido no aplicativo continua sendo o Linux registrado no artefato original, sem substituição pelos dados desta máquina.
- Capturas completas e do topo das páginas: `%TEMP%/lendingrisk-redesign/`; evidências estruturadas em `evidence.json` e `llm-real.json`. Capturas não entram nos relatórios científicos.

## Repetir a verificação visual (opcional)

Com os artefatos oficiais presentes e o Microsoft Edge instalado:

```powershell
.\.venv\Scripts\python.exe -m pip install playwright
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8503
# Em outro terminal, na raiz:
.\.venv\Scripts\python.exe scripts/check_redesign_browser.py
```

Playwright foi instalado somente no ambiente de desenvolvimento; não foi adicionado às dependências da aplicação. O script confere a proposta de referência usando o modelo real local. Não treina nem ajusta modelos e não gera métricas de experimento.

## Decisões

- Entradas nativas fora de `st.form`: cada alteração confirmada pelo widget invalida imediatamente o parecer, em vez de manter uma previsão antiga visível até o envio do formulário. O cálculo continua exclusivamente no botão de análise.
- Estado durável da proposta separado das chaves dos widgets: evita perder valores quando o Streamlit desmonta os campos ao trocar de página.
- CSS concentrado em arquivo local, com classes próprias e chaves públicas dos containers; seletores de widgets nativos limitados aos ajustes de apresentação necessários.
- Nenhuma alteração em treino, pipeline, inferência, prompts da LLM ou artefatos de experimento. As divergências entre números das imagens e relatórios foram resolvidas usando os relatórios.
- Revisão independente solicitada ao final, conforme a habilidade de revisão de código.
- Revisão independente concluída: nenhum problema de inferência ou perda de conteúdo; identificação de contraste insuficiente em textos secundários. Corrigidos os tokens de texto, rodapé, descrições e blocos de informação. Contrastes medidos dos tokens principais: mínimo **4,62:1**, acima de 4,5:1, nos fundos usados.

## Limites da verificação

- Navegador efetivamente testado: Microsoft Edge em Windows, com larguras emuladas; não houve teste em aparelhos físicos nem em Safari/Firefox.
- A checagem da LLM é heurística. O teste real sem avisos não garante correção de toda geração futura.
- Imagens analíticas originais foram preservadas, inclusive sua paleta; não foram regeneradas para combinar com o novo tema.
- A aplicação foi validada no Streamlit 1.67.0 instalado. Não foi testada uma matriz de todas as versões permitidas em `requirements.txt`.
