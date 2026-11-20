# Refinamento da estrutura global e sidebar

Escopo: remover header e menu nativo; padronizar navegação; usar hamburger funcional; compactar status real do Ollama e créditos. Conteúdo das páginas e lógica científica preservados.

## Plano

- [x] Auditar controles nativos e configuração da versão instalada do Streamlit.
- [x] Acrescentar regressões de navegação e atualização real do status com mocks.
- [x] Implementar apresentação global, botões de navegação uniformes e controles acessíveis.
- [x] Validar larguras/alinhamento, hover/foco, hamburger expandido/recolhido, status online/offline e mobile.
- [x] Executar suíte e comparar o bloco de conteúdo das páginas com a versão anterior.

## Decisões

- `client.toolbarMode = "minimal"` é a opção oficial que remove o menu quando não existem opções externas. CSS também oculta menu/ações e zera a altura do toolbar, preservando o botão nativo de expandir a sidebar que o Streamlit renderiza dentro dele.
- O hamburger reutiliza os controles nativos do Streamlit, com os mesmos eventos, rótulos acessíveis e atributos de expansão. Ícone e posição são alterados por seletores `data-testid`, sem classes geradas nem JavaScript injetado.
- A navegação usa botões nativos com uma única função de renderização e uma única regra de layout. Ícones vetoriais locais seguem o traço Lucide.
- O refresh invalida o mesmo cache e executa `check_ollama`; não modifica a integração ou os estados da previsão.
- A página ativa é anunciada por texto acessível com `role="status"` e `aria-live="polite"`, além do destaque visual. O anúncio foi acrescentado após a revisão independente identificar a perda dessa informação na troca de radios por botões.
- A largura máxima da sidebar é aplicada somente quando expandida; recolhida, sua largura máxima é zero, permitindo que o conteúdo ocupe a área liberada. O hamburger permanece no canto superior esquerdo.

## Validação

- Antes da rodada: **82 testes passaram** (27,336 s).
- Depois: `python -m unittest discover -s tests -v` — **84 testes passaram** (27,386 s). Inclui refresh online → offline com requisição mockada, preservação do parecer, navegação e identificação acessível da página atual.
- `python scripts/check_sidebar_browser.py` — Edge headless, 1440/900/390/320 px, com toque emulado nas duas menores larguras: todos passaram. Medições conferem mesma largura, altura, padding, raio, posição horizontal dos textos e dimensão dos ícones nos três destinos, inclusive após trocar a seleção.
- Hamburger verificado com teclado (Enter), visível sem hover, nos dois estados; conteúdo recupera a largura disponível ao recolher. Header com altura zero e botão do menu nativo ausente do DOM.
- Refresh real do Ollama verificado; opacidade zero fora de hover/foco, visível em hover, foco e dispositivo de toque, tooltip e rótulo “Verificar conexão”. O componente conserva suas dimensões ao revelar o ícone. Indisponibilidade também exercitada por mocks no AppTest; não foi desligado o serviço local.
- Árvore de acessibilidade verificada para anúncio da página atual em todos os destinos. Nenhum erro JavaScript/console na execução final.
- `python scripts/check_redesign_browser.py` — navegação, previsão real de referência (10,7%), parecer, nove gráficos de desempenho e auditoria continuaram passando nas quatro larguras.
- Comparação textual integral do bloco `if page == PAGES[0]:` até o fim de `app.py`, antes/depois desta rodada: **idêntico**, preservando as três páginas. Treinamento, inferência e integração LLM não foram alterados.
- `git diff --check` passou. Capturas e evidência JSON: `%TEMP%/lendingrisk-sidebar/`.

## Arquivos desta rodada

`app.py`, `src/ui.py`, `assets/styles.css`, `.streamlit/config.toml`, `tests/test_app.py`, `scripts/check_redesign_browser.py`, `scripts/check_sidebar_browser.py` e este registro. As alterações preexistentes em outros arquivos foram preservadas.

## Repetir

Reinicie o Streamlit para carregar a configuração e o módulo visual atualizados. Para verificar em uma porta de desenvolvimento:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8503
# Em outro terminal, com Playwright já instalado no ambiente de desenvolvimento:
.\.venv\Scripts\python.exe scripts/check_sidebar_browser.py
```

A validação usa Edge no Windows e emulação de toque/largura; não substitui testes em aparelhos físicos ou outros navegadores. Nenhuma dependência nova foi adicionada à aplicação.
