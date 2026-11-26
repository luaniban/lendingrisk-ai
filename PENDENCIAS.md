# Pendências do LendingRisk AI

Lista viva: cada item sai daqui quando for concluído (o histórico fica no `git log`).
Última revisão: 26/11/2026.

## Aguardando retorno do Luan

- [ ] **Validar o novo layout da interface.** A interface agora é um "dossiê de crédito": a
  proposta fica ao lado do parecer, há uma régua de risco e o veredito aparece como carimbo.
  Testar em `http://127.0.0.1:8501` e apontar o que mudar.

## Entregas acadêmicas

- [ ] **Artigo: compilar no Overleaf.** O texto está pronto em `artigo/` (`main.tex`,
  `sbc-template.bib` e `figuras/`). Falta colar no template SBC do Overleaf, trocar
  `EMAIL-HYAN` e `EMAIL-LUAN` pelos e-mails reais, compilar, conferir o PDF e o limite
  de páginas do professor, e entregar.
- [ ] **Pitch: gravar o vídeo.** O roteiro com falas está pronto em
  `docs/pitch_roteiro.md` (cerca de 4 min 30 s). Falta gravar, editar (cortar a espera
  da LLM) e entregar.

## Publicação e reprodutibilidade

- [ ] **Hyan aceitar o convite de colaborador.** O convite (permissão de escrita) foi enviado
  ao usuário `HyanFerreira` em 17/11/2026. Ele aceita pelo e-mail do GitHub ou em
  https://github.com/luaniban/lendingrisk-ai/invitations.

## Ambiente Windows do Hyan

Ollama 0.40.16 instalado e testado em 19/11/2026 com `llama3.2:3b` (2 GB).
O teste da interface com o serviço real gerou uma explicação em 31,8 s, em uma
tentativa, sem avisos do verificador. A probabilidade de 10,7% do perfil hipotético
foi preservada. Evidências locais em `.venv/teste-ollama-real.log` e
`.venv/ollama-verificacao.json`.

Instalação Python validada em 19/11/2026: Python 3.12.16 em `.venv`, 78 testes
aprovados, auditoria do CSV real, treino dos três modelos e simulação na interface.
Os relatórios desta execução ficam em `.venv/relatorios-windows`, selecionados por
`iniciar.ps1`. O XGBoost local teve AP de teste 0,24161 (oficial Linux: 0,24253), com
o mesmo limiar de 9%. Os relatórios oficiais foram preservados. Logs locais em
`.venv/testes-windows.log`, `.venv/auditoria-windows.log`, `.venv/treino-windows.log`
e `.venv/verificacao-app-real.log`.

## Melhorias opcionais

- [ ] **Experimento "com e sem `grade`".** A dupla decidiu deixar `grade`/`int_rate` fora do
  modelo (seção 5.3 de `docs/experimentos.md`). Rodar a comparação só se for útil para o
  artigo, como medida de quanto a nota do Lending Club acrescenta.
- [ ] **Bootstrap pareado entre a v1 e a v2.** A v2 melhorou um pouco a AP no teste
  (+0,002), sem teste de significância (seção 5.2). Um bootstrap diria se a diferença é
  robusta.
