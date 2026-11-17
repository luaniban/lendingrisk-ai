# Pendências do LendingRisk AI

Lista viva: cada item sai daqui quando for concluído (o histórico fica no `git log`).
Última revisão: 17/11/2026.

## Aguardando retorno do Luan

- [ ] **Validar o novo layout da interface.** A interface agora é um "dossiê de crédito": a
  proposta fica ao lado do parecer, há uma régua de risco e o veredito aparece como carimbo.
  Testar em `http://127.0.0.1:8501` e apontar o que mudar.

## Entregas acadêmicas (adiadas a pedido)

- [ ] Artigo científico (roteiro em `docs/artigo_roteiro.md`).
- [ ] Pitch (roteiro em `docs/pitch_roteiro.md`).

## Publicação e reprodutibilidade

- [ ] **Decidir a visibilidade do repositório.** Ele foi publicado como **privado** em
  https://github.com/luaniban/lendingrisk-ai. Antes de torná-lo público, considere que o README
  e o `INSTRUCOES_…md` têm os prontuários da dupla (inclusive o do Hyan), que os commits levam o
  e-mail do Luan e que os arquivos de instrução para agentes de IA (`CLAUDE.md`, `AGENTS.md`)
  estão versionados.
- [ ] **Adicionar o Hyan como colaborador.** Falta saber o usuário dele no GitHub.
- [ ] **Testar a instalação em outra máquina**, de preferência no Windows do Hyan. As
  instruções do README para Windows não foram testadas.

## Melhorias opcionais

- [ ] **Experimento "com e sem `grade`".** A dupla decidiu deixar `grade`/`int_rate` fora do
  modelo (seção 5.3 de `docs/experimentos.md`). Rodar a comparação só se for útil para o
  artigo, como medida de quanto a nota do Lending Club acrescenta.
- [ ] **Bootstrap pareado entre a v1 e a v2.** A v2 melhorou um pouco a AP no teste
  (+0,002), sem teste de significância (seção 5.2). Um bootstrap diria se a diferença é
  robusta.
- [ ] Gerar de novo os gráficos com o fundo da cor do tema (hoje são um pouco mais claros que a
  página). Exige rodar o treino (~3 min) ou criar um comando que só refaça as figuras.
