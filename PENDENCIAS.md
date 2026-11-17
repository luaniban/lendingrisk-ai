# Pendências do LendingRisk AI

Lista viva: cada item sai daqui quando for concluído (o histórico fica no `git log`).
Última revisão: 17/11/2026.

## Aguardando retorno do Luan

- [ ] **Validar o novo layout da interface.** A interface agora é um "dossiê de crédito": a
  proposta fica ao lado do parecer, há uma régua de risco e o veredito aparece como carimbo.
  Testar em `http://127.0.0.1:8501` e apontar o que mudar.

## Decisões de metodologia (precisam da dupla)

O teste reservado já foi usado. Qualquer mudança abaixo deve ser registrada em
`docs/experimentos.md` como **análise posterior**, sem substituir o resultado principal.

- [ ] **Limiar muito agressivo.** O critério F2 levou ao limiar de 9%, que sinaliza 72,7% dos
  empréstimos (acurácia de 38,6%, recall de 88,0%). Decidir entre manter (e justificar no
  artigo) ou testar outro critério, por exemplo F1 ou uma taxa de aprovação fixa.
- [ ] **Tempo de emprego "não informado".** O modelo subestima o risco desse grupo (21,7% de
  Charged Off real contra 16,3% previsto), porque a imputação usa a moda. Avaliar tratar a
  ausência como categoria própria.
- [ ] **Incluir `grade` / `int_rate`?** Melhorariam o desempenho, mas são notas do próprio
  modelo de risco do Lending Club. Isso muda os 10 atributos combinados com a dupla.

## Entregas acadêmicas (adiadas a pedido)

- [ ] Artigo científico (roteiro em `docs/artigo_roteiro.md`).
- [ ] Pitch (roteiro em `docs/pitch_roteiro.md`).

## Publicação e reprodutibilidade

- [ ] **Publicar no GitHub:** só com autorização. Hoje o repositório é só local, sem remoto.
- [ ] **Testar a instalação em outra máquina**, de preferência no Windows do Hyan. As
  instruções do README para Windows não foram testadas.

## Melhorias opcionais

- [ ] Gerar de novo os gráficos com o fundo da cor do tema (hoje são um pouco mais claros que a
  página). Exige rodar o treino (~3 min) ou criar um comando que só refaça as figuras.
