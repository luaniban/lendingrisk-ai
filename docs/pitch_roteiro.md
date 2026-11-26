# Roteiro do pitch — LendingRisk AI (cerca de 4 min 30 s)

Vídeo de 3 a 5 minutos: problema, solução, técnicas e resultados. As falas somam cerca
de 610 palavras, o que dá uns 4 min 30 s em ritmo calmo. Todos os números vêm de
`reports/metricas.json` e `docs/experimentos.md` (os mesmos do artigo).

**Quem fala:** **Luan** abre, apresenta os dados, a metodologia e os resultados. **Hyan**
apresenta a solução, a demonstração e fecha. Se quiserem outra divisão, basta trocar
os nomes.

**Antes de gravar:**
- Deixe o app aberto (`.\iniciar.ps1` ou `python -m streamlit run app.py`) e o Ollama
  ligado. Verifique se a barra lateral mostra o Ollama conectado.
- Gere uma explicação antes de gravar, para o modelo já estar carregado. Mesmo assim,
  cada explicação leva de 15 a 30 s: **corte a espera na edição**.
- Na demonstração, leia o número que aparecer na tela. Um modelo treinado em outra
  máquina pode dar um valor um pouco diferente do roteiro (por exemplo, 10,7% em vez de
  10,9%).

---

## 1. Problema — 0:00 a 0:35 · Luan

**Tela:** nossos rostos ou a página inicial do app.

> Quando alguém pede um empréstimo, quem empresta precisa responder uma pergunta antes
> de assinar: qual a chance de esse dinheiro não voltar? Em plataformas como o Lending
> Club, quem assume esse risco é o investidor. E mesmo quando existe um modelo que
> calcula essa chance, o resultado sai como um número solto, difícil de entender para
> quem não é da área. Eu sou o Luan, este é o Hyan, e nosso projeto é o LendingRisk AI.

## 2. Solução — 0:35 a 1:05 · Hyan

**Tela:** diagrama do fluxo do README ou a página "Dados e metodologia".

> O LendingRisk AI faz duas coisas. Primeiro, um modelo de Machine Learning estima a
> probabilidade de um empréstimo terminar como perda, o que a base chama de Charged Off.
> Depois, uma LLM rodando localmente, no próprio computador, explica esse resultado em
> português. Um cuidado importante: a LLM não calcula nada. Ela só recebe o número que o
> modelo já calculou e coloca em palavras. E é um protótipo acadêmico: não serve para
> aprovar ou recusar crédito de verdade.

## 3. Dados e metodologia — 1:05 a 2:15 · Luan

**Tela:** página "Dados e metodologia"; mostre o gráfico das coortes quando falar do
viés.

> Usamos a base pública do Lending Club, com mais de 2 milhões de empréstimos de 2007 a
> 2018. O alvo é simples: o empréstimo foi quitado ou foi baixado como perda.
> Usamos só dez informações que existem no momento do pedido, como valor, renda, score
> FICO e finalidade. Nada que só se sabe depois, como quanto a pessoa já pagou; isso
> seria vazamento de dados, e o modelo pareceria melhor do que é.
>
> Antes de treinar, auditamos a base e encontramos uma armadilha. Os empréstimos
> recentes ainda estavam em andamento, e os poucos que já tinham terminado eram, em boa
> parte, os que deram errado cedo. Neles, a taxa de perda era de 25%, contra 15% nos
> empréstimos já encerrados. Por isso ficamos só com empréstimos cujo prazo já tinha
> acabado: 673 mil no total.
>
> Separamos os dados por data: treino com o passado, validação e teste com os meses
> seguintes, como seria no mundo real. Treinamos três modelos: Regressão Logística,
> Random Forest e XGBoost. E o teste ficou guardado: usamos uma vez só, no final.

## 4. Resultados — 2:15 a 3:15 · Luan

**Tela:** página "Desempenho dos modelos": tabela de comparação, curva
precisão × recall e gráfico do limiar.

> Na validação, o XGBoost foi o melhor, por pouco. A métrica principal foi a Average
> Precision, porque só 15% dos empréstimos terminam em perda, e a acurácia engana nesse
> caso.
>
> No teste, o XGBoost chegou a uma Average Precision de 0,24, contra 0,15 de um palpite
> aleatório, e ROC-AUC de 0,66. É um desempenho modesto, mas estável entre validação e
> teste, e a probabilidade é bem calibrada: quando o modelo diz 10%, acontece perto de
> 10%.
>
> Escolhemos um limiar que prioriza não deixar passar quem vai dar prejuízo: o modelo
> encontra 88% das perdas. O custo é alto: ele marca 73% dos empréstimos como risco.
> Então o sistema serve para triagem, para indicar o que um analista deve revisar, e não
> para decidir sozinho.

## 5. Demonstração — 3:15 a 4:10 · Hyan

**Tela:** página "Simular empréstimo". Preencha com o perfil B da demonstração:
US$ 12.000, renda anual US$ 65.000, DTI 17,6%, FICO 690, 0 consultas em 6 meses,
rotativo 52%, 36 meses, consolidação de dívidas, aluguel, 3 anos de emprego.
Clique em **Estimar risco** e depois em **Explicar com LLM local (Ollama)**.

> Vamos ver na prática. Aqui eu preencho os dados de um pedido hipotético: doze mil
> dólares, renda de sessenta e cinco mil por ano, score 690. Clico em Estimar risco.
>
> O modelo estimou 10,9% de chance de perda. Como passa do limiar de 9%, o pedido é
> sinalizado para revisão.
>
> Agora eu peço a explicação para a LLM, que roda aqui no computador, sem mandar dados
> para a internet. Ela repete exatamente os números do modelo e deixa claro que os
> atributos citados são importantes em geral, não a causa deste caso. E se ela errar um
> número ou trocar a classe, o sistema detecta e avisa na tela.

## 6. Conclusão — 4:10 a 4:35 · Hyan

**Tela:** nossos rostos ou o repositório no GitHub.

> Resumindo: juntamos tratamento de dados, três modelos de Machine Learning e uma LLM
> local num sistema que estima o risco e explica o resultado. Os próximos passos são
> ajustar os modelos, testar com dados mais recentes e explicar cada caso
> individualmente. O código e o artigo estão no GitHub. Obrigado!

---

## Números citados (para conferir)

| Fala | Valor | Fonte |
|---|---|---|
| Empréstimos na base | 2.260.701 linhas, 2007–2018 | `docs/experimentos.md` §2.1 |
| Taxa de perda: safras recentes × encerradas | 25,12% × 14,82% | §2.3 |
| Empréstimos usados | 673.553 | §3.1 |
| AP no teste × aleatório | 0,2425 × 0,1484 | §3.3 |
| ROC-AUC no teste | 0,6623 | §3.3 |
| Recall / taxa sinalizada | 88,0% / 72,7% | §3.3 |
| Calibração (ECE) | 1,5 p.p. | §3.3 |
| Perfil B: probabilidade e limiar | 10,9% e 9,0% | `reports/llm_demonstracao.md` |
