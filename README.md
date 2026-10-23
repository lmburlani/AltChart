# AltChart

**README data de:** 23/10/2026

## Escopo

Criar uma ferramenta simples para acompanhar a parada Billboard Alternative Airplay e estimar quais musicas do Top 40 permanecem no chart na semana seguinte

O projeto serve como estudo de ML com um fluxo reproduzivel e futura publicacao web, nao como previsao de sucesso musical ou ferramenta de investimento

## Como foi feito

- Historico semanal do chart desde outubro de 2023
- Variaveis baseadas na posicao atual, posicao anterior, pico historico, semanas no chart e movimento recente
- Quatro modelos comparados: regressao logistica, Random Forest, KNN e Gradient Boosting
- Ajuste de parametros com validacao temporal, usando semanas passadas para avaliar semanas posteriores
- Ultimas 12 semanas reservadas para teste
- Comparacao com um baseline que prevê que todas as musicas permanecem
- Preparacao para salvar previsoes, metricas e modelo, com atualizacao semanal planejada para quinta-feira

## Resultado de 23/10/2026

O Gradient Boosting foi o melhor modelo na validacao temporal, com F1 medio de 0,970. No teste final, teve acuracia de 0,960, precisao de 0,962, recall de 0,998, F1 de 0,980, ROC-AUC de 0,918 e PR-AUC de 0,994

O baseline de permanencia total teve acuracia de 0,950 e F1 de 0,974. A diferenca de F1 foi pequena: 0,006, ou 0,6 ponto percentual

Nas 12 semanas avaliadas, a permanencia real ficou entre 90% e 97,5% das musicas. A acuracia semanal do modelo variou de 92,5% a 100%, e o F1 de 96,1% a 100%

## Interpretacao

Os numeros parecem fortes, mas a tarefa atual e facil para quem sempre responde que a musica vai permanecer, porque a maioria realmente permanece. O baseline ja alcanca F1 de 0,974

O modelo supera o baseline por uma margem pequena no F1 e mostra capacidade de separar os resultados no ROC-AUC, mas ainda nao provamos que ele identifica bem as musicas que vao sair do chart

O F1 atual mede principalmente o acerto na classe mais comum — musicas que permanecem (F1 resume o equilibrio entre precisao e recall para essa classe). A acuracia mede a proporcao total de previsoes corretas. O ROC-AUC mede o quanto o modelo consegue ordenar casos que permanecem acima dos que saem. Como ha poucas saidas, as metricas podem parecer excelentes sem que o modelo seja realmente bom em detectar essas saidas

## Conclusao

O pipeline funciona como primeira versao e os resultados sao promissores, mas ainda nao justificam afirmar que o modelo preve mudancas do chart com confianca. O ganho sobre o baseline e pequeno e a classe de interesse — musicas que saem — precisa de avaliacao propria

O proximo passo e manter o escopo enxuto, acompanhar o resultado semanal e, antes de tratar o modelo como confiavel, medir precision, recall e F1 especificamente para as musicas que saem do Top 40. O deploy e a atualizacao automatica de quinta-feira continuam como a proxima etapa do projeto
