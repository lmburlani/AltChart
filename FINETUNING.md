# AltChart — rodada de fine-tuning

Data: 23/10/2026

## O que mudou

- Historico de posicao nas ultimas 6 semanas
- Momentum recente, aceleracao, media e variacao de posicao
- Indicadores de proximidade do fim do chart, top 10 e tempo no chart
- Busca ampliada de hiperparametros para regressao logistica, Random Forest, KNN e Gradient Boosting
- KNN usa um subconjunto menor de variaveis para evitar distancia em excesso de dimensoes
- Baseline da classe mais comum e baseline que repete o movimento recente
- Metricas por classe, matriz de confusao, analise de erros e importancia das variaveis
- Testes automatizados para features, rotulos e baseline

## Interpretacao

O fine-tuning e escolhido por F1 macro em validacao temporal. As ultimas 12 semanas permanecem reservadas para o teste final

A meta nao e simplesmente aumentar a acuracia. O modelo precisa reconhecer as quatro classes — subida, queda, estabilidade e saida — sem favorecer uma classe quase sempre

O resultado do backtest real so sera conhecido depois de executar `refresh.py` com o historico atual da Billboard. Os resultados do conjunto de teste local foram usados apenas para validar que o pipeline completa a execucao, nao como estimativa do desempenho real

## Saidas novas

- `reports/class_metrics.csv`
- `reports/confusion_matrix.csv`
- `reports/error_analysis.csv`
- `reports/feature_importance.csv`

A tabela de modelos e o backtest semanal tambem incluem baselines para comparacao
