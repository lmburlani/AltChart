# AltChart

Projeto experimental de ML criado como estudo de caso para praticar deploy de uma aplicacao simples e explorar um fluxo basico de MLOps

O objetivo e acompanhar o Billboard Alternative Airplay e prever se cada musica vai subir, descer, se manter ou sair do Top 40 na semana seguinte

## Aplicacao

[**Acessar o dashboard AltChart**](https://altchart-mu9znfcwrahj2yguxscpo2.streamlit.app/)

## Arquivos

- `AltChart.ipynb` — experimento inicial
- `refresh.py` — coleta, treino, avaliacao e previsao
- `app.py` — dashboard Streamlit
- `requirements.txt` — dependencias do dashboard
- `requirements-mlops.txt` — dependencias do pipeline
- `.github/workflows/weekly_refresh.yml` — atualizacao semanal

## Como funciona

O pipeline coleta o historico semanal, cria variaveis de posicao e movimento recente e compara quatro modelos

- Regressao logistica
- Random Forest
- KNN
- Gradient Boosting

A escolha considera o F1 macro em validacao temporal, com as ultimas 12 semanas separadas para teste

As previsoes ficam registradas para comparar a direcao prevista com o resultado real quando o proximo chart e publicado
