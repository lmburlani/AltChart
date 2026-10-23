# AltChart

Projeto experimental de ML para estudo de caso de deploy e MLOps, com previsoes semanais do Billboard Alternative Airplay

[**Abrir dashboard**](https://altchart-mu9znfcwrahj2yguxscpo2.streamlit.app/)

## O que faz

Estima se cada musica do Top 40 vai subir, descer, manter a posicao ou sair do chart na semana seguinte

O pipeline cria variaveis com o historico recente de posicao, compara regressao logistica, Random Forest, KNN e Gradient Boosting e escolhe a configuracao pelo F1 macro em validacao temporal

O backtest avalia as ultimas 12 semanas e compara os modelos com baselines simples. As previsoes semanais ficam registradas para conferir o resultado real quando o proximo chart for publicado

## Arquivos

- `AltChart.ipynb` — experimento inicial
- `refresh.py` — coleta, engenharia de variaveis, ajuste e previsao
- `app.py` — dashboard Streamlit
- `requirements.txt` — dependencias do dashboard
- `requirements-mlops.txt` — dependencias do pipeline
- `.github/workflows/weekly_refresh.yml` — atualizacao automatizada
- `tests/test_refresh.py` — testes das variaveis e dos baselines

## Executar localmente

```bash
pip install -r requirements-mlops.txt
python refresh.py
```

Para abrir o dashboard:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para validar o codigo usando um historico ja baixado:

```bash
pip install -r requirements-mlops.txt pytest
pytest -q
python refresh.py --skip-fetch
```

## Atualizacao

O GitHub Actions executa o pipeline toda quinta-feira as 09:17 no horario de Sao Paulo e salva os dados, metricas e previsoes no repositorio

## Limitacoes

Este e um estudo de caso, nao um produto de previsao musical. A avaliacao usa semanas passadas para prever semanas futuras, mas os resultados historicos nao garantem acertos futuros
