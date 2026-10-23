# AltChart

Projeto de ML para acompanhar o Billboard Alternative Airplay e prever se cada musica vai subir, descer, se manter ou sair do Top 40 na semana seguinte

## Arquivos

- `AltChart.ipynb` — experimento inicial com previsao de permanencia no chart
- `refresh.py` — pipeline operacional de coleta, treino, teste e previsao de direcao
- `app.py` — dashboard Streamlit
- `requirements.txt` — dependencias do dashboard
- `requirements-mlops.txt` — dependencias do pipeline
- `.github/workflows/weekly_refresh.yml` — atualizacao automatica toda quinta-feira

## Como funciona

O pipeline baixa o historico semanal, cria variaveis com posicao e movimento recente, compara regressao logistica, Random Forest, KNN e Gradient Boosting e seleciona o modelo por F1 macro em validacao temporal

As ultimas 12 semanas com resultado conhecido ficam separadas para teste. As previsoes historicas sao salvas para comparar a direcao prevista com o movimento real

A previsao da semana mais recente fica como pendente ate a Billboard publicar o chart seguinte. Depois, o pipeline compara o resultado real com a previsao original, sem recalcular a previsao antiga

## Rodar localmente

Instale as dependencias do pipeline e execute a atualizacao

```bash
pip install -r requirements-mlops.txt
python refresh.py
```

Para abrir o dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Adicione os arquivos deste projeto ao repositorio `lmburlani/AltChart`
2. No Streamlit Community Cloud, crie um app a partir do repositorio, branch `main` e arquivo `app.py`
3. No GitHub, abra **Actions** e execute **AltChart weekly refresh** manualmente na primeira vez
4. Confirme que a execucao terminou com sucesso e que os arquivos de `data/`, `reports/` e `models/` foram salvos

O GitHub Actions executa o pipeline toda quinta-feira as 09:17 no horario de Sao Paulo e publica os novos artefatos no repositorio

## Leitura dos resultados

O modelo e experimental. F1 macro da mais peso as classes menos frequentes do que a acuracia simples, mas resultados historicos nao garantem acerto futuro. A previsao deve ser comparada com o baseline e com os resultados semanais reais
