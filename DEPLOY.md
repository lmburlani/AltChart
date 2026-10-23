# Deploy do AltChart

## 1. Envie os arquivos ao GitHub

Extraia o pacote na pasta local do repositorio. Mantenha o `AltChart.ipynb` existente. Adicione ou substitua os arquivos incluidos no pacote e envie o commit para a branch `main`

## 2. Rode o pipeline uma vez

No GitHub, abra `Actions` > `AltChart weekly refresh` > `Run workflow` > `Run workflow`

A primeira execucao busca o historico, treina e compara os modelos e cria os arquivos em `data/`, `reports/` e `models/`

## 3. Publique o dashboard

Em https://share.streamlit.io/

- Clique em `Create app`
- Repository: `lmburlani/AltChart`
- Branch: `main`
- Main file path: `app.py`
- Clique em `Deploy`

O dashboard usa `requirements.txt`. O workflow usa `requirements-mlops.txt`

## 4. Atualizacao semanal

O workflow e agendado para quinta-feira as 09:17 no horario de Sao Paulo. Tambem pode ser iniciado manualmente pela aba `Actions`
