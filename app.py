from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="AltChart", page_icon="🎸", layout="wide")


def read_csv(name: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / name
    if not path.exists():
        return None
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    return data if not data.empty else None


def read_json(name: str) -> dict:
    path = MODELS_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def fmt_date(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "—" if pd.isna(parsed) else parsed.strftime("%d/%m/%Y")


def result_label(actual, correct) -> str:
    if pd.isna(actual) or str(actual).strip() == "":
        return "Pendente"
    value = str(correct).strip().lower()
    return "Acertou" if value in {"true", "1", "1.0"} else "Errou"


def display_trend(value) -> str:
    labels = {
        "Subiu": "↑ Subiu",
        "Desceu": "↓ Desceu",
        "Manteve": "→ Manteve",
        "Saiu": "↓ Saiu do chart",
        "Estreia": "★ Estreia",
    }
    return labels.get(str(value), str(value) if pd.notna(value) else "—")


st.title("AltChart")
st.caption("Billboard Alternative Airplay · previsao semanal de movimento no Top 40")

forecast = read_csv("latest_forecast.csv")
comparison = read_csv("model_comparison.csv")
weekly = read_csv("weekly_backtest.csv")
backtest = read_csv("backtest_predictions.csv")
history = read_csv("forecast_history.csv")
metadata = read_json("model_metadata.json")

if forecast is None:
    st.info(
        "Ainda nao ha previsoes publicadas. No GitHub, abra Actions e execute "
        "'AltChart weekly refresh' em Run workflow. Depois atualize esta pagina"
    )
    st.stop()

forecast = forecast.copy()
for column in ["chart_date", "target_date"]:
    if column in forecast.columns:
        forecast[column] = pd.to_datetime(forecast[column], errors="coerce")

latest_chart_date = forecast["chart_date"].dropna().max() if "chart_date" in forecast else None
target_date = forecast["target_date"].dropna().max() if "target_date" in forecast else None
model_name = metadata.get("model_name", "—")
cv_f1 = metadata.get("cv_f1_macro")
verified = history[history["actual_trend_next"].notna()].copy() if history is not None and "actual_trend_next" in history else pd.DataFrame()
verified_accuracy = None
if not verified.empty:
    truth = verified["correct"].astype(str).str.lower().isin(["true", "1", "1.0"])
    verified_accuracy = truth.mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Chart atual", fmt_date(latest_chart_date))
m2.metric("Previsao para", fmt_date(target_date))
m3.metric("Modelo", str(model_name))
m4.metric("F1 macro validacao", f"{float(cv_f1):.3f}" if cv_f1 is not None else "—")

if verified_accuracy is not None:
    st.caption(f"Acerto das previsoes semanais ja verificadas: {verified_accuracy:.1%}")

st.divider()
st.subheader("Previsao do proximo chart")
st.caption("A previsao fica pendente ate a Billboard publicar a semana seguinte")

search = st.text_input("Buscar musica ou artista", placeholder="Digite um nome")
trend_options = ["Todas"] + [x for x in ["Subiu", "Desceu", "Manteve", "Saiu"] if x in forecast.get("predicted_trend", pd.Series(dtype=str)).astype(str).unique()]
selected_trend = st.selectbox("Filtrar previsao", trend_options)
view = forecast.copy()
if search.strip():
    mask = view["song"].astype(str).str.contains(search, case=False, na=False)
    mask |= view["artist"].astype(str).str.contains(search, case=False, na=False)
    view = view[mask]
if selected_trend != "Todas":
    view = view[view["predicted_trend"] == selected_trend]

view["Movimento atual"] = view["current_movement"].map(display_trend)
view["Previsao"] = view["predicted_trend"].map(display_trend)
view["Resultado"] = view.apply(
    lambda row: result_label(row.get("actual_trend_next"), row.get("correct")), axis=1
)
view["Confianca (%)"] = pd.to_numeric(view["confidence_pct"], errors="coerce")
view = view.sort_values("rank_at_forecast")

columns = [
    "rank_at_forecast", "song", "artist", "Movimento atual", "Previsao",
    "Confianca (%)", "Resultado",
]
columns = [column for column in columns if column in view.columns]
display_table = view[columns].rename(columns={
    "rank_at_forecast": "Posicao",
    "song": "Musica",
    "artist": "Artista",
})
st.dataframe(display_table, use_container_width=True, hide_index=True)

if history is not None and not verified.empty:
    st.divider()
    st.subheader("Previsoes do projeto ja verificadas")
    checked = verified.copy()
    checked["chart_date"] = pd.to_datetime(checked["chart_date"], errors="coerce")
    checked["target_date"] = pd.to_datetime(checked["target_date"], errors="coerce")
    checked = checked.sort_values(["target_date", "rank_at_forecast"], ascending=[False, True]).head(40).copy()
    checked["Previsao"] = checked["predicted_trend"].map(display_trend)
    checked["Resultado real"] = checked["actual_trend_next"].map(display_trend)
    checked["Resultado"] = checked.apply(
        lambda row: result_label(row.get("actual_trend_next"), row.get("correct")), axis=1
    )
    st.dataframe(
        checked[["target_date", "rank_at_forecast", "song", "artist", "Previsao", "Resultado real", "Resultado"]].rename(columns={
            "target_date": "Semana avaliada",
            "rank_at_forecast": "Posicao original",
            "song": "Musica",
            "artist": "Artista",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("Previsoes historicas verificadas")
if backtest is not None and not backtest.empty:
    backtest = backtest.copy()
    backtest["target_date"] = pd.to_datetime(backtest["target_date"], errors="coerce")
    weeks = sorted(backtest["target_date"].dropna().unique())
    selected_week = st.selectbox(
        "Semana avaliada",
        options=weeks,
        index=len(weeks) - 1,
        format_func=fmt_date,
    )
    week_view = backtest[backtest["target_date"] == selected_week].copy()
    correct_values = week_view["acertou"].astype(str).str.lower().isin(["true", "1", "1.0"])
    w1, w2, w3 = st.columns(3)
    w1.metric("Musicas avaliadas", len(week_view))
    w2.metric("Acertos", f"{int(correct_values.sum())}/{len(week_view)}" if len(week_view) else "—")
    w3.metric("Taxa de acerto", f"{correct_values.mean():.1%}" if len(week_view) else "—")

    week_view["Previsao"] = week_view["previsto"].map(display_trend)
    week_view["Resultado real"] = week_view["real"].map(display_trend)
    week_view["Resultado"] = correct_values.map({True: "Acertou", False: "Errou"}).values
    week_view = week_view.sort_values("rank")
    st.dataframe(
        week_view[["rank", "song", "artist", "Previsao", "Resultado real", "Resultado"]].rename(columns={
            "rank": "Posicao na semana anterior",
            "song": "Musica",
            "artist": "Artista",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("O backtest historico aparece aqui depois da primeira execucao do pipeline")

if weekly is not None and not weekly.empty:
    st.subheader("Acerto por semana")
    weekly = weekly.copy()
    weekly["target_date"] = pd.to_datetime(weekly["target_date"], errors="coerce")
    weekly = weekly.dropna(subset=["target_date"]).sort_values("target_date").set_index("target_date")
    chart_cols = [col for col in ["acuracia", "f1_macro"] if col in weekly.columns]
    if chart_cols:
        st.line_chart(weekly[chart_cols])

if comparison is not None:
    st.divider()
    st.subheader("Comparacao dos modelos")
    display_cols = [col for col in ["modelo", "cv_f1_macro", "acuracia", "precisao_macro", "recall_macro", "f1_macro"] if col in comparison.columns]
    st.dataframe(
        comparison[display_cols].rename(columns={
            "modelo": "Modelo",
            "cv_f1_macro": "F1 macro validacao",
            "acuracia": "Acuracia",
            "precisao_macro": "Precisao macro",
            "recall_macro": "Recall macro",
            "f1_macro": "F1 macro teste",
        }).round(3),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "Projeto experimental de ML para acompanhar o movimento semanal do Billboard Alternative Airplay. "
    "Confianca e estimativa do modelo, nao garantia de resultado"
)
