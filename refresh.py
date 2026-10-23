from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from billboard_charts import backfill, make_session
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
DATA_PATH = DATA_DIR / "alternative_airplay.csv"

CHART_SLUG = "alternative-airplay"
FIRST_WEEK = "2023-10-21"
HOLDOUT_WEEKS = 12
VALIDATION_WEEKS = 4
MIN_TRAIN_WEEKS = 26
RANDOM_STATE = 42

FEATURES = [
    "rank",
    "last_week_rank",
    "peak_position",
    "weeks_on_chart",
    "rank_movement",
    "rank_two_weeks_ago",
    "rank_movement_2w",
    "has_two_week_history",
    "is_first_week",
]
LABELS = ["Subiu", "Desceu", "Manteve", "Saiu"]

for folder in [DATA_DIR, REPORTS_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def last_saturday(today=None):
    if today is None:
        today = datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    return today - timedelta(days=(today.weekday() - 5) % 7)


def normalize_identity(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_history() -> pd.DataFrame:
    end_week = last_saturday().isoformat()
    result = backfill(
        slug=CHART_SLUG,
        csv_path=DATA_PATH,
        first_week=FIRST_WEEK,
        last_week=end_week,
        session=make_session(),
        checkpoint_every=12,
        max_attempts=3,
    )
    print(f"Semanas no historico: {result.weeks}")
    print(f"Semanas adicionadas nesta execucao: {len(result.written)}")
    print(f"Semanas indisponiveis ou com falha: {len(result.failed)}")

    if not DATA_PATH.exists():
        raise FileNotFoundError("O historico do chart nao foi criado")

    raw = pd.read_csv(DATA_PATH)
    rename_map = {
        "Date": "date",
        "Rank": "rank",
        "Song": "song",
        "Artist": "artist",
        "Last Week": "last_week_rank",
        "Peak Position": "peak_position",
        "Weeks on Chart": "weeks_on_chart",
    }
    raw = raw.rename(columns=rename_map)
    required = list(rename_map.values())
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no historico: {missing}")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    for col in ["rank", "last_week_rank", "peak_position", "weeks_on_chart"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["date", "rank", "song", "artist"]).copy()
    raw["last_week_rank"] = raw["last_week_rank"].fillna(0)
    raw["peak_position"] = raw["peak_position"].fillna(raw["rank"])
    raw["weeks_on_chart"] = raw["weeks_on_chart"].fillna(1)

    for col in ["rank", "last_week_rank", "peak_position", "weeks_on_chart"]:
        raw[col] = raw[col].astype(int)

    raw["song_key"] = (
        raw["song"].map(normalize_identity)
        + "|"
        + raw["artist"].map(normalize_identity)
    )
    raw = raw.sort_values(["date", "rank"])
    raw = raw.drop_duplicates(["date", "song_key"], keep="first")
    raw = raw.drop_duplicates(["date", "rank"], keep="first")
    raw = raw.sort_values(["date", "rank"]).reset_index(drop=True)

    if raw["date"].nunique() < MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + 12:
        raise ValueError("Historico insuficiente para treinar e testar o modelo")

    print(
        f"Historico pronto: {len(raw):,} linhas | {raw['date'].nunique()} semanas | "
        f"{raw['date'].min().date()} ate {raw['date'].max().date()}"
    )
    return raw


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    features = raw.copy()

    two_weeks_ago = raw[["date", "song_key", "rank"]].copy()
    two_weeks_ago["date"] = two_weeks_ago["date"] + pd.Timedelta(days=14)
    two_weeks_ago = two_weeks_ago.rename(columns={"rank": "rank_two_weeks_ago"})
    features = features.merge(two_weeks_ago, on=["date", "song_key"], how="left")

    features["rank_movement"] = np.where(
        features["last_week_rank"] > 0,
        features["last_week_rank"] - features["rank"],
        0,
    )
    features["rank_movement_2w"] = features["rank_two_weeks_ago"] - features["rank"]
    features["has_two_week_history"] = features["rank_two_weeks_ago"].notna().astype(int)
    features["rank_two_weeks_ago"] = features["rank_two_weeks_ago"].fillna(0)
    features["rank_movement_2w"] = features["rank_movement_2w"].fillna(0)
    features["is_first_week"] = (features["weeks_on_chart"] == 1).astype(int)

    available_dates = set(features["date"].unique())
    labeled_dates = {
        date for date in available_dates
        if date + pd.Timedelta(days=7) in available_dates
    }

    next_week = raw[["date", "song_key", "rank"]].copy()
    next_week["date"] = next_week["date"] - pd.Timedelta(days=7)
    next_week = next_week.rename(columns={"rank": "next_rank"})

    labeled = features[features["date"].isin(labeled_dates)].merge(
        next_week, on=["date", "song_key"], how="left"
    )

    labeled["target_trend"] = np.select(
        [
            labeled["next_rank"].isna(),
            labeled["next_rank"] < labeled["rank"],
            labeled["next_rank"] > labeled["rank"],
        ],
        ["Saiu", "Subiu", "Desceu"],
        default="Manteve",
    )
    labeled = labeled.sort_values(["date", "rank"]).reset_index(drop=True)
    return features, labeled


def make_temporal_splits(frame: pd.DataFrame, n_splits=4):
    weeks = np.array(sorted(frame["date"].unique()))
    possible = (len(weeks) - MIN_TRAIN_WEEKS) // VALIDATION_WEEKS
    n_splits = min(n_splits, possible)
    if n_splits < 2:
        raise ValueError("Historico insuficiente para validacao temporal")

    starts = np.linspace(
        MIN_TRAIN_WEEKS,
        len(weeks) - VALIDATION_WEEKS,
        n_splits,
        dtype=int,
    )
    splits = []
    for start in sorted(set(starts)):
        train_weeks = set(weeks[:start])
        valid_weeks = set(weeks[start:start + VALIDATION_WEEKS])
        train_idx = np.flatnonzero(frame["date"].isin(train_weeks).to_numpy())
        valid_idx = np.flatnonzero(frame["date"].isin(valid_weeks).to_numpy())
        if len(train_idx) and len(valid_idx):
            splits.append((train_idx, valid_idx))
    return splits


def trend_from_ranks(current_rank, next_rank) -> str:
    if pd.isna(next_rank):
        return "Saiu"
    if next_rank < current_rank:
        return "Subiu"
    if next_rank > current_rank:
        return "Desceu"
    return "Manteve"


def current_movement(row) -> str:
    previous = row.get("last_week_rank", 0)
    if pd.isna(previous) or int(previous) == 0:
        return "Estreia"
    if int(row["rank"]) < int(previous):
        return "Subiu"
    if int(row["rank"]) > int(previous):
        return "Desceu"
    return "Manteve"


def temporal_macro_f1(estimator, X, y_true):
    predictions = estimator.predict(X)
    return f1_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0)


def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao_macro": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }


def model_candidates():
    return {
        "Regressao logistica": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )),
            ]),
            {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [2, 5]},
        ),
        "KNN": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_jobs=1)),
            ]),
            {"model__n_neighbors": [5, 11, 21], "model__weights": ["uniform", "distance"]},
        ),
        "Gradient Boosting": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1], "model__max_depth": [1, 2]},
        ),
    }


def update_forecast_history(raw: pd.DataFrame, latest_date: pd.Timestamp, final_model, latest_features: pd.DataFrame):
    history_path = REPORTS_DIR / "forecast_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
    else:
        history = pd.DataFrame(columns=[
            "chart_date", "target_date", "song_key", "rank_at_forecast", "song", "artist",
            "current_movement", "predicted_trend", "confidence_pct", "actual_trend_next",
            "actual_next_rank", "correct",
        ])

    for col in ["chart_date", "target_date"]:
        if col in history.columns:
            history[col] = pd.to_datetime(history[col], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["actual_trend_next", "correct"]:
        if col in history.columns:
            history[col] = history[col].astype(object)

    available_dates = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in raw["date"].unique()}
    pending = history[history.get("actual_trend_next", pd.Series(index=history.index, dtype=object)).isna()]
    rank_lookup = {
        (pd.Timestamp(row.date).strftime("%Y-%m-%d"), row.song_key): int(row.rank)
        for row in raw[["date", "song_key", "rank"]].itertuples(index=False)
    }

    for index, row in pending.iterrows():
        target_date = str(row.get("target_date", ""))
        if target_date not in available_dates:
            continue
        actual_rank = rank_lookup.get((target_date, row["song_key"]))
        actual_trend = trend_from_ranks(row["rank_at_forecast"], actual_rank if actual_rank is not None else np.nan)
        history.loc[index, "actual_trend_next"] = actual_trend
        history.loc[index, "actual_next_rank"] = actual_rank if actual_rank is not None else np.nan
        history.loc[index, "correct"] = bool(str(row["predicted_trend"]) == actual_trend)

    latest_key = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
    already_forecast = (
        not history.empty
        and "chart_date" in history.columns
        and (history["chart_date"] == latest_key).any()
    )

    if not already_forecast:
        predictions = final_model.predict(latest_features[FEATURES])
        probabilities = final_model.predict_proba(latest_features[FEATURES])
        confidence = probabilities.max(axis=1) * 100
        rows = []
        target_date = (pd.Timestamp(latest_date) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

        for (_, row), prediction, score in zip(latest_features.iterrows(), predictions, confidence):
            rows.append({
                "chart_date": latest_key,
                "target_date": target_date,
                "song_key": row["song_key"],
                "rank_at_forecast": int(row["rank"]),
                "song": row["song"],
                "artist": row["artist"],
                "current_movement": current_movement(row),
                "predicted_trend": str(prediction),
                "confidence_pct": round(float(score), 1),
                "actual_trend_next": np.nan,
                "actual_next_rank": np.nan,
                "correct": np.nan,
            })
        new_rows = pd.DataFrame(rows)
        history = new_rows if history.empty else pd.concat([history, new_rows], ignore_index=True)

    history = history.drop_duplicates(["chart_date", "song_key"], keep="last")
    history = history.sort_values(["chart_date", "rank_at_forecast"]).reset_index(drop=True)
    history.to_csv(history_path, index=False)

    latest_forecast = history[history["chart_date"] == latest_key].copy()
    latest_forecast.to_csv(REPORTS_DIR / "latest_forecast.csv", index=False)
    return history, latest_forecast


def main():
    raw = load_history()
    features, labeled = build_features(raw)

    if labeled["target_trend"].nunique() < 3:
        raise ValueError("O historico ainda nao tem variacao suficiente entre as direcoes")

    all_weeks = sorted(labeled["date"].unique())
    if len(all_weeks) <= HOLDOUT_WEEKS + MIN_TRAIN_WEEKS + 8:
        raise ValueError("Semanas rotuladas insuficientes para separar treino e teste")

    test_weeks = set(all_weeks[-HOLDOUT_WEEKS:])
    development = labeled[~labeled["date"].isin(test_weeks)].copy().reset_index(drop=True)
    test = labeled[labeled["date"].isin(test_weeks)].copy().reset_index(drop=True)

    X_dev, y_dev = development[FEATURES], development["target_trend"]
    X_test, y_test = test[FEATURES], test["target_trend"]
    cv_splits = make_temporal_splits(development)

    searches = {}
    for name, (pipeline, param_grid) in model_candidates().items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=temporal_macro_f1,
            cv=cv_splits,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_dev, y_dev)
        searches[name] = search
        print(f"{name}: CV F1 macro = {search.best_score_:.3f} | {search.best_params_}")

    comparison_rows = []
    prediction_frames = []
    for name, search in searches.items():
        model = search.best_estimator_
        prediction = model.predict(X_test)
        metrics = calculate_metrics(y_test, prediction)
        comparison_rows.append({"modelo": name, "cv_f1_macro": search.best_score_, **metrics})

        out = test[["date", "rank", "song", "artist", "song_key", "target_trend"]].copy()
        out["target_date"] = out["date"] + pd.Timedelta(days=7)
        out["modelo"] = name
        out["previsao"] = prediction
        out["acertou"] = out["target_trend"] == out["previsao"]
        out["confianca_pct"] = np.round(model.predict_proba(X_test).max(axis=1) * 100, 1)
        prediction_frames.append(out)

    majority_label = y_dev.value_counts().idxmax()
    baseline_prediction = np.repeat(majority_label, len(y_test))
    baseline_metrics = calculate_metrics(y_test, baseline_prediction)
    comparison_rows.append({"modelo": "Baseline classe mais comum", "cv_f1_macro": np.nan, **baseline_metrics})

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "cv_f1_macro", ascending=False, na_position="last"
    ).reset_index(drop=True)
    test_predictions_all = pd.concat(prediction_frames, ignore_index=True)
    selected_name = max(searches, key=lambda name: searches[name].best_score_)
    selected_test = test_predictions_all[test_predictions_all["modelo"] == selected_name].copy()
    selected_test["acertou"] = selected_test["target_trend"] == selected_test["previsao"]

    weekly = (
        selected_test.groupby("target_date")
        .apply(lambda group: pd.Series({
            "musicas_avaliadas": len(group),
            "acuracia": accuracy_score(group["target_trend"], group["previsao"]),
            "f1_macro": f1_score(group["target_trend"], group["previsao"], labels=LABELS, average="macro", zero_division=0),
            "acertos": int(group["acertou"].sum()),
        }), include_groups=False)
        .reset_index()
    )

    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    weekly.to_csv(REPORTS_DIR / "weekly_backtest.csv", index=False)
    selected_test.rename(columns={"target_trend": "real", "previsao": "previsto"}).to_csv(
        REPORTS_DIR / "backtest_predictions.csv", index=False
    )

    # Refit using all known transitions to generate the next forecast
    final_model = clone(searches[selected_name].best_estimator_)
    final_model.fit(labeled[FEATURES], labeled["target_trend"])
    latest_date = features["date"].max()
    latest_features = features[features["date"] == latest_date].copy().sort_values("rank")
    history, latest_forecast = update_forecast_history(raw, latest_date, final_model, latest_features)

    model_path = MODELS_DIR / "altchart_direction_model.joblib"
    joblib.dump({
        "model": final_model,
        "model_name": selected_name,
        "features": FEATURES,
        "labels": LABELS,
    }, model_path)

    selected_row = comparison[comparison["modelo"] == selected_name].iloc[0]
    metadata = {
        "model_name": selected_name,
        "target": "direction_next_week",
        "cv_f1_macro": float(searches[selected_name].best_score_),
        "test_f1_macro": float(selected_row["f1_macro"]),
        "baseline_test_f1_macro": float(baseline_metrics["f1_macro"]),
        "trained_through_chart_date": pd.Timestamp(latest_date).date().isoformat(),
        "forecast_for": (pd.Timestamp(latest_date) + pd.Timedelta(days=7)).date().isoformat(),
        "test_start": pd.Timestamp(test["date"].min()).date().isoformat(),
        "test_end": pd.Timestamp(test["date"].max()).date().isoformat(),
        "features": FEATURES,
        "labels": LABELS,
        "source": "Billboard Alternative Airplay",
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Modelo selecionado: {selected_name}")
    print(f"F1 macro na validacao: {searches[selected_name].best_score_:.3f}")
    print(f"F1 macro no teste historico: {selected_row['f1_macro']:.3f}")
    print(f"F1 macro do baseline: {baseline_metrics['f1_macro']:.3f}")
    print(f"Previsoes atuais: {len(latest_forecast)}")
    print(f"Previsoes historicas verificadas: {history['correct'].notna().sum()}")
    print("Arquivos atualizados em data/, reports/ e models/")


if __name__ == "__main__":
    main()
