import numpy as np
import pandas as pd

from refresh import FEATURES, build_features, current_movement_baseline


def sample_history():
    start = pd.Timestamp("2025-01-04")
    rank_history = {
        "alpha": [5, 6, 4, 3, 3, 4],
        "beta": [12, 10, 11, None, None, None],
        "gamma": [28, 25, 24, 22, 21, 20],
    }
    rows = []
    for song_key, ranks in rank_history.items():
        for week, rank in enumerate(ranks):
            if rank is None:
                continue
            date = start + pd.Timedelta(days=7 * week)
            previous_rank = ranks[week - 1] if week > 0 else None
            past = [x for x in ranks[: week + 1] if x is not None]
            rows.append({
                "date": date,
                "rank": int(rank),
                "song": song_key.title(),
                "artist": "Test Artist",
                "last_week_rank": int(previous_rank) if previous_rank is not None else 0,
                "peak_position": int(min(past)),
                "weeks_on_chart": len(past),
                "song_key": song_key,
            })
    return pd.DataFrame(rows).sort_values(["date", "rank"]).reset_index(drop=True)


def test_build_features_only_uses_prior_chart_weeks():
    raw = sample_history()
    features, labeled = build_features(raw)

    target_date = pd.Timestamp("2025-01-25")
    alpha = features[(features["date"] == target_date) & (features["song_key"] == "alpha")].iloc[0]
    assert alpha["rank"] == 3
    assert alpha["rank_lag_1"] == 4
    assert alpha["rank_lag_2"] == 6
    assert "momentum_acceleration" in FEATURES
    assert not any(col.startswith("next_") for col in FEATURES)

    beta_leave = labeled[(labeled["date"] == pd.Timestamp("2025-01-18")) & (labeled["song_key"] == "beta")]
    assert beta_leave.iloc[0]["target_trend"] == "Saiu"


def test_feature_set_has_no_duplicate_names():
    assert len(FEATURES) == len(set(FEATURES))


def test_repeat_movement_baseline():
    frame = pd.DataFrame({"rank_movement_1w": [3, -2, 0, np.nan]})
    assert current_movement_baseline(frame).tolist() == ["Subiu", "Desceu", "Manteve", "Manteve"]
