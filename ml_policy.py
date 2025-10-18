import numpy as np
import pandas as pd
import datetime as dt

def _encode_features(df):
    out = pd.DataFrame()
    out["requested_volume_l"] = df["requested_volume_l"].astype(float)
    out["priority"] = df["priority"].astype(float)
    today = dt.date.today()
    out["days_to_due"] = (pd.to_datetime(df["due_date"]).dt.date - today).apply(lambda x: x.days).astype(float)
    out = pd.concat([
        out,
        pd.get_dummies(df["segment"], prefix="seg"),
        pd.get_dummies(df["protein_class"], prefix="prot"),
        pd.get_dummies(df["min_titer"], prefix="titer_min"),
        (df["blood_type"].astype(str).str.len()>0).rename("needs_btype").astype(int)
    ], axis=1)
    return out

def generate_history_and_train(inv_df, dem_df, seed=123):
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Install with `uv add scikit-learn`.") from e

    rng = np.random.default_rng(seed)
    hist = []
    for _, row in dem_df.iterrows():
        for _ in range(20):
            vol = max(1.0, row["requested_volume_l"] * float(rng.normal(1.0, 0.2)))
            dd_shift = int(rng.integers(-5, 10))
            prio = int(np.clip(row["priority"] + int(rng.integers(-1,2)), 1, 3))
            rarity = 0.0
            rarity += 1.0 if (row["protein_class"]=="IgG" and row["min_titer"]=="high") else 0.0
            rarity += 0.5 if (isinstance(row["blood_type"], str) and len(row["blood_type"])>0) else 0.0
            rarity += 0.5 if (row["segment"] in ["hospital","biotech"]) else 0.0
            today = dt.date.today()
            days_to_due = (pd.to_datetime(row["due_date"]).date() - today).days + dd_shift
            risk = (0.4 * (vol/400.0) + 0.3 * (max(0, 10 - days_to_due)/10.0) + 0.3 * rarity)
            urgency = 1 + 4 * np.tanh(risk)
            hist.append({
                "segment": row["segment"],
                "protein_class": row["protein_class"],
                "min_titer": row["min_titer"],
                "blood_type": row["blood_type"],
                "requested_volume_l": vol,
                "due_date": (pd.to_datetime(row["due_date"]) + pd.Timedelta(days=dd_shift)).date().isoformat(),
                "priority": prio,
                "urgency_score": float(urgency)
            })
    hist_df = pd.DataFrame(hist)
    X = _encode_features(hist_df)
    y = hist_df["urgency_score"].astype(float)
    model = GradientBoostingRegressor(random_state=seed)
    model.fit(X, y)
    return model

def score_orders(model, dem_df):
    X = _encode_features(dem_df)
    scores = model.predict(X)
    return np.clip(scores, 1.0, None)
