
import pandas as pd
import numpy as np
import datetime as dt

def _try_import_statsmodels():
    try:
        import statsmodels.api as sm
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from statsmodels.tsa.arima.model import ARIMA
        return sm, ExponentialSmoothing, ARIMA
    except Exception:
        return None, None, None

def make_synthetic_history(dem_df: pd.DataFrame, days:int=180, seed:int=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=days-1)
    idx = pd.date_range(start, end, freq="D")
    combos = dem_df[["protein_class","segment","site"]].drop_duplicates()
    rows = []
    for _, c in combos.iterrows():
        base = 20.0
        weekly = (np.sin(np.arange(len(idx)) * 2*np.pi/7) + 1.2)
        trend = np.ones(len(idx))
        # Segment-specific adjustments
        if c["segment"] == "hospital":
            base *= 1.3
            trend = np.linspace(0.95, 1.05, len(idx))
        elif c["segment"] == "pharma":
            base *= 1.6
            trend = np.linspace(1.0, 1.2, len(idx))  # growth
        elif c["segment"] == "research":
            base *= 0.7
            weekly = (np.sin(np.arange(len(idx)) * 2*np.pi/14) + 1.1)
        elif c["segment"] == "biotech":
            base *= 0.9
            trend = np.linspace(1.1, 0.8, len(idx))  # decaying
        elif c["segment"] == "blood_bank":
            base *= 1.1
            weekly = (np.sin(np.arange(len(idx)) * 2*np.pi/7 - 1.0) + 1.15)  # phase shift
        elif c["segment"] == "export":
            base *= 1.2
            weekly = (np.sin(np.arange(len(idx)) * 2*np.pi/30) + 1.1)  # monthly-ish

        # Protein-specific bumps
        if c["protein_class"] == "IgG":
            base *= 1.3
        elif c["protein_class"] == "Fibrinogen":
            base *= 0.85

        noise = rng.normal(1.0, 0.15, len(idx))
        series = np.maximum(base * weekly * trend * noise, 0.0)

        rows.extend([{
            "date": d.date().isoformat(),
            "protein_class": c["protein_class"],
            "segment": c["segment"],
            "site": c["site"],
            "demand_l": float(v)
        } for d, v in zip(idx, series)])
    return pd.DataFrame(rows)

def forecast_next(history: pd.DataFrame, horizon:int=14) -> pd.DataFrame:
    sm, ETS, ARIMA = _try_import_statsmodels()
    out = []
    for (prot, seg, site), grp in history.groupby(["protein_class","segment","site"]):
        s = grp.sort_values("date")
        y = s["demand_l"].to_numpy()
        if sm is not None:
            try:
                model = ETS(y, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated")
                fit = model.fit(optimized=True)
                yhat = fit.forecast(horizon)
            except Exception:
                try:
                    yhat = ARIMA(y, order=(1,1,1)).fit().forecast(horizon)
                except Exception:
                    yhat = np.full(horizon, y[-7:].mean() if len(y)>=7 else y.mean())
        else:
            yhat = np.full(horizon, y[-7:].mean() if len(y)>=7 else y.mean())

        start = pd.to_datetime(s["date"].iloc[-1]) + pd.Timedelta(days=1)
        for k, v in enumerate(yhat):
            out.append({
                "date": (start + pd.Timedelta(days=k)).date().isoformat(),
                "protein_class": prot, "segment": seg, "site": site,
                "forecast_l": float(max(0.0, v))
            })
    return pd.DataFrame(out)

def build_weights_from_forecast(dem_df: pd.DataFrame, fc_df: pd.DataFrame, alpha: float = 1.0) -> dict:
    dem = dem_df.copy()
    dem["due_date"] = pd.to_datetime(dem["due_date"]).dt.date
    weights = {}
    for idx, row in dem.iterrows():
        sub = fc_df[(fc_df["protein_class"]==row["protein_class"]) & (fc_df["segment"]==row["segment"]) & (fc_df["site"]==row["site"])]
        if sub.empty:
            w = 1.0
        else:
            sub = sub.copy()
            sub["date"] = pd.to_datetime(sub["date"]).dt.date
            sub["dist"] = sub["date"].apply(lambda d: abs((d - row["due_date"]).days))
            best = sub.sort_values("dist").iloc[0]
            w = max(0.1, float(best["forecast_l"]))
        weights[idx] = alpha * w
    return weights
