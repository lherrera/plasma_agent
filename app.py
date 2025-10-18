import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

st.set_page_config(page_title="Plasma Optimizer — Purposes Plus", layout="wide")
st.title("🧬 Plasma Inventory Optimizer — Heuristic · Pyomo · ML · Forecast-aware")
st.caption("Expanded downstream segments + purpose tuning sliders.")

def parse_date(s):
    return dt.date.fromisoformat(str(s))

titer_order = {"low":0,"med":1,"high":2}

def compatible(inv_row, order_row):
    if inv_row["compliance"] != "ok":
        return False
    if inv_row["protein_class"] != order_row["protein_class"]:
        return False
    if titer_order.get(inv_row["titer"],0) < titer_order.get(order_row["min_titer"],0):
        return False
    if str(order_row["blood_type"]).strip() != "" and inv_row["blood_type"] != order_row["blood_type"]:
        return False
    if parse_date(inv_row["expiry_date"]) < parse_date(order_row["due_date"]):
        return False
    return True

def allocate_heuristic(inventory, demand, weights=None):
    inv = inventory.copy().reset_index(drop=True)
    dem = demand.copy().reset_index(drop=True)

    inv["expiry_date"] = inv["expiry_date"].apply(parse_date)
    inv = inv.sort_values(by=["expiry_date"])

    dem["due_date"] = dem["due_date"].apply(parse_date)
    if isinstance(weights, dict) or isinstance(weights, pd.Series):
        dem["weighted_priority"] = pd.Series(weights).reindex(dem.index).fillna(dem["priority"]).astype(float)
    else:
        dem["weighted_priority"] = dem["priority"].astype(float)
    dem = dem.sort_values(by=["weighted_priority","due_date"], ascending=[False, True])

    allocations = []
    remaining_inv = inv["volume_l"].to_numpy().copy()

    for _, order in dem.iterrows():
        needed = order["requested_volume_l"]
        for i_idx, inv_row in inv.iterrows():
            if needed <= 0:
                break
            if remaining_inv[i_idx] <= 0:
                continue
            if not compatible(inv_row, order):
                continue
            take = min(remaining_inv[i_idx], needed)
            remaining_inv[i_idx] -= take
            needed -= take
            allocations.append({
                "order_id": order["order_id"],
                "segment": order["segment"],
                "protein_class": order["protein_class"],
                "min_titer": order["min_titer"],
                "blood_type_req": order["blood_type"],
                "due_date": order["due_date"].isoformat(),
                "batch_id": inv_row["batch_id"],
                "batch_titer": inv_row["titer"],
                "batch_blood_type": inv_row["blood_type"],
                "expiry_date": inv_row["expiry_date"].isoformat(),
                "allocated_volume_l": round(float(take),2)
            })
        if needed > 0:
            allocations.append({
                "order_id": order["order_id"],
                "segment": order["segment"],
                "protein_class": order["protein_class"],
                "min_titer": order["min_titer"],
                "blood_type_req": order["blood_type"],
                "due_date": order["due_date"].isoformat(),
                "batch_id": "",
                "batch_titer": "",
                "batch_blood_type": "",
                "expiry_date": "",
                "allocated_volume_l": 0.0
            })

    alloc_df = pd.DataFrame(allocations)
    order_alloc = alloc_df.groupby(["order_id","segment","protein_class","min_titer","blood_type_req","due_date"], as_index=False)["allocated_volume_l"].sum()
    order_alloc = order_alloc.merge(dem[["order_id","requested_volume_l"]], on="order_id", how="left")
    order_alloc["fulfillment_pct"] = np.where(order_alloc["requested_volume_l"]>0, 100*order_alloc["allocated_volume_l"]/order_alloc["requested_volume_l"], 0)

    total_inv = inv["volume_l"].sum()
    used_inv = float(sum(inv["volume_l"]) - sum(remaining_inv))
    utilization = 100 * used_inv / total_inv if total_inv>0 else 0

    soon = dt.date.today() + dt.timedelta(days=14)
    remaining_df = inv.copy()
    remaining_df["remaining_l"] = remaining_inv
    expiring_soon = remaining_df.loc[remaining_df["expiry_date"]<=soon, "remaining_l"].sum()
    expiry_risk_pct = 100 * expiring_soon / max(total_inv, 1)

    return alloc_df, order_alloc, utilization, expiry_risk_pct, remaining_df

# Sidebar
st.sidebar.header("Dataset & Mode")
dataset_size = st.sidebar.selectbox("Choose sample dataset", ["SMALL (5x4)", "LARGE (700x220)"], index=1)
use_uploads = st.sidebar.checkbox("Upload my own CSVs", value=False)
mode = st.sidebar.radio("Mode", ["Heuristic","Pyomo LP","ML (learned priorities)","Forecast-aware (LP/Heuristic)"], index=3)

st.sidebar.header("LP Settings")
expiry_bias = st.sidebar.number_input("LP expiry bias (0–0.01)", min_value=0.0, max_value=0.05, value=0.001, step=0.001, format="%.3f")

st.sidebar.header("Forecast Settings")
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14, 1)
alpha = st.sidebar.slider("Forecast weight scale (alpha)", 0.1, 5.0, 1.0, 0.1)

# Purpose sliders (including new segments)
st.sidebar.header("Downstream Purpose Tuning")
st.sidebar.caption("Multipliers reshape objective weights by customer type.")
hosp_urg   = st.sidebar.slider("🏥 Hospitals — Due-date urgency boost", 0.0, 3.0, 1.2, 0.1)
hosp_base  = st.sidebar.slider("🏥 Hospitals — Base weight", 0.5, 5.0, 3.0, 0.1)
pharma_thr = st.sidebar.slider("💊 Pharma — Throughput/volume boost", 0.0, 3.0, 1.2, 0.1)
pharma_base= st.sidebar.slider("💊 Pharma — Base weight", 0.5, 5.0, 2.5, 0.1)
res_spec   = st.sidebar.slider("🧪 Research — Specificity boost", 0.0, 3.0, 1.2, 0.1)
res_base   = st.sidebar.slider("🧪 Research — Base weight", 0.5, 5.0, 1.8, 0.1)
bio_base   = st.sidebar.slider("🧫 Biotech — Base weight", 0.5, 5.0, 2.0, 0.1)
bb_base    = st.sidebar.slider("🚑 Blood banks — Base weight", 0.5, 5.0, 2.2, 0.1)
exp_base   = st.sidebar.slider("🌍 Export — Base weight", 0.5, 5.0, 1.6, 0.1)
due_urg_threshold = st.sidebar.slider("Due-date urgency window (days)", 3, 30, 10, 1)

def apply_purpose_tuning(dem_df, base_weights: pd.Series) -> pd.Series:
    w = pd.Series(base_weights).copy().astype(float)
    today = dt.date.today()
    days_to_due = (pd.to_datetime(dem_df["due_date"]).dt.date - today).map(lambda d: d.days)
    near_due = (np.clip(due_urg_threshold - days_to_due, 0, None) / max(due_urg_threshold,1))

    seg = dem_df["segment"]
    is_hosp = seg == "hospital"
    is_pharma = seg == "pharma"
    is_res = seg == "research"
    is_bio = seg == "biotech"
    is_bb = seg == "blood_bank"
    is_exp = seg == "export"

    # Hospitals: urgency + base
    w.loc[is_hosp] *= hosp_base * (1.0 + hosp_urg * near_due.loc[is_hosp])

    # Pharma: throughput bias + base
    vol = dem_df["requested_volume_l"].astype(float)
    vol_norm = (vol - vol.min())/max(1e-9, (vol.max()-vol.min()))
    w.loc[is_pharma] *= pharma_base * (1.0 + pharma_thr * vol_norm.loc[is_pharma])

    # Research: specificity signal
    spec = (dem_df["min_titer"].map({"low":0,"med":0.5,"high":1.0}).fillna(0.0) + (dem_df["blood_type"].astype(str).str.len()>0).astype(float))
    spec = np.clip(spec, 0, 2) / 2.0
    w.loc[is_res] *= res_base * (1.0 + res_spec * spec.loc[is_res])

    # Biotech / Blood bank / Export: base multipliers
    w.loc[is_bio] *= bio_base
    w.loc[is_bb]  *= bb_base
    w.loc[is_exp] *= exp_base

    return w.replace([np.inf,-np.inf], np.nan).fillna(1.0).clip(lower=0.01)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

if use_uploads:
    inv_file = st.sidebar.file_uploader("Inventory CSV", type=["csv"])
    dem_file = st.sidebar.file_uploader("Demand CSV", type=["csv"])
    if not inv_file or not dem_file: st.stop()
    inv_df = load_csv(inv_file)
    dem_df = load_csv(dem_file)
else:
    inv_df = load_csv("sample_inventory_small.csv" if dataset_size.startswith("SMALL") else "sample_inventory_large.csv")
    dem_df = load_csv("sample_demand_small.csv" if dataset_size.startswith("SMALL") else "sample_demand_large.csv")

tab1, tab2, tab3 = st.tabs(["📦 Inventory","📥 Demand","🚚 Allocation"])

with tab1:
    st.dataframe(inv_df.head(100), use_container_width=True)

with tab2:
    st.dataframe(dem_df.head(100), use_container_width=True)

with tab3:
    if st.button("Optimize Now"):
        if mode=="Heuristic":
            base_w = dem_df["priority"].astype(float)
            tuned = apply_purpose_tuning(dem_df, base_w)
            alloc_df, order_alloc, util, exp_risk, remaining_df = allocate_heuristic(inv_df, dem_df, weights=tuned)

        elif mode=="Pyomo LP":
            base_w = (dem_df["segment"].map({"hospital":3.0,"pharma":2.5,"research":1.8,"biotech":2.0,"blood_bank":2.2,"export":1.6}).fillna(1.0) * dem_df["priority"].astype(float))
            tuned = apply_purpose_tuning(dem_df, base_w)
            try:
                from optimizer_pyomo import solve_with_pyomo
                alloc_df, order_alloc, util, exp_risk, remaining_df = solve_with_pyomo(inv_df, dem_df, tuned.to_dict(), expiry_bias=expiry_bias)
            except Exception as e:
                st.error(f"Pyomo/solver error: {e}")
                st.stop()

        elif mode=="ML (learned priorities)":
            try:
                from ml_policy import generate_history_and_train, score_orders
                model = generate_history_and_train(inv_df, dem_df, seed=123)
                scores = pd.Series(score_orders(model, dem_df), index=dem_df.index)
                tuned = apply_purpose_tuning(dem_df, scores)
                from optimizer_pyomo import solve_with_pyomo
                alloc_df, order_alloc, util, exp_risk, remaining_df = solve_with_pyomo(inv_df, dem_df, tuned.to_dict(), expiry_bias=expiry_bias)
            except Exception as e:
                st.warning(f"Solver unavailable; using heuristic with tuned ML weights. ({e})")
                alloc_df, order_alloc, util, exp_risk, remaining_df = allocate_heuristic(inv_df, dem_df, weights=tuned)

        else:  # Forecast-aware
            try:
                import forecasting as fc
            except Exception as e:
                st.error(f"Forecast module missing: {e}")
                st.stop()
            history = fc.make_synthetic_history(dem_df, days=180, seed=123)
            fc_df = fc.forecast_next(history, horizon=horizon)
            base_w = pd.Series(fc.build_weights_from_forecast(dem_df, fc_df, alpha=alpha))
            tuned = apply_purpose_tuning(dem_df, base_w)
            try:
                from optimizer_pyomo import solve_with_pyomo
                alloc_df, order_alloc, util, exp_risk, remaining_df = solve_with_pyomo(inv_df, dem_df, tuned.to_dict(), expiry_bias=expiry_bias)
            except Exception as e:
                st.warning(f"Solver unavailable; using heuristic with tuned forecast weights. ({e})")
                alloc_df, order_alloc, util, exp_risk, remaining_df = allocate_heuristic(inv_df, dem_df, weights=tuned)

        st.markdown(f"**Utilization:** {util:.1f}% &nbsp;&nbsp; **Expiry Risk (≤14 days):** {exp_risk:.1f}%")
        st.dataframe(order_alloc.sort_values('fulfillment_pct', ascending=False), use_container_width=True)
        st.dataframe(alloc_df, use_container_width=True)

        by_seg = order_alloc.groupby("segment", as_index=False)["fulfillment_pct"].mean()
        fig = plt.figure()
        plt.bar(by_seg["segment"], by_seg["fulfillment_pct"])
        plt.title("Average Fulfillment by Segment")
        plt.ylabel("Fulfillment (%)")
        plt.xlabel("Segment")
        st.pyplot(fig)

        soon = dt.date.today() + dt.timedelta(days=14)
        if "expiry_date" in remaining_df.columns:
            at_risk = remaining_df.loc[remaining_df["expiry_date"] <= soon, ["batch_id","expiry_date","remaining_l"]].copy()
            st.markdown("**Remaining Inventory Expiring ≤14 days:**")
            st.dataframe(at_risk.sort_values("expiry_date"), use_container_width=True)

st.info("One-liner: uvx --with pandas --with numpy --with matplotlib --with pyomo --with scikit-learn --with statsmodels streamlit run app.py  (add --with highspy for HiGHS via Appsi)")