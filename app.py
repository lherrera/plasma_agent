
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

st.set_page_config(page_title="Plasma Optimizer — Heuristic vs Pyomo vs ML", layout="wide")
st.title("🧬 Plasma Inventory Optimizer — Heuristic vs Pyomo vs ML")
st.caption("ML mode learns urgency scores from synthetic history via scikit-learn; used as weights in Pyomo LP (fallback to heuristic ordering if solver missing).")

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
    if weights is None:
        dem["weighted_priority"] = dem["priority"].astype(float)
    else:
        dem["weighted_priority"] = pd.Series(weights).reindex(dem.index).fillna(dem["priority"]).astype(float)
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

def to_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")

st.sidebar.header("Dataset")
dataset_size = st.sidebar.selectbox("Choose sample dataset", ["SMALL (5x4)", "LARGE (500x120)"], index=1)
use_uploads = st.sidebar.checkbox("Upload my own CSVs", value=False)
inv_file = dem_file = None
if use_uploads:
    inv_file = st.sidebar.file_uploader("Upload inventory CSV", type=["csv"])
    dem_file = st.sidebar.file_uploader("Upload demand CSV", type=["csv"])

mode = st.sidebar.radio("Mode", ["Heuristic (fast)","Pyomo LP (optimal)","ML (learned priorities)"], index=2)
expiry_bias = st.sidebar.number_input("LP expiry bias (0–0.01)", min_value=0.0, max_value=0.05, value=0.001, step=0.001, format="%.3f")

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

if use_uploads:
    if inv_file and dem_file:
        inv_df = pd.read_csv(inv_file)
        dem_df = pd.read_csv(dem_file)
    else:
        st.stop()
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
        if mode.startswith("Heuristic"):
            alloc_df, order_alloc, util, exp_risk, remaining_df = allocate_heuristic(inv_df, dem_df, weights=None)

        elif mode.startswith("Pyomo"):
            try:
                from optimizer_pyomo import solve_with_pyomo
                seg_w = {"hospital":3.0,"pharma":2.0,"research":1.0}
                weights = (dem_df["segment"].map(seg_w).fillna(1.0) * dem_df["priority"].astype(float)).to_dict()
                alloc_df, order_alloc, util, exp_risk, remaining_df = solve_with_pyomo(inv_df, dem_df, weights, expiry_bias=expiry_bias)
            except Exception as e:
                st.error(f"Pyomo/solver error: {e}")
                st.stop()

        else:
            try:
                from ml_policy import generate_history_and_train, score_orders
                model = generate_history_and_train(inv_df, dem_df, seed=123)
                scores = score_orders(model, dem_df)
                weights = {idx: float(scores[i]) for i, idx in enumerate(dem_df.index)}
            except Exception as e:
                st.error(f"ML training/scoring error (install scikit-learn): {e}")
                st.stop()
            try:
                from optimizer_pyomo import solve_with_pyomo
                alloc_df, order_alloc, util, exp_risk, remaining_df = solve_with_pyomo(inv_df, dem_df, weights, expiry_bias=expiry_bias)
            except Exception as e:
                st.warning(f"Solver unavailable; using heuristic with learned weights. ({e})")
                weight_series = pd.Series(weights)
                alloc_df, order_alloc, util, exp_risk, remaining_df = allocate_heuristic(inv_df, dem_df, weights=weight_series)

        st.markdown(f"**Utilization:** {util:.1f}% &nbsp;&nbsp; **Expiry Risk (≤14 days):** {exp_risk:.1f}%")
        st.markdown("**Order Fulfillment (Aggregated):**")
        st.dataframe(order_alloc.sort_values('fulfillment_pct', ascending=False), use_container_width=True)
        st.markdown("**Allocations (Batch-Level):**")
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

# Footer
st.info("One-liner: uvx --with pandas --with numpy --with matplotlib --with pyomo --with scikit-learn streamlit run app.py")
