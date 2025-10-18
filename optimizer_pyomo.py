import datetime as dt
import pandas as pd
import numpy as np

def solve_with_pyomo(inv_df: pd.DataFrame,
                     dem_df: pd.DataFrame,
                     weights: dict,
                     expiry_bias: float = 0.001):
    try:
        import pyomo.environ as pyo
    except Exception as e:
        raise RuntimeError("Pyomo is not installed. Install with `uv add pyomo`.") from e

    titer_order = {"low": 0, "med": 1, "high": 2}
    parse = lambda s: dt.date.fromisoformat(str(s))

    inv = inv_df.copy()
    dem = dem_df.copy()
    inv["expiry_date"] = inv["expiry_date"].apply(parse)
    dem["due_date"] = dem["due_date"].apply(parse)

    I = list(inv.index)
    J = list(dem.index)

    compat_pairs = []
    for i in I:
        ir = inv.loc[i]
        if ir["compliance"] != "ok":
            continue
        for j in J:
            jr = dem.loc[j]
            if ir["protein_class"] != jr["protein_class"]:
                continue
            if titer_order.get(ir["titer"], 0) < titer_order.get(jr["min_titer"], 0):
                continue
            if str(jr["blood_type"]).strip() and (ir["blood_type"] != jr["blood_type"]):
                continue
            if ir["expiry_date"] < jr["due_date"]:
                continue
            compat_pairs.append((i, j))

    if len(compat_pairs) == 0:
        raise RuntimeError("No feasible (batch, order) pairs. Check protein/titer/blood-type/expiry/compliance filters.")

    max_horizon = max((d - dt.date.today()).days for d in inv["expiry_date"])
    days_to_exp = {i: (inv.loc[i, "expiry_date"] - dt.date.today()).days for i in I}
    bonus = {i: (max_horizon - max(0, days_to_exp[i])) for i in I}

    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=I)
    m.J = pyo.Set(initialize=J)
    m.A = pyo.Set(initialize=compat_pairs, dimen=2)
    m.x = pyo.Var(m.A, within=pyo.NonNegativeReals)

    def obj_rule(m):
        alloc_term = sum(weights.get(j,1.0) * m.x[i, j] for (i, j) in m.A)
        expiry_term = expiry_bias * sum(bonus[i] * sum(m.x[i, j] for j in m.J if (i, j) in m.A) for i in m.I)
        return alloc_term + expiry_term
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    inv_volume = inv["volume_l"].to_dict()
    def cap_rule(m, i):
        cols = [j for j in m.J if (i, j) in m.A]
        if not cols:
            return pyo.Constraint.Skip
        return sum(m.x[i, j] for j in cols) <= inv_volume[i]
    m.Capacity = pyo.Constraint(m.I, rule=cap_rule)

    dem_req = dem["requested_volume_l"].to_dict()
    def dem_rule(m, j):
        rows = [i for i in m.I if (i, j) in m.A]
        if not rows:
            return pyo.Constraint.Skip
        return sum(m.x[i, j] for i in rows) <= dem_req[j]
    m.Demand = pyo.Constraint(m.J, rule=dem_rule)

    from pyomo.opt import SolverFactory
    solved = False
    for solver_name in ["appsi_highs", "highs", "cbc", "glpk"]:
        try:
            opt = SolverFactory(solver_name)
            if opt is not None and opt.available(False):
                res = opt.solve(m, tee=False)
                solved = True
                break
        except Exception:
            continue
    if not solved:
        raise RuntimeError("No solver available. Install `highspy` (for appsi_highs) or a system binary: highs/cbc/glpk.")

    alloc_rows = []
    for (i, j) in m.A:
        val = pyo.value(m.x[i, j])
        if val and val > 1e-6:
            ir = inv.loc[i]
            jr = dem.loc[j]
            alloc_rows.append({
                "order_id": jr["order_id"],
                "segment": jr["segment"],
                "protein_class": jr["protein_class"],
                "min_titer": jr["min_titer"],
                "blood_type_req": jr["blood_type"],
                "due_date": jr["due_date"].isoformat(),
                "batch_id": ir["batch_id"],
                "batch_titer": ir["titer"],
                "batch_blood_type": ir["blood_type"],
                "expiry_date": ir["expiry_date"].isoformat(),
                "allocated_volume_l": float(round(val, 2))
            })
    alloc_df = pd.DataFrame(alloc_rows)

    if len(alloc_df) == 0:
        order_alloc = dem[["order_id","segment","protein_class","min_titer","blood_type","due_date","requested_volume_l"]].copy()
        order_alloc = order_alloc.rename(columns={"blood_type": "blood_type_req"})
        order_alloc["allocated_volume_l"] = 0.0
        order_alloc["fulfillment_pct"] = 0.0
    else:
        order_alloc = alloc_df.groupby(["order_id","segment","protein_class","min_titer","blood_type_req","due_date"], as_index=False)["allocated_volume_l"].sum()
        order_alloc = order_alloc.merge(dem[["order_id","requested_volume_l"]], on="order_id", how="left")
        order_alloc["fulfillment_pct"] = np.where(order_alloc["requested_volume_l"] > 0,
                                                 100 * order_alloc["allocated_volume_l"] / order_alloc["requested_volume_l"],
                                                 0.0)

    used_by_batch = alloc_df.groupby("batch_id", as_index=False)["allocated_volume_l"].sum() if len(alloc_df) > 0 else pd.DataFrame(columns=["batch_id","allocated_volume_l"])
    remaining_df = inv.merge(used_by_batch, on="batch_id", how="left").fillna({"allocated_volume_l": 0.0})
    remaining_df["remaining_l"] = remaining_df["volume_l"] - remaining_df["allocated_volume_l"]

    total_inv = float(remaining_df["volume_l"].sum())
    used_inv = float(remaining_df["allocated_volume_l"].sum())
    utilization = 100.0 * used_inv / total_inv if total_inv > 0 else 0.0

    soon = dt.date.today() + dt.timedelta(days=14)
    expiring_soon = float(remaining_df.loc[remaining_df["expiry_date"] <= soon, "remaining_l"].sum())
    expiry_risk_pct = 100.0 * expiring_soon / max(total_inv, 1.0)

    return alloc_df, order_alloc, utilization, expiry_risk_pct, remaining_df
