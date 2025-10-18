# 🧬 Plasma Inventory Optimizer

**Forecast-aware, ML-augmented plasma allocation with downstream purpose tuning.**  
Built with Python, Streamlit, Pyomo, scikit-learn, and statsmodels.

> Hospitals, research labs, and pharma plants all need plasma — but not the same plasma, and not at the same time.  
> This tool simulates how to **forecast demand, allocate inventory, and tune priorities** in real time.

---

## 🚀 What It Does

The Plasma Inventory Optimizer allocates plasma batches to downstream customers under multiple strategies:

| Mode | Description |
|------|--------------|
| **Heuristic** | Fast greedy allocation (“oldest-first”, priority-weighted) |
| **Pyomo LP** | Linear optimization for globally optimal allocation |
| **ML (learned priorities)** | Learns urgency/importance weights from past data |
| **Forecast-aware** | Uses ARIMA/ETS forecasts per `(protein, segment, site)` to anticipate demand |

---

## 🧠 Why It Exists

Grifols (and similar plasma companies) must match **supply variability** (donations, expiry windows) with **volatile demand** from:
- 🏥 **Hospitals** – transfusions and rare-disease treatments  
- 💊 **Pharma manufacturing** – immunoglobulins, clotting factors, albumin  
- 🧪 **Research institutions** – controlled, type-specific plasma for experimentation  

This app helps visualize trade-offs between urgency, throughput, and specificity — and quantify them.

---

## 🧩 Key Features

- **Synthetic data generator** for plasma inventory & demand
- **Forecasting** (ETS/ARIMA fallback) using `statsmodels`
- **Pyomo optimization** with HiGHS / CBC / GLPK support
- **ML weighting** using Gradient Boosting (`scikit-learn`)
- **Interactive tuning sliders**:
  - Hospital urgency / base weight
  - Pharma throughput bias
  - Research specificity bias
- **Metrics dashboard**:
  - Utilization rate (% of inventory allocated)
  - Expiry risk (volume expiring ≤ 14 days)
  - Fulfillment by segment

---

## 🏗️ Architecture

```
app.py               # Streamlit interface (tabs, sidebar, sliders)
optimizer_pyomo.py   # LP/MIP model (Pyomo) with solver chain and safety guard
ml_policy.py         # GradientBoosting-based urgency model
forecasting.py       # Synthetic history + ETS/ARIMA forecasts
sample_inventory_*.csv
sample_demand_*.csv  # Synthetic datasets
.github/workflows/ci.yml  # CI import & LP sanity test
```

---

## ⚙️ Quickstart

### Option A — One-liner with [uv](https://github.com/astral-sh/uv)

```bash
uvx --with pandas --with numpy --with matplotlib     --with pyomo --with scikit-learn --with statsmodels     --with highspy streamlit run app.py
```

### Option B — Classic virtualenv

```bash
python3 -m venv venv
source venv/bin/activate          # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install highspy               # optional, HiGHS solver
streamlit run app.py
```

Then open <http://localhost:8501>.

---

## 🎛️ Sidebar Controls

| Category | Control | Purpose |
|-----------|----------|----------|
| **Dataset & Mode** | Choose SMALL/LARGE synthetic data; select mode (Heuristic / LP / ML / Forecast) | |
| **LP Settings** | *Expiry bias* | Nudges LP to prefer soon-expiring plasma |
| **Forecast Settings** | *Horizon, α (weight scale)* | Forecast horizon (days ahead) and forecast weight amplitude |
| **Downstream Purpose Tuning** | *Hospitals, Pharma, Research sliders* | Adjust segment-specific weighting logic |
| **Urgency Window** | *Due-date window (days)* | Defines when hospital urgency ramps up |

---

## 🧮 Under the Hood

- **Hospitals**  
  `weight = hosp_base * (1 + hosp_urg * near_due)`
- **Pharma**  
  `weight = pharma_base * (1 + pharma_thr * normalized_volume)`
- **Research**  
  `weight = res_base * (1 + res_spec * specificity_score)`
- **LP Objective**  
  Maximize `Σ weight[j]·x[i,j] + expiry_bias·Σ bonus[i]·x[i,*]`
- **Constraints**  
  - Capacity per batch  
  - Demand per order  
  - Non-negativity  
  - Compatibility: protein, min-titer, blood-type (if specified), due-date vs expiry

**Solver chain**: `appsi_highs` (via `highspy`) → `highs` → `cbc` → `glpk`.  
If none are available, the app **falls back to the heuristic allocator** (ML/forecast modes still drive weights).

---

## 🔬 Example Scenarios

### 🩸 Clinical Crisis (protect hospitals)
| Segment | Base | Boost | Notes |
|----------|------|--------|-------|
| Hospitals | 4.5 | 2.0 | wide urgency window |
| Pharma | 1.5 | 0.3 | deprioritized |
| Research | 1.0 | 0.5 | reduced |

### 💉 Throughput Maximizer (serve pharma first)
| Segment | Base | Boost | Notes |
|----------|------|--------|-------|
| Hospitals | 3.0 | 0.6 | narrow window |
| Pharma | 3.5 | 2.0 | large volume bias |
| Research | 1.2 | 0.5 | steady |

### 🧪 R&D Sprint
| Segment | Base | Boost | Notes |
|----------|------|--------|-------|
| Hospitals | 3.0 | 0.8 | |
| Pharma | 1.5 | 0.2 | |
| Research | 2.5 | 1.8 | titer-focused |

---

## 🧰 CI

`.github/workflows/ci.yml` runs:
- Dependency check
- Import validation
- Tiny LP solve with Pyomo  
(uses HiGHS if available, otherwise skips optimization)

---

## ⚠️ Troubleshooting

| Issue | Cause | Fix |
|-------|--------|-----|
| **Solver unavailable** | No HiGHS/CBC/GLPK installed | `pip install highspy` or install a system solver |
| **Invalid constraint expression** | Empty compatibility set | (Fixed) we now `Constraint.Skip` empty rows; check filters |
| **Streamlit blank page** | Missing rerun or missing data files | `streamlit run app.py` again |
| **Repo push denied** | Repo doesn’t exist or bad auth | Create repo then `git push -u origin main` |

---

## 🧩 Roadmap

- [ ] Add *site-level capacity* constraints (daily caps)
- [ ] Add *holiday & seasonality* awareness to forecasts
- [ ] Add *SLA penalties* for under-fulfillment
- [ ] Presets: Clinical / Throughput / R&D (save & load)
- [ ] Docker container + Databricks notebook version

---

## 👤 Author

**Luis Herrera** — innovation & AI systems

---

## 🪪 License
MIT License — free for educational and demo use.
