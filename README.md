# Plasma Inventory Optimizer — ML Edition

Adds a third approach: **ML-learned priorities**.
- Synthetic history from current demand
- Train GradientBoostingRegressor to predict an urgency score
- Use score as order weight in Pyomo LP (fallback to heuristic sorting)

## One-liner
```bash
uvx --with pandas --with numpy --with matplotlib --with pyomo --with scikit-learn streamlit run app.py
```

> To use LP, install a solver executable (HiGHS/CBC/GLPK).

Modes: Heuristic | Pyomo LP | ML (learned priorities)
