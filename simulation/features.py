from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-12

def add_resistance_conductance(df: pd.DataFrame, v_col: str, i_col: str) -> pd.DataFrame:
    v = df[v_col].astype(float).to_numpy()
    i = df[i_col].astype(float).to_numpy()

    r = np.where(np.abs(i) > EPS, v / i, np.nan)
    g = np.where(np.abs(v) > EPS, i / v, np.nan)

    out = df.copy()
    out["resistance_ohm"] = r
    out["conductance_S"] = g
    return out
