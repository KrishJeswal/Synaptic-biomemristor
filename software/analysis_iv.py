# software/analysis_iv.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class IVMetrics:
    run_id: str
    vread_V: float
    ion_A: Optional[float]
    ioff_A: Optional[float]
    on_off_ratio: Optional[float]
    vset_V: Optional[float]
    vreset_V: Optional[float]
    hysteresis_area: Optional[float]
    notes: str


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.abs(b) > eps
    out[mask] = a[mask] / b[mask]
    return out


def _closest_at_v(df: pd.DataFrame, v_target: float) -> Optional[pd.Series]:
    if df.empty:
        return None
    idx = (df["voltage_V"] - v_target).abs().idxmin()
    return df.loc[idx]


def _split_branches(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits an IV sweep into two branches based on the location of max |V|.
    Works for common sweeps like:
      0 -> +Vmax -> 0  OR  0 -> -Vmax -> 0  OR  -Vmax -> +Vmax (bipolar).
    """
    if df.empty:
        return df.copy(), df.copy()

    absmax_idx = df["voltage_V"].abs().idxmax()
    i = df.index.get_loc(absmax_idx)

    b1 = df.iloc[: i + 1].copy()
    b2 = df.iloc[i:].copy()
    return b1, b2


def _detect_switch_voltage(branch: pd.DataFrame, mode: str) -> Optional[float]:
    """
    Detect switching voltage using biggest jump in conductance along a branch.
    mode:
      - "set": find biggest positive jump (HRS -> LRS)
      - "reset": find biggest negative jump (LRS -> HRS)
    """
    if len(branch) < 5:
        return None

    v = branch["voltage_V"].to_numpy(dtype=float)
    i = branch["current_A"].to_numpy(dtype=float)

    g = _safe_div(i, v)  # conductance
    # Smooth very lightly to reduce noise sensitivity (moving average)
    g_s = pd.Series(g).rolling(window=3, center=True, min_periods=1).mean().to_numpy()

    dg = np.diff(g_s)
    if dg.size == 0:
        return None

    # Robust threshold to avoid calling noise "switching"
    # Use median absolute deviation as noise estimator.
    med = np.nanmedian(dg)
    mad = np.nanmedian(np.abs(dg - med))
    noise_scale = mad if mad > 0 else np.nanstd(dg)
    if not np.isfinite(noise_scale) or noise_scale == 0:
        noise_scale = 1e-12

    # Require a jump at least k * noise_scale
    k = 8.0
    if mode == "set":
        idx = int(np.nanargmax(dg))
        if dg[idx] < k * noise_scale:
            return None
        return float(v[idx + 1])
    elif mode == "reset":
        idx = int(np.nanargmin(dg))
        if dg[idx] > -k * noise_scale:
            return None
        return float(v[idx + 1])
    else:
        raise ValueError("mode must be 'set' or 'reset'")


def compute_iv_metrics(
    iv_csv_path: str,
    vread_V: float = 0.2,
    out_dir: str = "data/processed",
) -> IVMetrics:
    """
    Reads an IV CSV and computes:
      - Vset / Vreset from conductance jump detection
      - ION / IOFF at +/-Vread (or closest available)
      - ON/OFF ratio
      - hysteresis loop area (approx.)
    """
    in_path = Path(iv_csv_path)
    run_id = in_path.stem

    df = pd.read_csv(in_path)

    # Normalize expected column names
    # (Your current IV file uses: voltage_V, current_A) :contentReference[oaicite:0]{index=0}
    if "voltage_V" not in df.columns or "current_A" not in df.columns:
        raise ValueError(f"CSV must contain columns voltage_V and current_A. Found: {list(df.columns)}")

    df = df[["voltage_V", "current_A"]].dropna().copy()
    df["voltage_V"] = df["voltage_V"].astype(float)
    df["current_A"] = df["current_A"].astype(float)

    # Hysteresis area (signed integral around curve)
    # This is a simple metric for loop "strength".
    # For non-closed curves, this is still a useful proxy.
    try:
        hysteresis_area = float(np.trapz(df["current_A"].to_numpy(), df["voltage_V"].to_numpy()))
    except Exception:
        hysteresis_area = None

    # ON/OFF at Vread:
    # If your sweep includes both +Vread and -Vread, we can use both.
    # Otherwise we use +Vread only.
    row_pos = _closest_at_v(df, +abs(vread_V))
    row_neg = _closest_at_v(df, -abs(vread_V))

    ion_A = None
    ioff_A = None
    on_off_ratio = None
    notes = ""

    # Heuristic:
    # - If we have both sides, treat higher |I| as ON and lower |I| as OFF.
    # - If only one side is meaningful, use that side with absolute current.
    if row_pos is not None and row_neg is not None:
        i_pos = float(row_pos["current_A"])
        i_neg = float(row_neg["current_A"])
        candidates = [abs(i_pos), abs(i_neg)]
        ion_A = max(candidates)
        ioff_A = min(candidates)
        on_off_ratio = (ion_A / ioff_A) if (ioff_A and ioff_A > 0) else None
    elif row_pos is not None:
        i = abs(float(row_pos["current_A"]))
        ion_A = i
        ioff_A = None
        on_off_ratio = None
        notes += "Only +Vread available; OFF and ON/OFF not computed. "
    else:
        notes += "Vread not found; ON/OFF not computed. "

    # Vset / Vreset:
    b1, b2 = _split_branches(df)
    vset = _detect_switch_voltage(b1, "set")
    vreset = _detect_switch_voltage(b2, "reset")

    if vset is None:
        notes += "No clear SET jump detected. "
    if vreset is None:
        notes += "No clear RESET jump detected. "

    metrics = IVMetrics(
        run_id=run_id,
        vread_V=float(vread_V),
        ion_A=ion_A,
        ioff_A=ioff_A,
        on_off_ratio=on_off_ratio,
        vset_V=vset,
        vreset_V=vreset,
        hysteresis_area=hysteresis_area,
        notes=notes.strip(),
    )

    # Save processed outputs
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # CSV (one-row)
    out_csv = out_path / f"iv_metrics_{run_id}.csv"
    pd.DataFrame([asdict(metrics)]).to_csv(out_csv, index=False)

    # JSON (nice for pipelines)
    out_json = out_path / f"iv_metrics_{run_id}.json"
    out_json.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    return metrics


if __name__ == "__main__":
    # Example manual run:
    # python software/analysis_iv.py data/raw/iv_sweep.csv
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to IV CSV (must contain voltage_V,current_A)")
    ap.add_argument("--vread", type=float, default=0.2)
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    m = compute_iv_metrics(args.csv, vread_V=args.vread, out_dir=args.out)
    print("IV metrics saved for:", m.run_id)
    print(asdict(m))
