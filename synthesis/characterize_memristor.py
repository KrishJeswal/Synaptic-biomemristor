from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_SIMULATION_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SIMULATION_DIR.parents[0]

def _resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_PROJECT_ROOT / pp)

def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None

V_COL_CANDIDATES = [
    "voltage_V",
    "V",
    "voltage",
    "Voltage",
    "Voltage (V)",
    "device_voltage_V",
]

I_COL_CANDIDATES = [
    "current_A",
    "I_A",
    "I",
    "current",
    "Current",
    "Current (A)",
]

TIME_COL_CANDIDATES = [
    "time_s",
    "time_ms",
    "Time (s)",
    "Time (ms)",
]

PULSE_COL_CANDIDATES = [
    "pulse_number",
    "cycle_number",
    "cycle",
    "step",
    "pulse",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_map: Dict[str, str] = {}

    for c in V_COL_CANDIDATES:
        if c in df.columns:
            col_map[c] = "voltage_V"
            break

    for c in I_COL_CANDIDATES:
        if c in df.columns:
            col_map[c] = "current_A"
            break

    if "time_s" not in df.columns:
        if "time_ms" in df.columns:
            pass
        else:
            for c in TIME_COL_CANDIDATES:
                if c in df.columns:
                    if "ms" in c.lower():
                        col_map[c] = "time_ms"
                    else:
                        col_map[c] = "time_s"
                    break

    for c in PULSE_COL_CANDIDATES:
        if c in df.columns:
            col_map[c] = "pulse_number"
            break

    if "conductance_S" in df.columns:
        col_map["conductance_S"] = "conductance_S"

    if col_map:
        df = df.rename(columns=col_map)

    for c in ("voltage_V", "current_A", "conductance_S"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "time_ms" in df.columns and "time_s" not in df.columns:
        df["time_s"] = pd.to_numeric(df["time_ms"], errors="coerce") / 1000.0

    if "pulse_number" in df.columns:
        df["pulse_number"] = pd.to_numeric(df["pulse_number"], errors="coerce")

    if "conductance_S" not in df.columns and "voltage_V" in df.columns and "current_A" in df.columns:
        v = df["voltage_V"].to_numpy(dtype=float)
        i = df["current_A"].to_numpy(dtype=float)
        eps = 1e-12
        g = np.full_like(i, np.nan, dtype=float)
        m = np.abs(v) > eps
        g[m] = i[m] / v[m]
        df["conductance_S"] = g

    return df

def _read_csv_flexible(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    return _normalize_columns(df)

@dataclass
class IVMetricsSimple:
    source_csv: str
    vread_V: float
    ion_A: Optional[float]
    ioff_A: Optional[float]
    on_off_ratio: Optional[float]
    ron_Ohm: Optional[float]
    roff_Ohm: Optional[float]
    vset_V: Optional[float]
    vreset_V: Optional[float]
    hysteresis_area: Optional[float]
    notes: str

def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    m = np.abs(b) > eps
    out[m] = a[m] / b[m]
    return out

def _split_branches(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or len(df) < 3:
        return df.copy(), df.copy()

    v = df["voltage_V"].to_numpy(dtype=float)
    dv = np.diff(v)
    s = np.sign(dv)
    nz = np.where(s != 0)[0]
    if nz.size >= 2:
        s_nz = s[nz]
        ch = np.where(np.diff(s_nz) != 0)[0]
        if ch.size > 0:
            turn_dv_idx = int(nz[ch[0]])
            turn_idx = turn_dv_idx + 1
            b1 = df.iloc[: turn_idx + 1].copy()
            b2 = df.iloc[turn_idx:].copy()
            return b1, b2

    absmax_idx = df["voltage_V"].abs().idxmax()
    i = df.index.get_loc(absmax_idx)
    b1 = df.iloc[: i + 1].copy()
    b2 = df.iloc[i:].copy()
    return b1, b2

def _interp_current_vs_v(branch: pd.DataFrame, v_grid: np.ndarray) -> np.ndarray:
    if branch.empty:
        return np.full_like(v_grid, np.nan, dtype=float)
    v = branch["voltage_V"].to_numpy(dtype=float)
    i = branch["current_A"].to_numpy(dtype=float)
    order = np.argsort(v)
    v_s = v[order]
    i_s = i[order]
    v_u, idx = np.unique(v_s, return_index=True)
    i_u = i_s[idx]
    if v_u.size < 2:
        return np.full_like(v_grid, np.nan, dtype=float)
    return np.interp(v_grid, v_u, i_u)

def _hysteresis_area(b1: pd.DataFrame, b2: pd.DataFrame, n_grid: int = 400) -> Optional[float]:
    if b1.empty or b2.empty:
        return None
    v1 = b1["voltage_V"].to_numpy(dtype=float)
    v2 = b2["voltage_V"].to_numpy(dtype=float)
    vmin = max(np.nanmin(v1), np.nanmin(v2))
    vmax = min(np.nanmax(v1), np.nanmax(v2))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None
    vg = np.linspace(vmin, vmax, int(n_grid))
    i1g = _interp_current_vs_v(b1, vg)
    i2g = _interp_current_vs_v(b2, vg)
    return float(np.trapz(np.abs(i1g - i2g), vg))

def _closest_current_at_v(df: pd.DataFrame, v_target: float) -> Optional[float]:
    if df.empty:
        return None
    idx = (df["voltage_V"] - float(v_target)).abs().idxmin()
    try:
        return float(df.loc[idx, "current_A"])
    except Exception:
        return None

def _detect_switch_voltage(branch: pd.DataFrame, mode: str) -> Optional[float]:
    if branch is None or len(branch) < 5:
        return None

    v = branch["voltage_V"].to_numpy(dtype=float)
    i = branch["current_A"].to_numpy(dtype=float)
    g = _safe_div(i, v)  # S
    g_s = pd.Series(g).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
    dg = np.diff(g_s)
    if dg.size == 0:
        return None

    med = np.nanmedian(dg)
    mad = np.nanmedian(np.abs(dg - med))
    noise = mad if mad > 0 else np.nanstd(dg)
    if not np.isfinite(noise) or noise == 0:
        noise = 1e-12

    k = 8.0
    if mode == "set":
        idx = int(np.nanargmax(dg))
        if dg[idx] < k * noise:
            return None
        return float(v[idx + 1])
    if mode == "reset":
        idx = int(np.nanargmin(dg))
        if dg[idx] > -k * noise:
            return None
        return float(v[idx + 1])
    raise ValueError("mode must be 'set' or 'reset'")

def compute_iv_metrics_from_csv(
    iv_csv_path: str | Path,
    vread_V: float = 0.2,
    make_plot: bool = True,
    plots_dir: str | Path = "data/plots",
) -> IVMetricsSimple:
    """Compute basic I–V characterization metrics.

    Supports both:
      1) Hardware-style logs: columns like V / I_A
      2) Simulated logs (memristor_simulated_run.py input): voltage_V/current_A
         plus optional cycle/direction/index_in_sweep.

    For simulated logs with multiple cycles, we automatically select the
    latest (max) cycle.
    """

    df = _read_csv_flexible(iv_csv_path)
    notes = ""

    if "voltage_V" not in df.columns or "current_A" not in df.columns:
        raise ValueError(
            f"IV CSV must contain voltage/current columns. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["voltage_V", "current_A"]).copy()

    # --- simulated dataset support: pick one cycle ---
    if "cycle" in df.columns:
        try:
            cyc = pd.to_numeric(df["cycle"], errors="coerce").dropna().astype(int)
            if not cyc.empty:
                chosen_cycle = int(cyc.max())
                df = df[cyc.reindex(df.index).fillna(chosen_cycle).astype(int) == chosen_cycle].copy()
                notes += f"Selected cycle={chosen_cycle} from simulated IV log. "
        except Exception:
            pass

    # Prefer the provided sweep order if available
    if "time_s" in df.columns:
        df = df.sort_values("time_s", kind="stable")
    elif "index_in_sweep" in df.columns:
        df = df.sort_values(["index_in_sweep"], kind="stable")
    else:
        df = df.sort_index().reset_index(drop=True)

    # --- branch split ---
    b1: pd.DataFrame
    b2: pd.DataFrame

    if "direction" in df.columns:
        d = df["direction"].astype(str).str.lower()
        fwd = df[d.str.contains("forward")].copy()
        rev = df[d.str.contains("reverse")].copy()

        if not fwd.empty and not rev.empty:
            if "index_in_sweep" in df.columns:
                fwd = fwd.sort_values(["index_in_sweep"], kind="stable")
                rev = rev.sort_values(["index_in_sweep"], kind="stable")
            elif "time_s" in df.columns:
                fwd = fwd.sort_values(["time_s"], kind="stable")
                rev = rev.sort_values(["time_s"], kind="stable")

            b1 = fwd[["voltage_V", "current_A"]].copy()
            b2 = rev[["voltage_V", "current_A"]].copy()
        else:
            b1, b2 = _split_branches(df[["voltage_V", "current_A"]].copy())
    else:
        b1, b2 = _split_branches(df[["voltage_V", "current_A"]].copy())

    area = _hysteresis_area(b1, b2)

    # ON/OFF via the 2 branches around Vread
    i1 = _closest_current_at_v(b1, abs(vread_V))
    i2 = _closest_current_at_v(b2, abs(vread_V))

    ion = None
    ioff = None
    ratio = None
    ron = None
    roff = None

    candidates = []
    if i1 is not None:
        candidates.append(abs(float(i1)))
    if i2 is not None:
        candidates.append(abs(float(i2)))

    if len(candidates) >= 2:
        ion = float(max(candidates))
        ioff = float(min(candidates))
        if ioff > 0:
            ratio = float(ion / ioff)
        else:
            notes += "IOFF was ~0; ON/OFF ratio undefined. "
    elif len(candidates) == 1:
        ion = float(candidates[0])
        notes += "Only one branch near Vread; IOFF not available. "
    else:
        notes += "Could not find a point near Vread. "

    if ion is not None and ion > 0:
        ron = float(abs(vread_V) / ion)
    if ioff is not None and ioff > 0:
        roff = float(abs(vread_V) / ioff)

    # Switch thresholds
    vset_branch = b1
    vreset_branch = b2

    if df["voltage_V"].min() < 0 and df["voltage_V"].max() > 0:
        b1_pos = b1[b1["voltage_V"] >= 0]
        b2_neg = b2[b2["voltage_V"] <= 0]
        if len(b1_pos) >= 5:
            vset_branch = b1_pos
        if len(b2_neg) >= 5:
            vreset_branch = b2_neg

    vset = _detect_switch_voltage(vset_branch, "set")
    vreset = _detect_switch_voltage(vreset_branch, "reset")

    if vset is None:
        notes += "No clear SET jump detected. "
    if vreset is None:
        notes += "No clear RESET jump detected. "

    if make_plot:
        plots_p = _resolve_path(plots_dir)
        plots_p.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(b1["voltage_V"], b1["current_A"], marker="o", linewidth=1, label="Branch 1")
        plt.plot(b2["voltage_V"], b2["current_A"], marker="o", linewidth=1, label="Branch 2")

        title_bits = []
        if ratio is not None:
            title_bits.append(f"ON/OFF={ratio:.2g}@{vread_V:.2g}V")
        if vset is not None:
            title_bits.append(f"Vset={vset:.2g}V")
        if vreset is not None:
            title_bits.append(f"Vreset={vreset:.2g}V")
        if area is not None:
            title_bits.append(f"Area={area:.2e}")

        title = "I–V Characterization" + (" (" + ", ".join(title_bits) + ")" if title_bits else "")
        plt.title(title)
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.grid(True)
        plt.legend()

        out_plot = plots_p / f"iv_characterization_{Path(iv_csv_path).stem}.png"
        plt.savefig(out_plot, bbox_inches="tight")
        plt.close()

    return IVMetricsSimple(
        source_csv=str(Path(iv_csv_path)),
        vread_V=float(vread_V),
        ion_A=ion,
        ioff_A=ioff,
        on_off_ratio=ratio,
        ron_Ohm=ron,
        roff_Ohm=roff,
        vset_V=vset,
        vreset_V=vreset,
        hysteresis_area=area,
        notes=notes.strip(),
    )


@dataclass
class EnduranceMetricsSimple:
    source_csv: str
    n_cycles: int
    g_lrs_mean_S: Optional[float]
    g_hrs_mean_S: Optional[float]
    on_off_ratio_mean: Optional[float]
    drift_pct_overall: Optional[float]
    cv_last_window: Optional[float]
    failure_cycle: Optional[int]
    notes: str

def compute_endurance_metrics_from_csv(
    endurance_csv_path: str | Path,
    ratio_fail_threshold: float = 10.0,
    last_window_frac: float = 0.1,
    min_window: int = 10,
    make_plot: bool = True,
    plots_dir: str | Path = "data/plots",
) -> EnduranceMetricsSimple:
    """Compute endurance / pulse stability metrics.

    Works on:
      - Hardware pulse CSVs (typically V/I per pulse)
      - Simulated pulse-train logs (sim_pulse_train.csv) that include
        pulse_number + phase (pulse_high/read_low) + time_s.

    For the simulated pulse-train format, we automatically filter to
    phase='pulse_high' to avoid duplicate pulse_number rows.
    """

    df = _read_csv_flexible(endurance_csv_path)
    notes = ""

    if df.empty:
        raise ValueError("Endurance CSV is empty")

    # Simulated pulse-train logs contain both write pulses + readbacks.
    # For endurance/stability, default to write pulses (pulse_high).
    if "phase" in df.columns:
        phase_l = df["phase"].astype(str).str.lower()
        if phase_l.str.contains("pulse_high").any() and not phase_l.str.contains("set|reset").any():
            df = df[phase_l.str.contains("pulse_high")].copy()
            notes += "Filtered to phase='pulse_high' for endurance metrics (simulated pulse-train). "

    if df.empty:
        raise ValueError("Endurance CSV became empty after filtering")

    if "pulse_number" not in df.columns:
        df = df.copy()
        df["pulse_number"] = np.arange(1, len(df) + 1, dtype=float)
        notes += "No cycle column found; used row index as pulse_number. "

    if "conductance_S" not in df.columns:
        if "voltage_V" in df.columns and "current_A" in df.columns:
            v = df["voltage_V"].to_numpy(dtype=float)
            i = df["current_A"].to_numpy(dtype=float)
            eps = 1e-12
            g = np.full_like(i, np.nan, dtype=float)
            m = np.abs(v) > eps
            g[m] = i[m] / v[m]
            df["conductance_S"] = g
        else:
            raise ValueError(
                "Endurance CSV must have conductance_S or both voltage/current columns. "
                f"Found: {list(df.columns)}"
            )

    df = df.dropna(subset=["pulse_number", "conductance_S"]).copy()
    df["pulse_number"] = pd.to_numeric(df["pulse_number"], errors="coerce").astype(int)
    df = df.sort_values("pulse_number")

    n = int(len(df))
    if n == 0:
        raise ValueError("No valid endurance points found after cleaning")

    g_lrs = None
    g_hrs = None
    ratio_mean = None
    failure_cycle = None

    # Phase-based SET/RESET split (hardware endurance cycling)
    if "phase" in df.columns:
        phase_s = df["phase"].astype(str).str.upper()
        lrs_df = df[phase_s.str.contains("SET")]
        hrs_df = df[phase_s.str.contains("RESET")]

        if not lrs_df.empty:
            g_lrs = float(lrs_df["conductance_S"].mean())
        if not hrs_df.empty:
            g_hrs = float(hrs_df["conductance_S"].mean())
        if g_lrs is not None and g_hrs is not None and abs(g_hrs) > 1e-18:
            ratio_mean = float(abs(g_lrs) / abs(g_hrs))

        if not lrs_df.empty and not hrs_df.empty:
            lrs_by = lrs_df.groupby("pulse_number")["conductance_S"].mean()
            hrs_by = hrs_df.groupby("pulse_number")["conductance_S"].mean()
            common = lrs_by.index.intersection(hrs_by.index)
            if len(common) >= 3:
                ratios = (
                    lrs_by.loc[common].abs() / hrs_by.loc[common].abs()
                ).replace([np.inf, -np.inf], np.nan)
                ratios = ratios.dropna()
                if not ratios.empty:
                    bad = ratios[ratios < float(ratio_fail_threshold)]
                    if not bad.empty:
                        failure_cycle = int(bad.index.min())

        if g_lrs is None and g_hrs is None:
            notes += "Phase column present but no SET/RESET rows; using early/late window estimate. "
    else:
        notes += "No phase column; using early/late window estimate for HRS/LRS. "

    g = df["conductance_S"].to_numpy(dtype=float)

    # Fallback: estimate HRS/LRS from early vs late points
    if (g_lrs is None or g_hrs is None) and n >= 6:
        w_est = max(3, int(round(0.1 * n)))
        g_hrs_est = float(np.nanmean(np.abs(g[:w_est])))
        g_lrs_est = float(np.nanmean(np.abs(g[-w_est:])))
        if np.isfinite(g_hrs_est) and g_hrs_est > 1e-18 and np.isfinite(g_lrs_est):
            if g_hrs is None:
                g_hrs = g_hrs_est
            if g_lrs is None:
                g_lrs = g_lrs_est
            if ratio_mean is None:
                ratio_mean = float(g_lrs_est / g_hrs_est)
            notes += f"Estimated HRS/LRS using first/last {w_est} points. "

    g0 = float(g[0])
    g_end = float(g[-1])
    drift_pct = None
    if np.isfinite(g0) and abs(g0) > 1e-18 and np.isfinite(g_end):
        drift_pct = float((g_end - g0) / g0 * 100.0)

    last_n = int(max(min_window, round(n * float(last_window_frac))))
    last_n = int(min(last_n, n))
    cv_last = None
    if last_n >= 2:
        gw = g[-last_n:]
        mu = float(np.nanmean(gw))
        sd = float(np.nanstd(gw))
        if np.isfinite(mu) and abs(mu) > 1e-18:
            cv_last = float(sd / abs(mu))

    if make_plot:
        plots_p = _resolve_path(plots_dir)
        plots_p.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(df["pulse_number"], df["conductance_S"], marker="o", linewidth=1, label="Conductance")
        plt.title("Endurance / Pulse Stability")
        plt.xlabel("Cycle / Pulse")
        plt.ylabel("Conductance (S)")
        plt.grid(True)
        plt.legend()
        out_plot = plots_p / f"endurance_characterization_{Path(endurance_csv_path).stem}.png"
        plt.savefig(out_plot, bbox_inches="tight")
        plt.close()

    return EnduranceMetricsSimple(
        source_csv=str(Path(endurance_csv_path)),
        n_cycles=n,
        g_lrs_mean_S=g_lrs,
        g_hrs_mean_S=g_hrs,
        on_off_ratio_mean=ratio_mean,
        drift_pct_overall=drift_pct,
        cv_last_window=cv_last,
        failure_cycle=failure_cycle,
        notes=notes.strip(),
    )


@dataclass
class RetentionMetricsSimple:
    source_csv: str
    t0_s: Optional[float]
    t_end_s: Optional[float]
    g0_S: Optional[float]
    g_end_S: Optional[float]
    retention_ratio: Optional[float]
    t50_s: Optional[float]
    tau_s: Optional[float]
    r2_exp: Optional[float]
    notes: str

def _t_at_fraction(t: np.ndarray, g: np.ndarray, frac: float) -> Optional[float]:
    if t.size < 2:
        return None
    g0 = float(g[0])
    if not np.isfinite(g0):
        return None
    target = frac * g0
    if np.nanmin(g) > target:
        return None
    idx = np.where(g <= target)[0]
    if idx.size == 0:
        return None
    k = int(idx[0])
    if k == 0:
        return float(t[0])
    t1, t2 = float(t[k - 1]), float(t[k])
    g1, g2 = float(g[k - 1]), float(g[k])
    if not np.isfinite(g1) or not np.isfinite(g2) or (g2 - g1) == 0:
        return float(t2)
    alpha = (target - g1) / (g2 - g1)
    return float(t1 + alpha * (t2 - t1))

def compute_retention_metrics_from_csv(
    retention_csv_path: str | Path,
    dt_s_if_missing: Optional[float] = None,
    make_plot: bool = True,
    plots_dir: str | Path = "data/plots",
) -> RetentionMetricsSimple:
    """Compute retention metrics.

    Works on:
      - Dedicated retention logs (conductance vs time)
      - Simulated pulse-train logs (sim_pulse_train.csv): we automatically
        filter to phase='read_low' if present.
      - Hardware pulse CSVs without time: we infer time from row index.
    """

    df = _read_csv_flexible(retention_csv_path)
    notes = ""

    if df.empty:
        raise ValueError("Retention CSV is empty")

    # Simulated pulse-train has readback points labeled 'read_low'
    if "phase" in df.columns:
        phase_l = df["phase"].astype(str).str.lower()
        if phase_l.str.contains("read_low").any():
            df = df[phase_l.str.contains("read_low")].copy()
            notes += "Filtered to phase='read_low' for retention metrics (simulated pulse-train). "

    if df.empty:
        raise ValueError("Retention CSV became empty after filtering")

    if "conductance_S" not in df.columns:
        raise ValueError(
            "Retention CSV must have conductance_S or both voltage/current columns. "
            f"Found: {list(df.columns)}"
        )

    # If time is missing, infer from pulse_number or row index
    if "time_s" not in df.columns:
        if "pulse_number" not in df.columns:
            df = df.copy()
            df["pulse_number"] = np.arange(1, len(df) + 1, dtype=float)
            notes += "No time/pulse columns; used row index as pulse_number. "

        if dt_s_if_missing is None:
            dt_s_if_missing = 1.0
            notes += "No time column; inferred time from pulse_number using dt=1.0s. "
        else:
            notes += f"No time column; inferred time from pulse_number using dt={dt_s_if_missing}s. "

        p = pd.to_numeric(df["pulse_number"], errors="coerce").astype(float)
        df["time_s"] = (p - float(p.min())) * float(dt_s_if_missing)

    df = df.dropna(subset=["time_s", "conductance_S"]).copy().sort_values("time_s")
    t = df["time_s"].to_numpy(dtype=float)
    g = df["conductance_S"].to_numpy(dtype=float)

    n = int(len(df))
    if n < 2:
        raise ValueError("Not enough retention points")

    t0 = float(t[0])
    t_end = float(t[-1])
    g0 = float(g[0])
    g_end = float(g[-1])

    retention_ratio = None
    if np.isfinite(g0) and abs(g0) > 1e-18 and np.isfinite(g_end):
        retention_ratio = float(g_end / g0)

    t50 = _t_at_fraction(t, g, 0.5)

    tau = None
    r2 = None
    try:
        g_inf = float(np.nanmin(g))
        y = g - g_inf
        mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
        if np.sum(mask) >= 3:
            tt = t[mask]
            yy = y[mask]
            ln = np.log(yy)
            slope, intercept = np.polyfit(tt, ln, 1)
            if slope < 0:
                tau = float(-1.0 / slope)
                ln_hat = slope * tt + intercept
                ss_res = float(np.sum((ln - ln_hat) ** 2))
                ss_tot = float(np.sum((ln - np.mean(ln)) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
            else:
                notes += "Exponential fit slope was non-negative; tau not computed. "
        else:
            notes += "Not enough positive points for exponential fit. "
    except Exception:
        notes += "Exponential fit failed. "

    if make_plot:
        plots_p = _resolve_path(plots_dir)
        plots_p.mkdir(parents=True, exist_ok=True)
        plt.figure()
        if np.nanmin(t) > 0 and (np.nanmax(t) / np.nanmin(t) >= 100):
            plt.semilogx(t, g, marker="o", linewidth=1)
            plt.xlabel("Time (s) [log]")
        else:
            plt.plot(t, g, marker="o", linewidth=1)
            plt.xlabel("Time (s)")
        plt.ylabel("Conductance (S)")

        title_bits = []
        if retention_ratio is not None:
            title_bits.append(f"G_end/G0={retention_ratio:.3f}")
        if t50 is not None:
            title_bits.append(f"t50={t50:.2f}s")
        if tau is not None:
            title_bits.append(f"tau={tau:.2f}s")

        plt.title("Retention" + (" (" + ", ".join(title_bits) + ")" if title_bits else ""))
        plt.grid(True)
        out_plot = plots_p / f"retention_characterization_{Path(retention_csv_path).stem}.png"
        plt.savefig(out_plot, bbox_inches="tight")
        plt.close()

    return RetentionMetricsSimple(
        source_csv=str(Path(retention_csv_path)),
        t0_s=t0,
        t_end_s=t_end,
        g0_S=g0,
        g_end_S=g_end,
        retention_ratio=retention_ratio,
        t50_s=t50,
        tau_s=tau,
        r2_exp=r2,
        notes=notes.strip(),
    )


@dataclass
class CharacterizationSummary:
    iv: Optional[IVMetricsSimple]
    endurance: Optional[EnduranceMetricsSimple]
    retention: Optional[RetentionMetricsSimple]
    notes: str

def _guess_csvs_from_dir(input_dir: Path) -> Dict[str, Path]:
    """Auto-detect likely IV / endurance / retention CSVs from a folder.

    Updated to support outputs from memristor_simulated_run.py.

    Priority rules (when multiple CSVs exist):
      - Prefer simulated outputs (filenames containing 'sim_') over hardware logs
      - Prefer hysteresis I–V logs for richer IV metrics
      - Use pulse_train as both endurance + retention fallback when needed
    """

    out: Dict[str, Path] = {}
    csvs = sorted([p for p in input_dir.glob('*.csv') if p.is_file()])
    if not csvs:
        return out

    def first_match(preds):
        for pred in preds:
            for p in csvs:
                name = p.name.lower()
                if pred(name):
                    return p
        return None

    # IV
    iv_p = first_match([
        lambda n: ('sim' in n) and ('iv' in n) and ('hysteresis' in n),
        lambda n: ('sim' in n) and ('iv' in n) and ('sweep' in n),
        lambda n: ('iv' in n) and ('hysteresis' in n),
        lambda n: ('iv' in n) and ('sweep' in n),
        lambda n: ('iv' in n),
    ])
    if iv_p is not None:
        out['iv'] = iv_p

    # Endurance
    end_p = first_match([
        lambda n: ('sim' in n) and ('pulse' in n) and ('train' in n),
        lambda n: ('endurance' in n) or ('cycle' in n),
        lambda n: ('pulse' in n) and ('train' in n),
    ])
    if end_p is not None:
        out['endurance'] = end_p

    # Retention
    ret_p = first_match([
        lambda n: 'retention' in n,
        lambda n: ('sim' in n) and ('pulse' in n) and ('train' in n),
        lambda n: ('pulse' in n) and ('train' in n),
    ])
    if ret_p is not None:
        out['retention'] = ret_p

    # If there's no dedicated retention file but we have endurance, reuse it
    if 'retention' not in out and 'endurance' in out:
        out['retention'] = out['endurance']

    return out


def write_summary_files(
    summary: CharacterizationSummary,
    out_dir: str | Path = "data/processed",
    prefix: str = "characterization_summary",
) -> Tuple[str, str]:
    out_p = _resolve_path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    json_path = out_p / f"{prefix}.json"
    json_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    flat: Dict[str, object] = {}
    if summary.iv is not None:
        for k, v in asdict(summary.iv).items():
            flat[f"iv_{k}"] = v
    if summary.endurance is not None:
        for k, v in asdict(summary.endurance).items():
            flat[f"endurance_{k}"] = v
    if summary.retention is not None:
        for k, v in asdict(summary.retention).items():
            flat[f"retention_{k}"] = v
    flat["notes"] = summary.notes

    csv_path = out_p / f"{prefix}.csv"
    pd.DataFrame([flat]).to_csv(csv_path, index=False)
    return str(json_path), str(csv_path)

def main() -> int:
    ap = argparse.ArgumentParser(description="Compute memristor characterization metrics from CSV logs")
    ap.add_argument("--input", default=None, help="Folder containing CSVs (e.g., data/final values)")
    ap.add_argument("--iv", default=None, help="Path to IV sweep CSV")
    ap.add_argument("--endurance", default=None, help="Path to endurance cycling CSV")
    ap.add_argument("--pulse", default=None, help="Path to pulse-train CSV (fallback for endurance)")
    ap.add_argument("--retention", default=None, help="Path to retention CSV")
    ap.add_argument("--vread", type=float, default=0.2, help="Read voltage for ON/OFF ratio")
    ap.add_argument("--ratio-fail", type=float, default=10.0, help="Failure threshold for endurance ON/OFF")
    ap.add_argument("--out", default="data/processed", help="Output directory for summary JSON/CSV")
    ap.add_argument("--plots", default="data/plots", help="Output directory for plots")
    ap.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    args = ap.parse_args()

    iv_path = args.iv
    endurance_path = args.endurance
    retention_path = args.retention

    if args.pulse and not endurance_path:
        endurance_path = args.pulse

    notes = ""
    if args.input:
        guess = _guess_csvs_from_dir(_resolve_path(args.input))
        iv_path = iv_path or (str(guess.get("iv")) if guess.get("iv") else None)
        endurance_path = endurance_path or (str(guess.get("endurance")) if guess.get("endurance") else None)
        retention_path = retention_path or (str(guess.get("retention")) if guess.get("retention") else None)
        notes += f"Auto-detected from {args.input}: { {k: str(v.name) for k, v in guess.items()} }. "

    iv_metrics = None
    endurance_metrics = None
    retention_metrics = None

    make_plots = not args.no_plots

    if iv_path:
        try:
            iv_metrics = compute_iv_metrics_from_csv(iv_path, vread_V=args.vread, make_plot=make_plots, plots_dir=args.plots)
        except Exception as e:
            notes += f"IV metrics failed: {e}. "
    else:
        notes += "No IV CSV provided/found. "

    if endurance_path:
        try:
            endurance_metrics = compute_endurance_metrics_from_csv(
                endurance_path,
                ratio_fail_threshold=args.ratio_fail,
                make_plot=make_plots,
                plots_dir=args.plots,
            )
        except Exception as e:
            notes += f"Endurance metrics failed: {e}. "
    else:
        notes += "No endurance/pulse CSV provided/found. "

    if retention_path:
        try:
            retention_metrics = compute_retention_metrics_from_csv(retention_path, make_plot=make_plots, plots_dir=args.plots)
        except Exception as e:
            notes += f"Retention metrics failed: {e}. "
    else:
        notes += "No retention CSV provided/found. "

    summary = CharacterizationSummary(
        iv=iv_metrics,
        endurance=endurance_metrics,
        retention=retention_metrics,
        notes=notes.strip(),
    )

    json_path, csv_path = write_summary_files(summary, out_dir=args.out)
    print("Saved summary:")
    print("  JSON:", json_path)
    print("  CSV :", csv_path)
    if iv_metrics:
        print("IV:", asdict(iv_metrics))
    if endurance_metrics:
        print("Endurance:", asdict(endurance_metrics))
    if retention_metrics:
        print("Retention:", asdict(retention_metrics))
    if summary.notes:
        print("Notes:", summary.notes)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
