from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_project_root() / pp)


@dataclass
class RetentionMetrics:
    run_id: str
    t0_s: Optional[float]
    t_end_s: Optional[float]
    g0_S: Optional[float]
    g_end_S: Optional[float]
    retention_ratio: Optional[float]
    t50_s: Optional[float]
    tau_s: Optional[float]
    r2_exp: Optional[float]
    g_inf_S: Optional[float]
    notes: str


def _get_conductance(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "conductance_S" in df.columns:
        df = df.copy()
        df["conductance_S"] = df["conductance_S"].astype(float)
        return df, "conductance_S"
    if "current_A" in df.columns and "voltage_V" in df.columns:
        df = df.copy()
        v = df["voltage_V"].astype(float).to_numpy()
        i = df["current_A"].astype(float).to_numpy()
        eps = 1e-12
        df["conductance_S"] = np.where(np.abs(v) > eps, i / v, np.nan)
        return df, "conductance_S"
    raise ValueError(
        "Retention CSV must contain either conductance_S, or both voltage_V and current_A columns. "
        f"Found columns: {list(df.columns)}"
    )


def _get_time_seconds(df: pd.DataFrame, dt_s: Optional[float]) -> tuple[pd.DataFrame, str, str]:
    note = ""

    if "time_s" in df.columns:
        df = df.copy()
        df["time_s"] = df["time_s"].astype(float)
        return df, "time_s", note

    if "time_ms" in df.columns:
        df = df.copy()
        df["time_s"] = df["time_ms"].astype(float) / 1000.0
        note = "time_s derived from time_ms. "
        return df, "time_s", note

    for c in ("pulse_number", "cycle_number", "cycle", "step"):
        if c in df.columns:
            if dt_s is None:
                dt_s = 0.01
                note += "No time column found; time inferred from cycle index using dt_s=0.01s. "
            else:
                note += f"No time column found; time inferred from cycle index using dt_s={dt_s}s. "
            df = df.copy()
            df["time_s"] = (df[c].astype(float) - float(df[c].astype(float).min())) * float(dt_s)
            return df, "time_s", note

    raise ValueError(
        f"Found columns: {list(df.columns)}"
    )


def _t_at_fraction(t: np.ndarray, g: np.ndarray, frac: float) -> Optional[float]:
    if t.size < 2:
        return None
    g0 = g[0]
    target = frac * g0
    if np.nanmin(g) > target:
        return None
    idx = np.where(g <= target)[0]
    if idx.size == 0:
        return None
    k = int(idx[0])
    if k == 0:
        return float(t[0])
    # Linear interpolation between k-1 and k
    t1, t2 = float(t[k - 1]), float(t[k])
    g1, g2 = float(g[k - 1]), float(g[k])
    if not np.isfinite(g1) or not np.isfinite(g2) or (g2 - g1) == 0:
        return float(t2)
    alpha = (target - g1) / (g2 - g1)
    return float(t1 + alpha * (t2 - t1))


def compute_retention_metrics(
    retention_csv_path: str,
    dt_s: Optional[float] = None,
    out_dir: str = "data/processed",
    plots_dir: str = "data/plots",
    make_plot: bool = True,
) -> RetentionMetrics:
    in_path = Path(retention_csv_path)
    run_id = in_path.stem
    df = pd.read_csv(in_path).dropna(how="all")

    df, g_col = _get_conductance(df)
    df, t_col, note_time = _get_time_seconds(df, dt_s)

    df = df[[t_col, g_col]].dropna().sort_values(t_col)
    t = df[t_col].to_numpy(dtype=float)
    g = df[g_col].to_numpy(dtype=float)
    n = int(len(df))

    notes = note_time
    if n < 5:
        notes += "Not enough points for robust retention fit. "

    t0 = float(t[0]) if n else None
    t_end = float(t[-1]) if n else None
    g0 = float(g[0]) if n else None
    g_end = float(g[-1]) if n else None

    retention_ratio = None
    if g0 is not None and np.isfinite(g0) and abs(g0) > 1e-18 and g_end is not None:
        retention_ratio = float(g_end / g0)

    t50 = None
    if n >= 2 and g0 is not None and np.isfinite(g0):
        t50 = _t_at_fraction(t, g, 0.5)

    # Exponential fit with offset g_inf
    tau = None
    r2 = None
    g_inf = None
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
            notes += "Not enough positive (g - g_inf) points for exponential fit. "
    except Exception:
        notes += "Exponential fit failed. "

    metrics = RetentionMetrics(
        run_id=run_id,
        t0_s=t0,
        t_end_s=t_end,
        g0_S=g0,
        g_end_S=g_end,
        retention_ratio=retention_ratio,
        t50_s=t50,
        tau_s=tau,
        r2_exp=r2,
        g_inf_S=g_inf,
        notes=notes.strip(),
    )

    out_p = _resolve_path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / f"retention_metrics_{run_id}.csv").write_text(
        pd.DataFrame([asdict(metrics)]).to_csv(index=False),
        encoding="utf-8",
    )
    (out_p / f"retention_metrics_{run_id}.json").write_text(
        json.dumps(asdict(metrics), indent=2),
        encoding="utf-8",
    )

    if make_plot:
        plots_p = _resolve_path(plots_dir)
        plots_p.mkdir(parents=True, exist_ok=True)

        plt.figure()
        # Use a log x-axis when time spans multiple decades
        if n >= 2 and np.nanmin(t) > 0 and (np.nanmax(t) / np.nanmin(t) >= 100):
            plt.semilogx(t, g, marker="o", linewidth=1)
            plt.xlabel("Time (s) [log]")
        else:
            plt.plot(t, g, marker="o", linewidth=1)
            plt.xlabel("Time (s)")

        title_bits = []
        if retention_ratio is not None:
            title_bits.append(f"G_end/G0={retention_ratio:.3f}")
        if tau is not None:
            title_bits.append(f"tau={tau:.2f}s")
        title = "Retention" + (" (" + ", ".join(title_bits) + ")" if title_bits else "")
        plt.title(title)
        plt.ylabel("Conductance (S)")
        plt.grid(True)

        plot_path = plots_p / f"retention_{run_id}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to retention CSV")
    ap.add_argument("--dt", type=float, default=None, help="Seconds per cycle (if inferring time)")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--plots", default="data/plots")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    m = compute_retention_metrics(
        args.csv,
        dt_s=args.dt,
        out_dir=args.out,
        plots_dir=args.plots,
        make_plot=not args.no_plot,
    )
    print(json.dumps(asdict(m), indent=2))
