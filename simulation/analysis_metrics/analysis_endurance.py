from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_project_root() / pp)


@dataclass
class EnduranceMetrics:
    run_id: str
    n_cycles: int
    g_start_S: Optional[float]
    g_end_S: Optional[float]
    g_min_S: Optional[float]
    g_max_S: Optional[float]
    g_mean_S: Optional[float]
    g_std_S: Optional[float]
    drift_pct: Optional[float]
    slope_S_per_cycle: Optional[float]
    r2_linear: Optional[float]
    window_last_n: int
    cv_last_window: Optional[float]
    lrs_drift_pct: Optional[float] = None
    hrs_drift_pct: Optional[float] = None
    window_ratio_mean: Optional[float] = None
    window_ratio_min: Optional[float] = None
    window_ratio_max: Optional[float] = None
    notes: str = ""


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
        f"Found columns: {list(df.columns)}"
    )


def compute_endurance_metrics(
    pulse_csv_path: str,
    out_dir: str = "data/processed",
    plots_dir: str = "data/plots",
    last_window_frac: float = 0.1,
    min_window: int = 10,
    make_plot: bool = True,
) -> EnduranceMetrics:
    in_path = _resolve_path(pulse_csv_path)
    run_id = in_path.stem
    df = pd.read_csv(in_path).dropna(how="all")

    cycle_col = None
    for c in ("pulse_number", "cycle_number", "cycle", "step"):
        if c in df.columns:
            cycle_col = c
            break
    if cycle_col is None:
        raise ValueError(
            "Pulse CSV must contain a cycle index column like 'pulse_number'. "
            f"Found columns: {list(df.columns)}"
        )
    df = df.copy()
    df[cycle_col] = df[cycle_col].astype(int)
    df, g_col = _get_conductance(df)
    keep_cols = [cycle_col, g_col] + (["phase"] if "phase" in df.columns else [])
    df = df[keep_cols].dropna().sort_values(cycle_col)

    x = df[cycle_col].to_numpy(dtype=float)
    g = df[g_col].to_numpy(dtype=float)
    n = int(len(df))

    notes = ""
    if n < 5:
        notes += "Not enough cycles for robust endurance metrics. "

    g_start = float(g[0]) if n else None
    g_end = float(g[-1]) if n else None
    g_min = float(np.nanmin(g)) if n else None
    g_max = float(np.nanmax(g)) if n else None
    g_mean = float(np.nanmean(g)) if n else None
    g_std = float(np.nanstd(g)) if n else None

    drift_pct = None
    if g_start is not None and np.isfinite(g_start) and abs(g_start) > 1e-18 and g_end is not None:
        drift_pct = float((g_end - g_start) / g_start * 100.0)

    slope = None
    r2 = None
    if n >= 2 and np.isfinite(g).all():
        try:
            m, b = np.polyfit(x, g, 1)
            slope = float(m)
            g_hat = m * x + b
            ss_res = float(np.sum((g - g_hat) ** 2))
            ss_tot = float(np.sum((g - np.mean(g)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
        except Exception:
            notes += "Linear fit failed. "

    last_n = int(max(min_window, round(n * float(last_window_frac)))) if n else 0
    last_n = int(min(last_n, n))
    cv_last = None
    if last_n >= 2:
        gw = g[-last_n:]
        mu = float(np.nanmean(gw))
        sd = float(np.nanstd(gw))
        if np.isfinite(mu) and abs(mu) > 1e-18:
            cv_last = float(sd / abs(mu))
    else:
        notes += "Not enough points for last-window CV. "

    lrs_drift = None
    hrs_drift = None
    w_mu = None
    w_min = None
    w_max = None

    if "phase" in df.columns:
        df_set = df[df["phase"].astype(str).str.upper().str.contains("SET")].copy()
        df_rst = df[df["phase"].astype(str).str.upper().str.contains("RESET")].copy()

        def _drift_pct(series: np.ndarray) -> Optional[float]:
            if series.size < 2:
                return None
            s0 = float(series[0])
            s1 = float(series[-1])
            if not np.isfinite(s0) or abs(s0) <= 1e-18 or not np.isfinite(s1):
                return None
            return float((s1 - s0) / s0 * 100.0)

        if len(df_set) >= 2:
            lrs_drift = _drift_pct(df_set[g_col].to_numpy(dtype=float))
        if len(df_rst) >= 2:
            hrs_drift = _drift_pct(df_rst[g_col].to_numpy(dtype=float))

        if not df_set.empty and not df_rst.empty:
            g_set_by = df_set.groupby(cycle_col)[g_col].mean()
            g_rst_by = df_rst.groupby(cycle_col)[g_col].mean()
            common = g_set_by.index.intersection(g_rst_by.index)
            if len(common) >= 2:
                window = (g_set_by.loc[common] / g_rst_by.loc[common]).replace([np.inf, -np.inf], np.nan).dropna()
                if len(window) >= 1:
                    w_mu = float(window.mean())
                    w_min = float(window.min())
                    w_max = float(window.max())

    metrics = EnduranceMetrics(
        run_id=run_id,
        n_cycles=n,
        g_start_S=g_start,
        g_end_S=g_end,
        g_min_S=g_min,
        g_max_S=g_max,
        g_mean_S=g_mean,
        g_std_S=g_std,
        drift_pct=drift_pct,
        slope_S_per_cycle=slope,
        r2_linear=r2,
        window_last_n=last_n,
        cv_last_window=cv_last,
        lrs_drift_pct=lrs_drift,
        hrs_drift_pct=hrs_drift,
        window_ratio_mean=w_mu,
        window_ratio_min=w_min,
        window_ratio_max=w_max,
        notes=notes.strip(),
    )

    out_p = _resolve_path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / f"endurance_metrics_{run_id}.csv").write_text(
        pd.DataFrame([asdict(metrics)]).to_csv(index=False),
        encoding="utf-8",
    )
    (out_p / f"endurance_metrics_{run_id}.json").write_text(
        json.dumps(asdict(metrics), indent=2),
        encoding="utf-8",
    )

    if make_plot:
        plots_p = _resolve_path(plots_dir)
        plots_p.mkdir(parents=True, exist_ok=True)

        plt.figure()
        if "phase" in df.columns:
            df_set = df[df["phase"].astype(str).str.upper().str.contains("SET")].copy()
            df_rst = df[df["phase"].astype(str).str.upper().str.contains("RESET")].copy()
            if not df_set.empty:
                plt.plot(
                    df_set[cycle_col].to_numpy(dtype=float),
                    df_set[g_col].to_numpy(dtype=float),
                    marker="o",
                    linewidth=1,
                    label="LRS (after SET)",
                )
            if not df_rst.empty:
                plt.plot(
                    df_rst[cycle_col].to_numpy(dtype=float),
                    df_rst[g_col].to_numpy(dtype=float),
                    marker="o",
                    linewidth=1,
                    label="HRS (after RESET)",
                )
            title = "Endurance Cycling" 
            if lrs_drift is not None or hrs_drift is not None:
                parts = []
                if lrs_drift is not None:
                    parts.append(f"LRS drift={lrs_drift:.2f}%")
                if hrs_drift is not None:
                    parts.append(f"HRS drift={hrs_drift:.2f}%")
                title += " (" + ", ".join(parts) + ")"
            plt.title(title)
        else:
            plt.plot(x, g, marker="o", linewidth=1, label="Conductance")
            if n >= 5:
                g_rm = pd.Series(g).rolling(window=5, min_periods=1).mean().to_numpy()
                plt.plot(x, g_rm, linewidth=2, label="Rolling mean (5)")
            title = f"Endurance (drift={drift_pct:.2f}%)" if drift_pct is not None else "Endurance"
            plt.title(title)

        plt.xlabel("Cycle")
        plt.ylabel("Conductance (S)")
        plt.grid(True)
        plt.legend()

        plot_path = plots_p / f"endurance_{run_id}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to pulse/cycle CSV")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--plots", default="data/plots")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--last-window-frac", type=float, default=0.1)
    ap.add_argument("--min-window", type=int, default=10)
    args = ap.parse_args()

    m = compute_endurance_metrics(
        args.csv,
        out_dir=args.out,
        plots_dir=args.plots,
        last_window_frac=args.last_window_frac,
        min_window=args.min_window,
        make_plot=not args.no_plot,
    )
    print(json.dumps(asdict(m), indent=2))
