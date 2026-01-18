from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_project_root() / pp)


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


def _compute_hysteresis_area_from_branches(
    b1: pd.DataFrame,
    b2: pd.DataFrame,
    n_grid: int = 400,
) -> tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if b1.empty or b2.empty:
        return None, None, None, None

    v1 = b1["voltage_V"].to_numpy(dtype=float)
    v2 = b2["voltage_V"].to_numpy(dtype=float)

    vmin = max(np.nanmin(v1), np.nanmin(v2))
    vmax = min(np.nanmax(v1), np.nanmax(v2))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None, None, None, None

    v_grid = np.linspace(vmin, vmax, int(n_grid))
    i1g = _interp_current_vs_v(b1, v_grid)
    i2g = _interp_current_vs_v(b2, v_grid)

    area = float(np.trapz(np.abs(i1g - i2g), v_grid))
    return area, v_grid, i1g, i2g


def _compute_hysteresis_area(
    df: pd.DataFrame,
) -> tuple[Optional[float], pd.DataFrame, pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if df.empty or len(df) < 5:
        return None, df.copy(), df.copy(), None, None, None

    v_all = df["voltage_V"].to_numpy(dtype=float)
    dv = np.diff(v_all)
    if np.all(dv >= 0) or np.all(dv <= 0):
        v_s = np.unique(np.sort(v_all))
        if v_s.size < 2:
            return 0.0, df.copy(), df.copy(), None, None, None
        i_s = _interp_current_vs_v(df, v_s)
        return 0.0, df.copy(), df.copy(), v_s, i_s, i_s

    b1, b2 = _split_branches(df)
    area, v_grid, i1g, i2g = _compute_hysteresis_area_from_branches(b1, b2)
    if area is None or v_grid is None or i1g is None or i2g is None:
        return None, b1, b2, None, None, None
    return area, b1, b2, v_grid, i1g, i2g


def plot_hysteresis(
    iv_csv_path: str,
    out_path: str | None = None,
    title: str | None = None,
    show: bool = False,
) -> Optional[str]:
    in_path = _resolve_path(iv_csv_path)
    df = pd.read_csv(in_path)
    if "voltage_V" not in df.columns or "current_A" not in df.columns:
        raise ValueError(f"CSV must contain columns voltage_V and current_A. Found: {list(df.columns)}")
    df = df.dropna(subset=["voltage_V", "current_A"]).copy()
    df["voltage_V"] = df["voltage_V"].astype(float)
    df["current_A"] = df["current_A"].astype(float)

    if "sweep" in df.columns and set(df["sweep"].astype(str).unique()) >= {"fwd", "rev"}:
        b1 = df[df["sweep"].astype(str) == "fwd"][["voltage_V", "current_A"]].copy()
        b2 = df[df["sweep"].astype(str) == "rev"][["voltage_V", "current_A"]].copy()
        area, v_grid, i1g, i2g = _compute_hysteresis_area_from_branches(b1, b2)
    else:
        core = df[["voltage_V", "current_A"]].copy()
        area, b1, b2, v_grid, i1g, i2g = _compute_hysteresis_area(core)

    plt.figure()
    plt.plot(b1["voltage_V"], b1["current_A"], marker="o", linewidth=1, label="Branch 1")
    plt.plot(b2["voltage_V"], b2["current_A"], marker="o", linewidth=1, label="Branch 2")
    if v_grid is not None and i1g is not None and i2g is not None:
        plt.fill_between(v_grid, i1g, i2g, alpha=0.2, label="Hysteresis area")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    if title is None:
        title = f"I–V Hysteresis" + (f" (Area={area:.3e})" if area is not None else "")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if out_path is None:
        out_path = str(_resolve_path(f"data/plots/hysteresis_{in_path.stem}.png"))

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return str(out_p)

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
            turn_idx = turn_dv_idx + 1  # split after this point
            b1 = df.iloc[: turn_idx + 1].copy()
            b2 = df.iloc[turn_idx:].copy()
            return b1, b2

    absmax_idx = df["voltage_V"].abs().idxmax()
    i = df.index.get_loc(absmax_idx)
    b1 = df.iloc[: i + 1].copy()
    b2 = df.iloc[i:].copy()
    return b1, b2

def _detect_switch_voltage(branch: pd.DataFrame, mode: str) -> Optional[float]:

    if len(branch) < 5:
        return None
    v = branch["voltage_V"].to_numpy(dtype=float)
    i = branch["current_A"].to_numpy(dtype=float)
    g = _safe_div(i, v)  
    g_s = pd.Series(g).rolling(window=3, center=True, min_periods=1).mean().to_numpy()

    dg = np.diff(g_s)
    if dg.size == 0:
        return None
    med = np.nanmedian(dg)
    mad = np.nanmedian(np.abs(dg - med))
    noise_scale = mad if mad > 0 else np.nanstd(dg)
    if not np.isfinite(noise_scale) or noise_scale == 0:
        noise_scale = 1e-12

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
    plots_dir: str = "data/plots",
    make_plot: bool = True,
) -> IVMetrics:
    in_path = _resolve_path(iv_csv_path)
    run_id = in_path.stem

    df_full = pd.read_csv(in_path)
    if "voltage_V" not in df_full.columns or "current_A" not in df_full.columns:
        raise ValueError(f"CSV must contain columns voltage_V and current_A. Found: {list(df_full.columns)}")

    df = df_full.dropna(subset=["voltage_V", "current_A"]).copy()
    df["voltage_V"] = df["voltage_V"].astype(float)
    df["current_A"] = df["current_A"].astype(float)
    if "sweep" in df.columns and set(df["sweep"].astype(str).unique()) >= {"fwd", "rev"}:
        b1 = df[df["sweep"].astype(str) == "fwd"][["voltage_V", "current_A"]].copy()
        b2 = df[df["sweep"].astype(str) == "rev"][["voltage_V", "current_A"]].copy()
        hysteresis_area, v_grid, i1g, i2g = _compute_hysteresis_area_from_branches(b1, b2)
    else:
        core = df[["voltage_V", "current_A"]].copy()
        hysteresis_area, b1, b2, v_grid, i1g, i2g = _compute_hysteresis_area(core)

    # ON/OFF at Vread:
    row_pos = _closest_at_v(df, +abs(vread_V))
    row_neg = _closest_at_v(df, -abs(vread_V))

    ion_A = None
    ioff_A = None
    on_off_ratio = None
    notes = ""

    # Heuristic:
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

    out_path = _resolve_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    out_csv = out_path / f"iv_metrics_{run_id}.csv"
    pd.DataFrame([asdict(metrics)]).to_csv(out_csv, index=False)

    out_json = out_path / f"iv_metrics_{run_id}.json"
    out_json.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    if make_plot:
        plots_p = _resolve_path(plots_dir)
        plots_p.mkdir(parents=True, exist_ok=True)
        plot_hysteresis(
            str(in_path),
            out_path=str(plots_p / f"hysteresis_{run_id}.png"),
            title=f"I–V Hysteresis (Area={hysteresis_area:.3e})" if hysteresis_area is not None else "I–V Hysteresis",
            show=False,
        )

    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to IV CSV (must contain voltage_V,current_A)")
    ap.add_argument("--vread", type=float, default=0.2)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--plots", default="data/plots")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    m = compute_iv_metrics(
        args.csv,
        vread_V=args.vread,
        out_dir=args.out,
        plots_dir=args.plots,
        make_plot=not args.no_plot,
    )
    print("IV metrics saved for:", m.run_id)
    print(asdict(m))
