from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


def _guess_path(name: str) -> Path | None:
    candidates = [
        REPO_ROOT / name,
        REPO_ROOT / "data" / "raw" / name,
        REPO_ROOT / "data" / "final values" / name,
        REPO_ROOT / "data" / "processed" / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_iv_hysteresis(iv_hys_path: Path) -> pd.DataFrame:
    df = pd.read_csv(iv_hys_path)
    required = {"voltage_V", "current_A", "cycle", "direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"IV hysteresis CSV is missing columns: {sorted(missing)}. "
            "Expected a simulated dataset like sim_iv_sweep_hysteresis.csv"
        )
    return df


def load_pulse_train(pulse_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pulse_path)
    required = {"pulse_number", "phase", "voltage_V", "current_A"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Pulse train CSV is missing columns: {sorted(missing)}. "
            "Expected a simulated dataset like sim_pulse_train.csv"
        )
    return df


def export_hardware_like_csvs(
    iv_df: pd.DataFrame,
    pulse_df: pd.DataFrame,
    out_dir: Path,
    *,
    vmax_label: str = "0_to_3.3V",
) -> dict[str, Path]:
    out_paths: dict[str, Path] = {}

    iv_forward = iv_df[(iv_df["cycle"] == 1) & (iv_df["direction"] == "forward")].copy()
    iv_forward = iv_forward.sort_values(["index_in_sweep"], kind="stable")
    iv_forward_out = pd.DataFrame({
        "V": iv_forward["voltage_V"],
        "I_A": iv_forward["current_A"],
    })
    p1 = out_dir / f"sim_iv_sweep_{vmax_label}.csv"
    iv_forward_out.to_csv(p1, index=False)
    out_paths["iv_sweep_csv"] = p1

    iv_hys = iv_df[iv_df["cycle"] == 1].copy()
    iv_hys["_dir_order"] = iv_hys["direction"].map({"forward": 0, "reverse": 1}).fillna(2)
    iv_hys = iv_hys.sort_values(["_dir_order", "index_in_sweep"], kind="stable")
    iv_hys_out = pd.DataFrame({
        "V": iv_hys["voltage_V"],
        "I_A": iv_hys["current_A"],
    })
    p2 = out_dir / f"sim_iv_sweep_hysteresis_{vmax_label}.csv"
    iv_hys_out.to_csv(p2, index=False)
    out_paths["iv_hysteresis_csv"] = p2

    pulse_high = pulse_df[pulse_df["phase"] == "pulse_high"].copy()
    pulse_high = pulse_high.sort_values(["pulse_number"], kind="stable")
    pulse_out = pd.DataFrame({
        "V": pulse_high["voltage_V"],
        "I_A": pulse_high["current_A"],
    })
    p3 = out_dir / "sim_pulse_train.csv"
    pulse_out.to_csv(p3, index=False)
    out_paths["pulse_train_csv"] = p3

    return out_paths


def plot_iv_sweep(iv_df: pd.DataFrame, out_dir: Path, *, show: bool = False) -> Path:
    df = iv_df[(iv_df["cycle"] == 1) & (iv_df["direction"] == "forward")].copy()
    df = df.sort_values(["index_in_sweep"], kind="stable")

    plt.figure(figsize=(7, 4.5))
    plt.plot(df["voltage_V"], df["current_A"] * 1000.0)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (mA)")
    plt.title("I–V Sweep (0 → 3.3V) — Simulated Honey Memristor")
    plt.grid(True)

    out_path = out_dir / "sim_iv_sweep.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    if show:
        plt.show()
    plt.close()
    return out_path


def plot_iv_hysteresis(iv_df: pd.DataFrame, out_dir: Path, *, show: bool = False) -> Path:
    df = iv_df[iv_df["cycle"] == 1].copy()
    fwd = df[df["direction"] == "forward"].sort_values(["index_in_sweep"], kind="stable")
    rev = df[df["direction"] == "reverse"].sort_values(["index_in_sweep"], kind="stable")

    plt.figure(figsize=(7, 4.5))
    plt.plot(fwd["voltage_V"], fwd["current_A"] * 1000.0, label="Forward")
    plt.plot(rev["voltage_V"], rev["current_A"] * 1000.0, label="Reverse")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (mA)")
    plt.title("I–V Sweep Hysteresis — (Forward vs Reverse)")
    plt.legend()
    plt.grid(True)

    out_path = out_dir / "sim_iv_hysteresis.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    if show:
        plt.show()
    plt.close()
    return out_path


def plot_pulse_train(pulse_df: pd.DataFrame, out_dir: Path, *, show: bool = False) -> Path:
    high = pulse_df[pulse_df["phase"] == "pulse_high"].copy()
    low = pulse_df[pulse_df["phase"] == "read_low"].copy()
    high = high.sort_values(["pulse_number"], kind="stable")
    low = low.sort_values(["pulse_number"], kind="stable")

    plt.figure(figsize=(7, 4.5))
    plt.plot(high["pulse_number"], high["current_A"] * 1000.0, label="Pulse High (write)")

    if not low.empty:
        plt.plot(low["pulse_number"], low["current_A"] * 1000.0, label="Read Low (retention)")

    plt.xlabel("Pulse Number")
    plt.ylabel("Current (mA)")
    plt.title("Output Current vs Pulse Train — Potentiation + Retention")
    plt.legend()
    plt.grid(True)

    out_path = out_dir / "sim_pulse_train.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    if show:
        plt.show()
    plt.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot demo memristor graphs from simulated datasets (no hardware needed)."
    )
    parser.add_argument(
        "--iv-hys",
        type=str,
        default=None,
        help="Path to sim_iv_sweep_hysteresis.csv (forward+reverse sweeps).",
    )
    parser.add_argument(
        "--pulse",
        type=str,
        default=None,
        help="Path to sim_pulse_train.csv (pulse_high + read_low).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "final values"),
        help="Directory to save PNG outputs (default: data/final values).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots in a window in addition to saving PNG files.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export simplified hardware-style CSVs alongside the plots.",
    )
    args = parser.parse_args()

    iv_path = Path(args.iv_hys) if args.iv_hys else _guess_path("sim_iv_sweep_hysteresis.csv")
    pulse_path = Path(args.pulse) if args.pulse else _guess_path("sim_pulse_train.csv")

    if iv_path is None or not iv_path.exists():
        raise FileNotFoundError(
            "Could not find sim_iv_sweep_hysteresis.csv. "
            "Place it in the repo root, data/raw/, or pass --iv-hys <path>."
        )
    if pulse_path is None or not pulse_path.exists():
        raise FileNotFoundError(
            "Could not find sim_pulse_train.csv. "
            "Place it in the repo root, data/raw/, or pass --pulse <path>."
        )

    out_dir = _ensure_out_dir(Path(args.out_dir))

    print("\n=== SIMULATED MEMRISTOR DEMO (Cu / Honey / Al) ===")
    print("Using simulated datasets (no serial / ESP32 required)")
    print("IV dataset:    ", iv_path)
    print("Pulse dataset: ", pulse_path)
    print("Saving outputs:", out_dir)

    iv_df = load_iv_hysteresis(iv_path)
    pulse_df = load_pulse_train(pulse_path)

    p_iv = plot_iv_sweep(iv_df, out_dir, show=args.show)
    p_hys = plot_iv_hysteresis(iv_df, out_dir, show=args.show)
    p_pulse = plot_pulse_train(pulse_df, out_dir, show=args.show)

    print("\nSaved plots:")
    print(" -", p_iv)
    print(" -", p_hys)
    print(" -", p_pulse)

    if args.export_csv:
        paths = export_hardware_like_csvs(iv_df, pulse_df, out_dir)
        print("\nExported simplified CSVs:")
        for k, v in paths.items():
            print(f" - {k}: {v}")

    print("\nDone ✅")


if __name__ == "__main__":
    main()
