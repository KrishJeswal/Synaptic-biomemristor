from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

from backend_sim import run_iv_sweep

def load_cfg() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def run_iv(cfg: dict | None = None) -> str:
    if cfg is None:
        cfg = load_cfg()

    iv = cfg["iv"]
    IV_START = float(iv["start_V"])
    IV_END = float(iv["end_V"])
    IV_STEPS = int(iv["steps"])

    print("Running I–V sweep (forward + reverse for hysteresis)")

    # Generate a bidirectional sweep so the hysteresis analysis has two branches.
    voltages, currents = run_iv_sweep(IV_START, IV_END, IV_STEPS, bidirectional=True)

    csv_name = "data/raw/iv_sweep.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        # Include a sweep label for convenience (analysis scripts will ignore extra columns).
        writer.writerow(["voltage_V", "current_A", "sweep"])

        # The first (IV_STEPS) points are forward (including the endpoint).
        # The remaining points are reverse (endpoint removed to avoid a duplicate point).
        for idx, (v, i) in enumerate(zip(voltages, currents)):
            sweep = "fwd" if idx < IV_STEPS else "rev"
            writer.writerow([v, i, sweep])

    print("CSV saved:", csv_name)

    plt.figure()
    # Plot the full loop; it should visually differ from the hysteresis plot (which shades area).
    plt.plot(voltages, currents, marker="o", linewidth=1)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("I–V Sweep (Simulated, Forward + Reverse)")
    plt.grid(True)

    plot_name=f"data/plots/iv_curve.png"
    plt.savefig(plot_name)
    plt.show()

    print("Plot saved:", plot_name)
    return csv_name

if __name__ == "__main__":
    run_iv()
