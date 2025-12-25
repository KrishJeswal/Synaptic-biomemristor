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

    print("Running I–V sweep")

    voltages, currents = run_iv_sweep(IV_START, IV_END, IV_STEPS)

    csv_name = "data/raw/iv_sweep.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["voltage_V", "current_A"])
        for v, i in zip(voltages, currents):
            writer.writerow([v, i])

    print("CSV saved:", csv_name)

    plt.figure()
    plt.plot(voltages, currents, marker="o")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("I–V Curve (Simulated)")
    plt.grid(True)

    plot_name=f"data/plots/iv_curve.png"
    plt.savefig(plot_name)
    plt.show()

    print("Plot saved:", plot_name)
    return csv_name

if __name__ == "__main__":
    run_iv()
