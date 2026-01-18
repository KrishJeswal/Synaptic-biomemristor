from __future__ import annotations
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

_SIM_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _SIM_DIR.parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from backend_sim import run_pulse_experiment

def load_cfg() -> dict:
    cfg_path = _SIM_DIR / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def run_pulse(cfg: dict | None = None) -> str:
    if cfg is None:
        cfg = load_cfg()

    pulse = cfg["pulse"]
    PULSE_VOLTAGE = float(pulse["voltage_V"])
    PULSE_WIDTH_MS = int(pulse["width_ms"])
    NUM_PULSES = int(pulse["num_pulses"])

    print("Running pulse experiment")

    currents = run_pulse_experiment(PULSE_VOLTAGE, PULSE_WIDTH_MS, NUM_PULSES)
    conductance = [i / PULSE_VOLTAGE for i in currents]

    csv_name = str(_PROJECT_ROOT / "data" / "raw" / f"pulse_{PULSE_VOLTAGE}V_{PULSE_WIDTH_MS}ms.csv")
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pulse_number", "voltage_V", "current_A", "conductance_S"])
        for idx, (i, g) in enumerate(zip(currents, conductance), start=1):
            writer.writerow([idx, PULSE_VOLTAGE, i, g])

    print("CSV saved:", csv_name)

    plt.figure()
    plt.plot(range(1, NUM_PULSES + 1), conductance, marker="o")
    plt.xlabel("Pulse number")
    plt.ylabel("Conductance (S)")
    plt.title("STP → LTP (Simulated)")
    plt.grid(True)

    plot_name = str(_PROJECT_ROOT / "data" / "plots" / f"pulse_{PULSE_VOLTAGE}V.png")
    plt.savefig(plot_name)
    plt.show()
    print("Plot saved:", plot_name)
    return csv_name

if __name__ == "__main__":
    run_pulse()
