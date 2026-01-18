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

from backend_sim import run_endurance_experiment


def load_cfg() -> dict:
    cfg_path = _SIM_DIR / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def run_endurance(cfg: dict | None = None) -> str:

    if cfg is None:
        cfg = load_cfg()

    end_cfg = (cfg.get("endurance") or {})

    n_cycles = int(end_cfg.get("n_cycles", 200))
    set_voltage_V = float(end_cfg.get("set_voltage_V", 1.2))
    reset_voltage_V = float(end_cfg.get("reset_voltage_V", -1.2))
    read_voltage_V = float(end_cfg.get("read_voltage_V", float(cfg.get("vread_V", 0.2))))
    width_ms = int(end_cfg.get("width_ms", int(cfg.get("pulse", {}).get("width_ms", 10))))

    print(f"Running endurance: {n_cycles} cycles (SET/RESET + READ)")

    out = run_endurance_experiment(
        set_voltage_V=set_voltage_V,
        reset_voltage_V=reset_voltage_V,
        read_voltage_V=read_voltage_V,
        width_ms=width_ms,
        n_cycles=n_cycles,
    )

    csv_name = str(_PROJECT_ROOT / "data" / "raw" / f"endurance_cycles_{n_cycles}.csv")
    Path(csv_name).parent.mkdir(parents=True, exist_ok=True)

    headers = list(out.keys())
    rows = zip(*[out[h] for h in headers])

    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(r)

    print("CSV saved:", csv_name)

    cycles = out["cycle_number"]
    phases = out["phase"]
    g = out["conductance_S"]

    cyc_set = [c for c, p in zip(cycles, phases) if p == "SET_READ"]
    g_set = [gv for gv, p in zip(g, phases) if p == "SET_READ"]
    cyc_rst = [c for c, p in zip(cycles, phases) if p == "RESET_READ"]
    g_rst = [gv for gv, p in zip(g, phases) if p == "RESET_READ"]

    plt.figure()
    plt.plot(cyc_set, g_set, marker="o", linewidth=1, label="LRS (after SET)")
    plt.plot(cyc_rst, g_rst, marker="o", linewidth=1, label="HRS (after RESET)")
    plt.xlabel("Cycle")
    plt.ylabel("Conductance (S)")
    plt.title("Endurance Cycling (Simulated)")
    plt.grid(True)
    plt.legend()

    plot_name = str(_PROJECT_ROOT / "data" / "plots" / f"endurance_cycles_{n_cycles}.png")
    Path(plot_name).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_name, bbox_inches="tight")
    plt.show()

    print("Plot saved:", plot_name)
    return csv_name


if __name__ == "__main__":
    run_endurance()
