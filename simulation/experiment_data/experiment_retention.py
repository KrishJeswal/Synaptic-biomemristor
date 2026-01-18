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

from backend_sim import run_retention_experiment


def load_cfg() -> dict:
    cfg_path = _SIM_DIR / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _default_delays_s() -> list[float]:
    return [
        0,
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
    ]


def run_retention(cfg: dict | None = None) -> str:
    if cfg is None:
        cfg = load_cfg()

    ret_cfg = (cfg.get("retention") or {})
    read_voltage_V = float(ret_cfg.get("read_voltage_V", float(cfg.get("vread_V", 0.2))))

    delays = ret_cfg.get("delays_s")
    if not delays:
        delays_s = _default_delays_s()
    else:
        delays_s = [float(x) for x in delays]

    print(f"Running retention: {len(delays_s)} read points")

    out = run_retention_experiment(read_voltage_V=read_voltage_V, delays_s=delays_s)

    csv_name = str(_PROJECT_ROOT / "data" / "raw" / "retention_reads.csv")
    Path(csv_name).parent.mkdir(parents=True, exist_ok=True)

    headers = list(out.keys())
    rows = zip(*[out[h] for h in headers])

    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(r)

    print("CSV saved:", csv_name)

    t = out["time_s"]
    g = out["conductance_S"]

    plt.figure()
    plt.plot(t, g, marker="o", linewidth=1)
    if len(t) >= 2 and min(t) >= 0:
        plt.xscale("symlog", linthresh=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Conductance (S)")
    plt.title("Retention (Simulated)")
    plt.grid(True)

    plot_name = str(_PROJECT_ROOT / "data" / "plots" / "retention_reads.png")
    Path(plot_name).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_name, bbox_inches="tight")
    plt.show()

    print("Plot saved:", plot_name)
    return csv_name


if __name__ == "__main__":
    run_retention()
