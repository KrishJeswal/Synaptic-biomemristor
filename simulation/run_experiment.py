from __future__ import annotations
import argparse
import sys
import time
import yaml
from pathlib import Path

_SIMULATION_DIR = Path(__file__).resolve().parent
if str(_SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMULATION_DIR))

_EXPERIMENT_DIR = _SIMULATION_DIR / "experiment_data"
_ANALYSIS_DIR = _SIMULATION_DIR / "analysis_metrics"
for _p in (_EXPERIMENT_DIR, _ANALYSIS_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
CFG_PATH = Path(__file__).resolve().parent / "config.yaml"
cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))

from experiment_pulse import run_pulse
from experiment_iv import run_iv
from experiment_endurance import run_endurance
from experiment_retention import run_retention
from backend_sim import run_pulse_experiment as sim_pulse
from backend_sim import run_iv_sweep as sim_iv
from analysis_iv import compute_iv_metrics
from analysis_endurance import compute_endurance_metrics
from analysis_retention import compute_retention_metrics

class SimBackend:
    def run_pulse_experiment(self, voltage_V, width_ms, num_pulses):
        return sim_pulse(voltage_V, width_ms, num_pulses)

    def run_iv_sweep(self, start_V, end_V, steps):
        return sim_iv(start_V, end_V, steps)

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_backend(cfg: dict):
    backend_type = (cfg.get("backend") or "sim").lower()
    if backend_type == "sim":
        return SimBackend(), None
    if backend_type == "esp32":
        from device_serial import ESP32SerialBackend, SerialConfig
        scfg = cfg["serial"]
        backend = ESP32SerialBackend(
            SerialConfig(
                port=scfg["port"],
                baud=int(scfg.get("baud", 115200)),
                timeout_s=float(scfg.get("timeout_s", 2.0)),
            )
        )
        backend.open()
        return backend, backend
    raise ValueError(f"Unknown backend: {backend_type}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="simulation/config.yaml")
    ap.add_argument("--mode", choices=["pulse", "iv", "endurance", "retention", "all"], required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    paths = cfg.get("paths", {}) or {}
    raw_dir = str(paths.get("data_raw_dir", "data/raw"))
    processed_dir = str(paths.get("data_processed_dir", "data/processed"))
    plots_dir = str(paths.get("plots_dir", "data/plots"))
    if plots_dir.strip().lower() in {"plots", "plot"}:
        plots_dir = "data/plots"

    backend, closer = make_backend(cfg)
    try:
        if args.mode == "pulse":
            run_pulse(cfg)
        elif args.mode == "iv":
            iv_csv = run_iv(cfg)
            compute_iv_metrics(iv_csv, vread_V=float(cfg["vread_V"]), out_dir=processed_dir, plots_dir=plots_dir)
        elif args.mode == "endurance":
            end_csv = run_endurance(cfg)
            compute_endurance_metrics(end_csv, out_dir=processed_dir, plots_dir=plots_dir)
        elif args.mode == "retention":
            ret_csv = run_retention(cfg)
            compute_retention_metrics(ret_csv, out_dir=processed_dir, plots_dir=plots_dir)
        elif args.mode == "all":
            iv_csv = run_iv(cfg)
            time.sleep(0.5)
            pulse_csv = run_pulse(cfg)
            time.sleep(0.5)
            end_csv = run_endurance(cfg)
            time.sleep(0.5)
            ret_csv = run_retention(cfg)

            compute_iv_metrics(iv_csv, vread_V=float(cfg["vread_V"]), out_dir=processed_dir, plots_dir=plots_dir)
            compute_endurance_metrics(end_csv, out_dir=processed_dir, plots_dir=plots_dir)
            compute_retention_metrics(ret_csv, out_dir=processed_dir, plots_dir=plots_dir)
    finally:
        if closer is not None:
            closer.close()

if __name__ == "__main__":
    main()
