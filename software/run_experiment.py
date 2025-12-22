# software/run_experiment.py
from __future__ import annotations
import argparse
import time
import yaml
from pathlib import Path
CFG_PATH = Path(__file__).resolve().parent / "config.yaml"
cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))

from experiment_pulse import run_pulse
from experiment_iv import run_iv

from backend_sim import run_pulse_experiment as sim_pulse
from backend_sim import run_iv_sweep as sim_iv
from analysis_iv import compute_iv_metrics

from device_serial import ESP32SerialBackend, SerialConfig

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
    ap.add_argument("--config", default="software/config.yaml")
    ap.add_argument("--mode", choices=["pulse", "iv", "all"], required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    backend, closer = make_backend(cfg)

    try:
        if args.mode == "pulse":
            run_pulse(cfg)
        elif args.mode == "iv":
            iv_csv = run_iv(cfg)
            compute_iv_metrics(iv_csv, vread_V=float(cfg["vread_V"]), out_dir="data/processed")
        elif args.mode == "all":
            iv_csv = run_iv(cfg)
            compute_iv_metrics(iv_csv, vread_V=float(cfg["vread_V"]), out_dir="data/processed")
            time.sleep(2)
            run_pulse(cfg)
    finally:
        if closer is not None:
            closer.close()

if __name__ == "__main__":
    main()
