from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import serial

@dataclass
class SerialConfig:
    port: str
    baud: int = 115200
    timeout_s: float = 2.0

class ESP32SerialBackend:
    """
    Real backend that talks to ESP32 over Serial.
    """

    def __init__(self, cfg: SerialConfig):
        self.cfg = cfg
        self.ser: Optional[serial.Serial] = None

    def open(self) -> None:
        self.ser = serial.Serial(self.cfg.port, self.cfg.baud, timeout=self.cfg.timeout_s)
        # Give ESP32 time to reboot when serial opens
        time.sleep(1.2)
        self._flush()

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _flush(self) -> None:
        if not self.ser:
            return
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def _write_line(self, line: str) -> None:
        assert self.ser is not None, "Serial not open"
        self.ser.write((line.strip() + "\n").encode("utf-8"))

    def _read_lines_until_end(self, max_seconds: float = 10.0) -> List[str]:
        """
        Read lines until we see 'END' or time out.
        """
        assert self.ser is not None, "Serial not open"
        lines: List[str] = []
        t0 = time.time()
        while True:
            if (time.time() - t0) > max_seconds:
                break
            raw = self.ser.readline()
            if not raw:
                continue
            s = raw.decode("utf-8", errors="ignore").strip()
            if not s:
                continue
            lines.append(s)
            if "END" in s.upper():
                break
        return lines

    @staticmethod
    def _parse_floats_from_line(s: str) -> List[float]:
        # Accept comma-separated or space-separated numeric lines
        s = s.replace(",", " ")
        parts = [p for p in s.split() if p]
        out: List[float] = []
        for p in parts:
            try:
                out.append(float(p))
            except ValueError:
                # ignore non-numeric tokens
                pass
        return out

    def identify(self) -> str:
        self._flush()
        self._write_line("ID")
        lines = self._read_lines_until_end(max_seconds=2.0)
        # Return last non-empty line as ID-ish string
        return lines[-1] if lines else "UNKNOWN"

    def run_pulse_experiment(self, voltage_V: float, width_ms: int, num_pulses: int) -> List[float]:
        """
        Returns: currents_A list, length ~= num_pulses (depends on firmware sampling).
        We accept either:
            - one current per pulse
            - or multiple samples; in that case you can post-process to per-pulse later.
        """
        self._flush()
        cmd = f"PULSE {voltage_V} {width_ms} {num_pulses}"
        self._write_line(cmd)
        lines = self._read_lines_until_end(max_seconds=max(10.0, num_pulses * (width_ms / 1000.0 + 0.05)))

        currents: List[float] = []
        for s in lines:
            vals = self._parse_floats_from_line(s)
            # Heuristic: if line has 1 float, treat it as current
            # if line has 2 floats, treat as (voltage,current) and take current
            if len(vals) == 1:
                currents.append(vals[0])
            elif len(vals) >= 2:
                currents.append(vals[-1])
        return currents

    def run_iv_sweep(self, start_V: float, end_V: float, steps: int) -> Tuple[List[float], List[float]]:
        """
        Returns: (voltages_V, currents_A).
        """
        self._flush()
        cmd = f"SWEEP {start_V} {end_V} {steps}"
        self._write_line(cmd)
        lines = self._read_lines_until_end(max_seconds=10.0 + steps * 0.1)

        v_list: List[float] = []
        i_list: List[float] = []
        for s in lines:
            vals = self._parse_floats_from_line(s)
            if len(vals) >= 2:
                v_list.append(vals[0])
                i_list.append(vals[1])
        return v_list, i_list
