import os
import serial
import time
import pandas as pd
import matplotlib.pyplot as plt

PORT = "COM7"
BAUD = 115200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "..", "data", "final values")
os.makedirs(SAVE_DIR, exist_ok=True)

def out(name: str) -> str:
    return os.path.join(SAVE_DIR, name)

def open_serial():
    ser = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(2.0)
    ser.reset_input_buffer()
    return ser

def send_cmd(ser, cmd: str):
    ser.write((cmd.strip() + "\n").encode("utf-8"))

def read_until_end(ser, *, idle_timeout_s: float = 6.0, hard_timeout_s: float = 60.0):
    data = []
    raw_lines = []
    status = "OK"

    t0 = time.time()
    last_rx = t0
    consecutive_timeouts = 0

    while True:
        raw = ser.readline()
        if not raw:
            consecutive_timeouts += 1
            if (time.time() - last_rx) >= idle_timeout_s or (time.time() - t0) >= hard_timeout_s:
                status = "TIMEOUT"
                break
            continue

        consecutive_timeouts = 0
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        last_rx = time.time()

        raw_lines.append(line)

        if line == "END":
            break
        if line.startswith("ESP32_MEMRISTOR_CONTROLLER"):
            status = "RESET"
            break

        if line.startswith("TRIP"):
            status = "TRIP"
            parts = line.split()
            if len(parts) >= 3:
                try:
                    data.append({"V": float(parts[1]), "I_A": float(parts[2])})
                except:
                    pass
            continue

        if line.startswith("ERR"):
            status = "ERR"
            continue

        parts = line.split()
        if len(parts) >= 2:
            try:
                data.append({"V": float(parts[0]), "I_A": float(parts[1])})
            except:
                pass

    return status, data, raw_lines

def run_id_and_limit(ser, limit_mA=1.0):
    send_cmd(ser, "ID")
    read_until_end(ser)
    send_cmd(ser, f"SETLIM {limit_mA}")
    read_until_end(ser)

def run_sweep(ser, start_v, end_v, steps, settle_ms=30, hysteresis=0):
    cmd = f"SWEEP {start_v} {end_v} {steps} {settle_ms} {hysteresis}"
    send_cmd(ser, cmd)
    return read_until_end(ser)

def run_pulse_train(ser, v, width_ms, n, gap_ms):
    cmd = f"PULSE {v} {width_ms} {n} {gap_ms}"
    send_cmd(ser, cmd)
    return read_until_end(ser)

def save_iv_plot(df, filename_png, title):
    if df is None or df.empty or ("V" not in df.columns) or ("I_A" not in df.columns):
        return

    plt.figure()
    plt.plot(df["V"], df["I_A"])
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(title)
    plt.grid(True)
    plt.savefig(out(filename_png), dpi=200)
    plt.close()

def save_pulse_plot(df, filename_png, title):
    if df is None or df.empty or ("I_A" not in df.columns):
        return

    plt.figure()
    plt.plot(range(1, len(df) + 1), df["I_A"])
    plt.xlabel("Pulse #")
    plt.ylabel("Current (A)")
    plt.title(title)
    plt.grid(True)
    plt.savefig(out(filename_png), dpi=200)
    plt.close()

def main():
    print("Opening serial on", PORT)
    print("Saving outputs to:", SAVE_DIR)

    ser = open_serial()
    run_id_and_limit(ser, limit_mA=3.0)

    print("\nRunning IV sweep (0 → 1V)...")
    status, data, raw = run_sweep(ser, start_v=0.0, end_v=3.0, steps=50, settle_ms=30, hysteresis=0)
    df_iv = pd.DataFrame(data)

    with open(out("iv_sweep_0_to_1V.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(raw) + "\n")

    df_iv.to_csv(out("iv_sweep_0_to_1V.csv"), index=False)
    save_iv_plot(df_iv, "iv_sweep_0_to_1V.png", f"I–V Sweep 0→1V (status: {status})")

    print("Saved:")
    print("  -", out("iv_sweep_0_to_1V.csv"))
    print("  -", out("iv_sweep_0_to_1V.png"))
    print("Status:", status, "Points:", len(df_iv))
    print("\nRunning IV sweep hysteresis (0 → 1V → 0)...")
    status2, data2, raw2 = run_sweep(ser, start_v=0.0, end_v=3.0, steps=50, settle_ms=30, hysteresis=1)
    df_hys = pd.DataFrame(data2)

    with open(out("iv_sweep_hysteresis_0_to_1V.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(raw2) + "\n")

    df_hys.to_csv(out("iv_sweep_hysteresis_0_to_1V.csv"), index=False)
    save_iv_plot(df_hys, "iv_sweep_hysteresis_0_to_1V.png", f"I–V Sweep Hysteresis 0→1V→0 (status: {status2})")

    print("Saved:")
    print("  -", out("iv_sweep_hysteresis_0_to_1V.csv"))
    print("  -", out("iv_sweep_hysteresis_0_to_1V.png"))
    print("Status:", status2, "Points:", len(df_hys))
    print("\nRunning pulse train...")
    status3, data3, raw3 = run_pulse_train(ser, v=1.2, width_ms=20, n=50, gap_ms=100)
    df_pulse = pd.DataFrame(data3)

    with open(out("pulse_train.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(raw3) + "\n")

    df_pulse.to_csv(out("pulse_train.csv"), index=False)
    save_pulse_plot(df_pulse, "pulse_train.png", f"Pulse Train (status: {status3})")

    print("Saved:")
    print("  -", out("pulse_train.csv"))
    print("  -", out("pulse_train.png"))
    print("Status:", status3, "Pulses:", len(df_pulse))

    ser.close()
    print("\nAll files saved into:")
    print("   ", SAVE_DIR)

if __name__ == "__main__":
    main()
