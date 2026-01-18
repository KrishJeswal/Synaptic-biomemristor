import random
import math


def run_pulse_experiment(voltage, width_ms, num_pulses):
    currents = []
    base_conductance = 1e-6

    for n in range(num_pulses):
        conductance = base_conductance * (1 + 0.05 * n)
        noise = random.uniform(-0.05, 0.05) * conductance
        current = (conductance + noise) * voltage
        currents.append(current)

    return currents

def _iv_conductance_base(v: float) -> float:
    return 1e-5 + 1e-5 * math.tanh(v)


def run_iv_sweep(v_start, v_end, steps, bidirectional: bool = False):
    def _one_way(v0: float, v1: float, n: int, direction: int) -> tuple[list[float], list[float]]:
        vv: list[float] = []
        ii: list[float] = []
        for k in range(n):
            v = v0 + k * (v1 - v0) / (n - 1)
            g0 = _iv_conductance_base(v)
            loop = 2.5e-6 * math.tanh(v / 0.4)
            g = g0 + direction * loop

            g *= (1.0 + random.uniform(-0.01, 0.01))

            i_val = g * v
            vv.append(v)
            ii.append(i_val)
        return vv, ii

    if not bidirectional:
        return _one_way(v_start, v_end, steps, direction=+1)

    v_f, i_f = _one_way(v_start, v_end, steps, direction=+1)
    v_r, i_r = _one_way(v_end, v_start, steps, direction=-1)
    if len(v_r) > 0:
        v_r = v_r[1:]
        i_r = i_r[1:]
    return v_f + v_r, i_f + i_r


def run_endurance_experiment(
    set_voltage_V: float,
    reset_voltage_V: float,
    read_voltage_V: float,
    width_ms: int,
    n_cycles: int,
) -> dict:
    g_lrs0 = 8e-5
    g_hrs0 = 2e-6
    lrs_drift = -2.5e-7  
    hrs_drift = +1.0e-8  

    cycle_number: list[int] = []
    phase: list[str] = []
    voltage_V: list[float] = []
    current_A: list[float] = []
    conductance_S: list[float] = []

    g_lrs = g_lrs0
    g_hrs = g_hrs0

    for c in range(1, n_cycles + 1):
        g_lrs = max(1e-7, g_lrs + lrs_drift + random.uniform(-0.02, 0.02) * g_lrs)
        g_read = g_lrs * (1.0 + random.uniform(-0.01, 0.01))
        i_read = g_read * read_voltage_V

        cycle_number.append(c)
        phase.append("SET_READ")
        voltage_V.append(read_voltage_V)
        current_A.append(i_read)
        conductance_S.append(g_read)

        g_hrs = max(1e-9, g_hrs + hrs_drift + random.uniform(-0.03, 0.03) * g_hrs)
        g_read = g_hrs * (1.0 + random.uniform(-0.02, 0.02))
        i_read = g_read * read_voltage_V

        cycle_number.append(c)
        phase.append("RESET_READ")
        voltage_V.append(read_voltage_V)
        current_A.append(i_read)
        conductance_S.append(g_read)

    return {
        "cycle_number": cycle_number,
        "phase": phase,
        "voltage_V": voltage_V,
        "current_A": current_A,
        "conductance_S": conductance_S,
        "set_voltage_V": [set_voltage_V] * len(cycle_number),
        "reset_voltage_V": [reset_voltage_V] * len(cycle_number),
        "pulse_width_ms": [width_ms] * len(cycle_number),
    }


def run_retention_experiment(
    read_voltage_V: float,
    delays_s: list[float],
    g0_S: float = 6e-5,
    g_inf_S: float = 2e-5,
    tau_s: float = 120.0,
) -> dict:
    time_s: list[float] = []
    voltage_V: list[float] = []
    current_A: list[float] = []
    conductance_S: list[float] = []

    for t in delays_s:
        g_t = g_inf_S + (g0_S - g_inf_S) * math.exp(-float(t) / float(tau_s))
        g_t *= (1.0 + random.uniform(-0.01, 0.01))
        i_t = g_t * read_voltage_V

        time_s.append(float(t))
        voltage_V.append(float(read_voltage_V))
        current_A.append(float(i_t))
        conductance_S.append(float(g_t))

    return {
        "time_s": time_s,
        "voltage_V": voltage_V,
        "current_A": current_A,
        "conductance_S": conductance_S,
    }