import random
import math

def run_pulse_experiment(voltage, width_ms, num_pulses):
    """
    Fake memristor behavior:
    conductance slowly increases with pulse number
    """
    currents = []
    base_conductance = 1e-6

    for n in range(num_pulses):
        conductance = base_conductance * (1 + 0.05 * n)
        noise = random.uniform(-0.05, 0.05) * conductance
        current = (conductance + noise) * voltage
        currents.append(current)

    return currents


def run_iv_sweep(v_start, v_end, steps):
    voltages = []
    currents = []

    for i in range(steps):
        v = v_start + i * (v_end - v_start) / (steps - 1)
        g = 1e-5 + 1e-5 * math.tanh(v) 
        i_val = g * v
        voltages.append(v)
        currents.append(i_val)

    return voltages, currents