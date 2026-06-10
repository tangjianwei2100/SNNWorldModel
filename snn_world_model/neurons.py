"""Leaky integrate-and-fire neuron primitives."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LIFParams:
    tau_mem_ms: float = 20.0
    resistance: float = 1.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0


@dataclass(frozen=True)
class LIFState:
    voltage: float = 0.0


def lif_step(state: LIFState, input_current: float, dt_ms: float, params: LIFParams) -> tuple[LIFState, int]:
    """Advance one LIF neuron by one time step."""
    leak = -(state.voltage - params.v_rest)
    drive = params.resistance * input_current
    dv = (leak + drive) * (dt_ms / params.tau_mem_ms)
    voltage = state.voltage + dv
    if voltage >= params.v_threshold:
        return LIFState(params.v_reset), 1
    return LIFState(voltage), 0


def simulate_lif(currents: list[float], dt_ms: float, params: LIFParams | None = None) -> tuple[list[float], list[int]]:
    """Simulate a LIF neuron over a current sequence."""
    params = params or LIFParams()
    state = LIFState(params.v_rest)
    voltages: list[float] = []
    spikes: list[int] = []
    for current in currents:
        state, spike = lif_step(state, current, dt_ms, params)
        voltages.append(state.voltage)
        spikes.append(spike)
    return voltages, spikes
