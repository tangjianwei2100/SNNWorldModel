"""Trainable SNN-style world model.

The model is intentionally small: fixed LIF reservoir features plus a trained
linear readout. This keeps the first engineering milestone inspectable while
still giving us a real train/save/load/evaluate loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path

from .environment import SensorResidual, Transition, dead_reckoning_next, transition_target, world_model_features
from .neurons import LIFParams, LIFState, lif_step
from .robot import RobotState, wrap_angle
from .types import Action


def _dot(a: list[float] | tuple[float, ...], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class WorldModelMetrics:
    dead_reckoning_mae: float
    snn_mae: float
    reliability_mae: float


class LIFReservoirWorldModel:
    """Predict next RobotState and reliability from state/action."""

    def __init__(
        self,
        input_size: int = 13,
        reservoir_size: int = 28,
        seed: int = 11,
        lif_params: LIFParams | None = None,
        residual_scale: float = 1.0,
    ):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.seed = seed
        self.residual_scale = residual_scale
        self.lif_params = lif_params or LIFParams(tau_mem_ms=12.0, v_threshold=1.0)
        rng = random.Random(seed)
        self.input_weights = [
            [rng.uniform(-1.2, 1.2) for _ in range(input_size)]
            for _ in range(reservoir_size)
        ]
        self.bias = [rng.uniform(-0.35, 0.35) for _ in range(reservoir_size)]
        self.readout = [[0.0 for _ in range(reservoir_size + input_size + 1)] for _ in range(4)]

    def encode(
        self,
        state: RobotState,
        action: Action,
        sensor_residual: SensorResidual | None = None,
    ) -> list[float]:
        inputs = world_model_features(state, action, sensor_residual)
        reservoir = []
        for weights, bias in zip(self.input_weights, self.bias):
            neuron_state = LIFState()
            spike_total = 0
            current = 0.95 + 0.75 * _dot(inputs, weights) + bias
            for _ in range(5):
                neuron_state, spike = lif_step(neuron_state, current, 1.0, self.lif_params)
                spike_total += spike
            reservoir.append(spike_total / 5.0)
        return [1.0, *inputs, *reservoir]

    def predict_residual_and_reliability(
        self,
        state: RobotState,
        action: Action,
        sensor_residual: SensorResidual | None = None,
    ) -> tuple[float, float, float, float]:
        features = self.encode(state, action, sensor_residual)
        output = [_dot(features, row) for row in self.readout]
        reliability = max(0.0, min(1.0, output[3]))
        return output[0], output[1], output[2], reliability

    def predict_delta_and_reliability(
        self,
        state: RobotState,
        action: Action,
        sensor_residual: SensorResidual | None = None,
    ) -> tuple[float, float, float, float]:
        dead_next = dead_reckoning_next(state, action)
        residual_dx, residual_dy, residual_dtheta, reliability = self.predict_residual_and_reliability(
            state,
            action,
            sensor_residual,
        )
        return (
            dead_next.x - state.x + self.residual_scale * residual_dx,
            dead_next.y - state.y + self.residual_scale * residual_dy,
            wrap_angle(dead_next.theta - state.theta + self.residual_scale * residual_dtheta),
            reliability,
        )

    def predict_next(
        self,
        state: RobotState,
        action: Action,
        sensor_residual: SensorResidual | None = None,
    ) -> RobotState:
        dx, dy, dtheta, _ = self.predict_delta_and_reliability(state, action, sensor_residual)
        return RobotState(state.x + dx, state.y + dy, wrap_angle(state.theta + dtheta))

    def train(
        self,
        transitions: list[Transition],
        epochs: int = 18,
        learning_rate: float = 0.018,
        l2: float = 0.0004,
        sample_weights: list[float] | None = None,
    ) -> None:
        if sample_weights is not None and len(sample_weights) != len(transitions):
            raise ValueError("sample_weights must match transitions length")
        rng = random.Random(self.seed + 1000)
        order = list(range(len(transitions)))
        for _ in range(epochs):
            rng.shuffle(order)
            for idx in order:
                transition = transitions[idx]
                sample_weight = 1.0 if sample_weights is None else sample_weights[idx]
                features = self.encode(transition.state, transition.action, transition.sensor_residual)
                target_dx, target_dy, target_dtheta, target_reliability = transition_target(transition)
                dead_next = dead_reckoning_next(transition.state, transition.action)
                targets = (
                    target_dx - (dead_next.x - transition.state.x),
                    target_dy - (dead_next.y - transition.state.y),
                    wrap_angle(target_dtheta - wrap_angle(dead_next.theta - transition.state.theta)),
                    target_reliability,
                )
                for output_idx, target in enumerate(targets):
                    prediction = _dot(features, self.readout[output_idx])
                    error = prediction - target
                    row = self.readout[output_idx]
                    for feature_idx, value in enumerate(features):
                        row[feature_idx] -= learning_rate * (sample_weight * error * value + l2 * row[feature_idx])

    def _train_one_target(
        self,
        features: list[float],
        targets: tuple[float, float, float, float],
        learning_rate: float,
        l2: float,
    ) -> None:
        for output_idx, target in enumerate(targets):
            prediction = _dot(features, self.readout[output_idx])
            error = prediction - target
            row = self.readout[output_idx]
            for feature_idx, value in enumerate(features):
                row[feature_idx] -= learning_rate * (error * value + l2 * row[feature_idx])

    def train_rollout_aware(
        self,
        transitions: list[Transition],
        horizon: int = 8,
        epochs: int = 6,
        learning_rate: float = 0.006,
        l2: float = 0.0004,
        stride: int = 4,
    ) -> None:
        """Fine-tune on short windows starting from the model's own predictions."""
        if len(transitions) < horizon + 1:
            raise ValueError("not enough transitions for rollout-aware training")
        starts = list(range(0, len(transitions) - horizon, stride))
        rng = random.Random(self.seed + 2000)
        for _ in range(epochs):
            rng.shuffle(starts)
            for start in starts:
                predicted = transitions[start].state
                sensor_residual = transitions[start].sensor_residual
                for offset in range(horizon):
                    transition = transitions[start + offset]
                    features = self.encode(predicted, transition.action, sensor_residual)
                    target = transition.next_state
                    dead_next = dead_reckoning_next(predicted, transition.action)
                    targets = (
                        target.x - dead_next.x,
                        target.y - dead_next.y,
                        wrap_angle(target.theta - dead_next.theta),
                        transition.reliability,
                    )
                    self._train_one_target(features, targets, learning_rate, l2)
                    predicted = self.predict_next(predicted, transition.action, sensor_residual)
                    sensor_residual = transition.sensor_residual

    def evaluate(self, transitions: list[Transition]) -> WorldModelMetrics:
        if not transitions:
            raise ValueError("evaluate() needs at least one transition")
        snn_error = 0.0
        dead_error = 0.0
        reliability_error = 0.0
        for transition in transitions:
            dx, dy, dtheta, reliability = self.predict_delta_and_reliability(
                transition.state,
                transition.action,
                transition.sensor_residual,
            )
            target_dx, target_dy, target_dtheta, target_reliability = transition_target(transition)
            dead_next = dead_reckoning_next(transition.state, transition.action)
            dead_dx = dead_next.x - transition.state.x
            dead_dy = dead_next.y - transition.state.y
            dead_dtheta = wrap_angle(dead_next.theta - transition.state.theta)
            snn_error += (abs(dx - target_dx) + abs(dy - target_dy) + abs(dtheta - target_dtheta)) / 3.0
            dead_error += (
                abs(dead_dx - target_dx)
                + abs(dead_dy - target_dy)
                + abs(dead_dtheta - target_dtheta)
            ) / 3.0
            reliability_error += abs(reliability - target_reliability)
        n = len(transitions)
        return WorldModelMetrics(dead_error / n, snn_error / n, reliability_error / n)

    def save(self, path: str | Path) -> None:
        payload = {
            "input_size": self.input_size,
            "reservoir_size": self.reservoir_size,
            "seed": self.seed,
            "residual_scale": self.residual_scale,
            "lif_params": self.lif_params.__dict__,
            "input_weights": self.input_weights,
            "bias": self.bias,
            "readout": self.readout,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LIFReservoirWorldModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            input_size=payload["input_size"],
            reservoir_size=payload["reservoir_size"],
            seed=payload["seed"],
            lif_params=LIFParams(**payload["lif_params"]),
            residual_scale=payload.get("residual_scale", 1.0),
        )
        model.input_weights = payload["input_weights"]
        model.bias = payload["bias"]
        model.readout = payload["readout"]
        return model
