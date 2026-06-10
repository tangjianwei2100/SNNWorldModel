"""Simulation environment and dataset generation for the engineering stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, cos, hypot, sin
import random

from .robot import DifferentialDriveParams, RobotState, diff_drive_step, wrap_angle
from .types import Action


@dataclass(frozen=True)
class SensorResidual:
    encoder_forward: float = 0.0
    encoder_turn: float = 0.0
    imu_yaw: float = 0.0
    motor_load: float = 0.0
    contact: float = 0.0


@dataclass(frozen=True)
class Transition:
    state: RobotState
    action: Action
    next_state: RobotState
    reliability: float
    left_slip: float
    right_slip: float
    sensor_residual: SensorResidual = field(default_factory=SensorResidual)


@dataclass(frozen=True)
class SimConfig:
    seed: int = 7
    steps: int = 1600
    dt: float = 0.10
    slip_min: float = 0.72
    slip_max: float = 1.03
    pose_noise: float = 0.003


class SlipRobotEnvironment:
    """Differential-drive simulator with slowly changing wheel slip."""

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self.rng = random.Random(self.config.seed)
        self.params = DifferentialDriveParams(dt=self.config.dt)
        self.state = RobotState(-0.55, -0.20, 0.10)
        self.left_slip = 0.95
        self.right_slip = 0.95
        self.sensor_residual = SensorResidual()

    def sample_action(self) -> Action:
        base = self.rng.uniform(0.18, 0.86)
        turn = self.rng.uniform(-0.48, 0.48)
        return Action(base - turn, base + turn)

    def _update_slip(self) -> None:
        dx = self.state.x - 0.90
        dy = self.state.y - 0.42
        low_reliability_patch = exp(-(dx * dx + dy * dy) / 0.42)
        left_target = 0.97 - 0.20 * low_reliability_patch
        right_target = 0.98 - 0.13 * low_reliability_patch
        self.left_slip += 0.18 * (left_target - self.left_slip) + self.rng.uniform(-0.010, 0.010)
        self.right_slip += 0.18 * (right_target - self.right_slip) + self.rng.uniform(-0.010, 0.010)
        self.left_slip = max(self.config.slip_min, min(self.config.slip_max, self.left_slip))
        self.right_slip = max(self.config.slip_min, min(self.config.slip_max, self.right_slip))

    def _noisy(self, state: RobotState) -> RobotState:
        noise = self.config.pose_noise
        return RobotState(
            state.x + self.rng.uniform(-noise, noise),
            state.y + self.rng.uniform(-noise, noise),
            wrap_angle(state.theta + self.rng.uniform(-noise, noise)),
        )

    def step(self, action: Action) -> Transition:
        observed = self._noisy(self.state)
        sensor_residual = self.sensor_residual
        self._update_slip()
        next_state = diff_drive_step(
            self.state,
            action,
            self.params,
            left_slip=self.left_slip,
            right_slip=self.right_slip,
        )
        speed = abs(action.left) + abs(action.right)
        slip_loss = abs(1.0 - self.left_slip) + abs(1.0 - self.right_slip)
        reliability = max(0.05, min(1.0, 1.0 - 0.95 * slip_loss - 0.03 * speed))
        dead_next = dead_reckoning_next(self.state, action, self.params)
        nominal_forward = hypot(dead_next.x - self.state.x, dead_next.y - self.state.y)
        actual_forward = hypot(next_state.x - self.state.x, next_state.y - self.state.y)
        nominal_turn = wrap_angle(dead_next.theta - self.state.theta)
        actual_turn = wrap_angle(next_state.theta - self.state.theta)
        transition = Transition(
            state=observed,
            action=action,
            next_state=self._noisy(next_state),
            reliability=reliability,
            left_slip=self.left_slip,
            right_slip=self.right_slip,
            sensor_residual=sensor_residual,
        )
        self.sensor_residual = SensorResidual(
            encoder_forward=nominal_forward - actual_forward,
            encoder_turn=(action.right * (1.0 - self.right_slip)) - (action.left * (1.0 - self.left_slip)),
            imu_yaw=wrap_angle(actual_turn - nominal_turn),
            motor_load=speed * slip_loss,
            contact=max(0.0, 1.0 - reliability),
        )
        self.state = next_state
        if abs(self.state.x) > 3.5 or abs(self.state.y) > 3.5:
            self.state = RobotState(
                self.rng.uniform(-0.8, 0.2),
                self.rng.uniform(-0.6, 0.4),
                self.rng.uniform(-0.8, 0.8),
            )
        return transition


def generate_transitions(config: SimConfig | None = None) -> list[Transition]:
    env = SlipRobotEnvironment(config)
    transitions: list[Transition] = []
    steps = env.config.steps
    for _ in range(steps):
        transitions.append(env.step(env.sample_action()))
    return transitions


def transition_target(transition: Transition) -> tuple[float, float, float, float]:
    dx = transition.next_state.x - transition.state.x
    dy = transition.next_state.y - transition.state.y
    dtheta = wrap_angle(transition.next_state.theta - transition.state.theta)
    return dx, dy, dtheta, transition.reliability


def dead_reckoning_next(state: RobotState, action: Action, params: DifferentialDriveParams | None = None) -> RobotState:
    """Nominal no-slip baseline used for comparison."""
    return diff_drive_step(state, action, params or DifferentialDriveParams())


def body_frame_features(state: RobotState, action: Action) -> tuple[float, ...]:
    """Compact input vector using state symbols and wheel commands."""
    return (
        state.x,
        state.y,
        sin(state.theta),
        cos(state.theta),
        action.left,
        action.right,
        action.right - action.left,
        0.5 * (action.left + action.right),
    )


def sensor_residual_features(sensor_residual: SensorResidual | None) -> tuple[float, ...]:
    sensor_residual = sensor_residual or SensorResidual()
    return (
        sensor_residual.encoder_forward,
        sensor_residual.encoder_turn,
        sensor_residual.imu_yaw,
        sensor_residual.motor_load,
        sensor_residual.contact,
    )


def world_model_features(
    state: RobotState,
    action: Action,
    sensor_residual: SensorResidual | None = None,
) -> tuple[float, ...]:
    return (*body_frame_features(state, action), *sensor_residual_features(sensor_residual))
