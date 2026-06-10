"""Differential-drive robot dynamics used by the planner demos."""

from dataclasses import dataclass
from math import cos, pi, sin

from .types import Action


@dataclass(frozen=True)
class RobotState:
    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class DifferentialDriveParams:
    dt: float = 0.10
    wheel_base: float = 0.42
    max_wheel_speed: float = 1.00


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle


def diff_drive_step(
    state: RobotState,
    action: Action,
    params: DifferentialDriveParams | None = None,
    left_slip: float = 1.0,
    right_slip: float = 1.0,
) -> RobotState:
    """Advance a differential-drive robot by one step."""
    params = params or DifferentialDriveParams()
    left = clip(action.left, -params.max_wheel_speed, params.max_wheel_speed) * left_slip
    right = clip(action.right, -params.max_wheel_speed, params.max_wheel_speed) * right_slip
    forward = 0.5 * (left + right)
    omega = (right - left) / params.wheel_base
    theta = wrap_angle(state.theta + omega * params.dt)
    return RobotState(
        x=state.x + forward * cos(theta) * params.dt,
        y=state.y + forward * sin(theta) * params.dt,
        theta=theta,
    )
