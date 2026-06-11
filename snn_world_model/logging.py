"""Trajectory log writing helpers for bench robot runs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .datasets import TRAJECTORY_LOG_REQUIRED_COLUMNS, validate_trajectory_log_csv
from .robot import DifferentialDriveParams, RobotState, diff_drive_step
from .types import Action


@dataclass(frozen=True)
class RealLogMetadata:
    dataset_version: str
    robot_id: str
    routine: str
    battery_voltage: float
    floor_patch_id: str
    payload_kg: float
    wheel_condition: str
    sensor_rate_hz: int


@dataclass(frozen=True)
class LoggerRunConfig:
    episode_id: str
    duration_s: float
    command_left: float
    command_right: float
    metadata: RealLogMetadata

    @property
    def sample_count(self) -> int:
        return int(round(self.duration_s * self.metadata.sensor_rate_hz))

    @property
    def dt_s(self) -> float:
        return 1.0 / self.metadata.sensor_rate_hz


@dataclass(frozen=True)
class LoggerWriteSummary:
    schema: str
    path: str
    episode_id: str
    rows: int
    transitions: int
    max_time_gap_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def trajectory_row(
    *,
    config: LoggerRunConfig,
    step: int,
    time_s: float,
    state: RobotState,
    encoder_forward: float,
    encoder_turn: float,
    motor_load: float,
    contact: float,
) -> dict[str, str]:
    metadata = config.metadata
    return {
        "episode_id": config.episode_id,
        "step": str(step),
        "time_s": f"{time_s:.3f}",
        "command_left": f"{config.command_left:.6f}",
        "command_right": f"{config.command_right:.6f}",
        "state_x": f"{state.x:.6f}",
        "state_y": f"{state.y:.6f}",
        "state_theta": f"{state.theta:.6f}",
        "encoder_forward": f"{encoder_forward:.6f}",
        "encoder_turn": f"{encoder_turn:.6f}",
        "imu_yaw": f"{state.theta:.6f}",
        "motor_load": f"{motor_load:.6f}",
        "contact": f"{contact:.6f}",
        "dataset_version": metadata.dataset_version,
        "robot_id": metadata.robot_id,
        "routine": metadata.routine,
        "battery_voltage": f"{metadata.battery_voltage:.3f}",
        "floor_patch_id": metadata.floor_patch_id,
        "payload_kg": f"{metadata.payload_kg:.3f}",
        "wheel_condition": metadata.wheel_condition,
        "sensor_rate_hz": str(metadata.sensor_rate_hz),
    }


def make_dry_run_trajectory_rows(
    config: LoggerRunConfig,
    *,
    initial_state: RobotState | None = None,
    drive_params: DifferentialDriveParams | None = None,
) -> list[dict[str, str]]:
    if config.sample_count < 2:
        raise ValueError("logger run needs at least two samples")
    if config.metadata.sensor_rate_hz <= 0:
        raise ValueError("sensor_rate_hz must be positive")

    state = initial_state or RobotState(0.0, 0.0, 0.0)
    params = drive_params or DifferentialDriveParams()
    action = Action(config.command_left, config.command_right)
    rows: list[dict[str, str]] = []
    previous_state = state

    for step in range(config.sample_count):
        if step > 0:
            previous_state = state
            state = diff_drive_step(state, action, params, config.dt_s)
        encoder_forward = ((state.x - previous_state.x) ** 2 + (state.y - previous_state.y) ** 2) ** 0.5
        encoder_turn = state.theta - previous_state.theta
        motor_load = min(1.0, 0.2 + abs(config.command_left - config.command_right) * 1.5)
        rows.append(
            trajectory_row(
                config=config,
                step=step,
                time_s=step * config.dt_s,
                state=state,
                encoder_forward=encoder_forward,
                encoder_turn=encoder_turn,
                motor_load=motor_load,
                contact=1.0,
            )
        )
    return rows


def write_trajectory_log_csv(rows: list[dict[str, str]], path: str | Path) -> LoggerWriteSummary:
    if not rows:
        raise ValueError("cannot write an empty trajectory log")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(TRAJECTORY_LOG_REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    validation = validate_trajectory_log_csv(path)
    return LoggerWriteSummary(
        schema=validation.schema,
        path=str(path),
        episode_id=str(rows[0]["episode_id"]),
        rows=validation.rows,
        transitions=max(0, validation.rows - validation.episodes),
        max_time_gap_s=validation.max_time_gap_s,
    )


def write_dry_run_trajectory_log_csv(
    config: LoggerRunConfig,
    path: str | Path,
    *,
    initial_state: RobotState | None = None,
    drive_params: DifferentialDriveParams | None = None,
) -> LoggerWriteSummary:
    rows = make_dry_run_trajectory_rows(
        config,
        initial_state=initial_state,
        drive_params=drive_params,
    )
    return write_trajectory_log_csv(rows, path)
