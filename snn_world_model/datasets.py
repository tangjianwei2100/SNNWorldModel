"""Dataset versioning helpers for transition logs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .environment import SensorResidual, SimConfig, Transition
from .robot import RobotState
from .types import Action


DATASET_SCHEMA = "snn-transition-v1"
TRAJECTORY_LOG_SCHEMA = "snn-trajectory-log-v1"

TRAJECTORY_LOG_STREAM_COLUMNS = (
    "episode_id",
    "step",
    "time_s",
    "command_left",
    "command_right",
    "state_x",
    "state_y",
    "state_theta",
    "encoder_forward",
    "encoder_turn",
    "imu_yaw",
    "motor_load",
    "contact",
)

TRAJECTORY_LOG_METADATA_COLUMNS = (
    "dataset_version",
    "robot_id",
    "routine",
    "battery_voltage",
    "floor_patch_id",
    "payload_kg",
    "wheel_condition",
    "sensor_rate_hz",
)

TRAJECTORY_LOG_REQUIRED_COLUMNS = TRAJECTORY_LOG_STREAM_COLUMNS + TRAJECTORY_LOG_METADATA_COLUMNS


@dataclass(frozen=True)
class TrajectoryLogSummary:
    schema: str
    rows: int
    episodes: int
    routines: list[str]
    dataset_versions: list[str]
    max_step_gap: int
    max_time_gap_s: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryBuildSummary:
    schema: str
    source_rows: int
    source_episodes: int
    transition_count: int
    dataset_versions: list[str]
    routines: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DatasetVersion:
    name: str
    dataset_sha256: str
    dataset_csv: str
    sim_config: dict
    train_count: int
    test_count: int
    total_count: int

    @classmethod
    def from_manifest(cls, name: str, manifest: dict) -> "DatasetVersion":
        return cls(
            name=name,
            dataset_sha256=manifest["dataset_sha256"],
            dataset_csv=manifest["dataset_csv"],
            sim_config=manifest["sim_config"],
            train_count=manifest["train_count"],
            test_count=manifest["test_count"],
            total_count=manifest["total_count"],
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DatasetRegistry:
    versions: list[DatasetVersion]

    def by_name(self, name: str) -> DatasetVersion:
        for version in self.versions:
            if version.name == name:
                return version
        raise KeyError(f"unknown dataset version: {name}")

    def to_dict(self) -> dict:
        return {
            "schema": "snn-dataset-registry-v1",
            "versions": [version.to_dict() for version in self.versions],
        }


def write_dataset_registry(path: str | Path, versions: list[DatasetVersion]) -> dict:
    if not versions:
        raise ValueError("dataset registry needs at least one version")
    names = [version.name for version in versions]
    if len(names) != len(set(names)):
        raise ValueError("dataset version names must be unique")
    payload = DatasetRegistry(versions).to_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def read_dataset_registry(path: str | Path) -> DatasetRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    versions = [DatasetVersion(**item) for item in payload["versions"]]
    return DatasetRegistry(versions)


def transition_to_row(transition: Transition) -> dict[str, float]:
    return {
        "state_x": transition.state.x,
        "state_y": transition.state.y,
        "state_theta": transition.state.theta,
        "action_left": transition.action.left,
        "action_right": transition.action.right,
        "next_x": transition.next_state.x,
        "next_y": transition.next_state.y,
        "next_theta": transition.next_state.theta,
        "reliability": transition.reliability,
        "left_slip": transition.left_slip,
        "right_slip": transition.right_slip,
        "encoder_forward": transition.sensor_residual.encoder_forward,
        "encoder_turn": transition.sensor_residual.encoder_turn,
        "imu_yaw": transition.sensor_residual.imu_yaw,
        "motor_load": transition.sensor_residual.motor_load,
        "contact": transition.sensor_residual.contact,
    }


def row_to_transition(row: dict[str, str]) -> Transition:
    values = {key: float(value) for key, value in row.items()}
    return Transition(
        state=RobotState(values["state_x"], values["state_y"], values["state_theta"]),
        action=Action(values["action_left"], values["action_right"]),
        next_state=RobotState(values["next_x"], values["next_y"], values["next_theta"]),
        reliability=values["reliability"],
        left_slip=values["left_slip"],
        right_slip=values["right_slip"],
        sensor_residual=SensorResidual(
            encoder_forward=values["encoder_forward"],
            encoder_turn=values["encoder_turn"],
            imu_yaw=values["imu_yaw"],
            motor_load=values["motor_load"],
            contact=values["contact"],
        ),
    )


def write_transition_csv(transitions: list[Transition], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [transition_to_row(transition) for transition in transitions]
    if not rows:
        raise ValueError("cannot write an empty transition dataset")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_transition_csv(path: str | Path) -> list[Transition]:
    with Path(path).open(encoding="utf-8", newline="") as file:
        return [row_to_transition(row) for row in csv.DictReader(file)]


def trajectory_log_schema() -> dict:
    return {
        "schema": TRAJECTORY_LOG_SCHEMA,
        "required_columns": list(TRAJECTORY_LOG_REQUIRED_COLUMNS),
        "stream_columns": list(TRAJECTORY_LOG_STREAM_COLUMNS),
        "metadata_columns": list(TRAJECTORY_LOG_METADATA_COLUMNS),
        "acceptance_checks": [
            "all required columns are present",
            "metadata columns are non-empty for every row",
            "step increments by 1 within each episode",
            "time_s is monotonic within each episode",
        ],
    }


def validate_trajectory_log_rows(rows: list[dict[str, str]]) -> TrajectoryLogSummary:
    if not rows:
        raise ValueError("trajectory log cannot be empty")

    columns = set(rows[0].keys())
    missing = [column for column in TRAJECTORY_LOG_REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"trajectory log missing required columns: {', '.join(missing)}")

    episodes: dict[str, list[dict[str, str]]] = {}
    routines: set[str] = set()
    dataset_versions: set[str] = set()
    for row_index, row in enumerate(rows, start=1):
        for column in TRAJECTORY_LOG_METADATA_COLUMNS:
            if not str(row[column]).strip():
                raise ValueError(f"row {row_index} has empty metadata column: {column}")
        episode_id = str(row["episode_id"])
        episodes.setdefault(episode_id, []).append(row)
        routines.add(str(row["routine"]))
        dataset_versions.add(str(row["dataset_version"]))

    max_step_gap = 0
    max_time_gap_s = 0.0
    for episode_id, episode_rows in episodes.items():
        ordered = sorted(episode_rows, key=lambda row: int(row["step"]))
        previous_step: int | None = None
        previous_time: float | None = None
        for row in ordered:
            step = int(row["step"])
            time_s = float(row["time_s"])
            if previous_step is not None:
                step_gap = step - previous_step
                if step_gap != 1:
                    raise ValueError(f"episode {episode_id} has non-contiguous step gap: {step_gap}")
                time_gap = time_s - previous_time
                if time_gap < 0:
                    raise ValueError(f"episode {episode_id} has decreasing time_s")
                max_step_gap = max(max_step_gap, step_gap)
                max_time_gap_s = max(max_time_gap_s, time_gap)
            previous_step = step
            previous_time = time_s

    return TrajectoryLogSummary(
        schema=TRAJECTORY_LOG_SCHEMA,
        rows=len(rows),
        episodes=len(episodes),
        routines=sorted(routines),
        dataset_versions=sorted(dataset_versions),
        max_step_gap=max_step_gap,
        max_time_gap_s=max_time_gap_s,
    )


def validate_trajectory_log_csv(path: str | Path) -> TrajectoryLogSummary:
    with Path(path).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return validate_trajectory_log_rows(rows)


def trajectory_log_rows_to_transitions(rows: list[dict[str, str]]) -> list[Transition]:
    validate_trajectory_log_rows(rows)
    episodes: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        episodes.setdefault(str(row["episode_id"]), []).append(row)

    transitions: list[Transition] = []
    for episode_rows in episodes.values():
        ordered = sorted(episode_rows, key=lambda row: int(row["step"]))
        for current, following in zip(ordered, ordered[1:]):
            transitions.append(
                Transition(
                    state=RobotState(
                        float(current["state_x"]),
                        float(current["state_y"]),
                        float(current["state_theta"]),
                    ),
                    action=Action(
                        float(current["command_left"]),
                        float(current["command_right"]),
                    ),
                    next_state=RobotState(
                        float(following["state_x"]),
                        float(following["state_y"]),
                        float(following["state_theta"]),
                    ),
                    reliability=float(current["contact"]),
                    left_slip=0.0,
                    right_slip=0.0,
                    sensor_residual=SensorResidual(
                        encoder_forward=float(current["encoder_forward"]),
                        encoder_turn=float(current["encoder_turn"]),
                        imu_yaw=float(current["imu_yaw"]),
                        motor_load=float(current["motor_load"]),
                        contact=float(current["contact"]),
                    ),
                )
            )
    return transitions


def build_transition_dataset_from_trajectory_log(
    trajectory_log_csv: str | Path,
    transition_csv: str | Path,
) -> TrajectoryBuildSummary:
    with Path(trajectory_log_csv).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    log_summary = validate_trajectory_log_rows(rows)
    transitions = trajectory_log_rows_to_transitions(rows)
    write_transition_csv(transitions, transition_csv)
    return TrajectoryBuildSummary(
        schema=DATASET_SCHEMA,
        source_rows=log_summary.rows,
        source_episodes=log_summary.episodes,
        transition_count=len(transitions),
        dataset_versions=log_summary.dataset_versions,
        routines=log_summary.routines,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset_manifest(
    *,
    path: str | Path,
    dataset_csv: str | Path,
    config: SimConfig,
    train_count: int,
    test_count: int,
    split_ratio: float,
) -> dict:
    manifest = {
        "schema": DATASET_SCHEMA,
        "dataset_csv": str(Path(dataset_csv)),
        "dataset_sha256": file_sha256(dataset_csv),
        "sim_config": asdict(config),
        "split_ratio": split_ratio,
        "train_count": train_count,
        "test_count": test_count,
        "total_count": train_count + test_count,
    }
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
