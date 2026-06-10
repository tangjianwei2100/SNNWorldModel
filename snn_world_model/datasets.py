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
