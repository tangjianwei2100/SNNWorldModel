import csv
import tempfile
import unittest
from pathlib import Path

from snn_world_model import (
    DatasetVersion,
    SimConfig,
    build_transition_dataset_from_trajectory_log,
    generate_transitions,
    read_dataset_registry,
    read_transition_csv,
    trajectory_log_rows_to_transitions,
    validate_trajectory_log_rows,
    write_dataset_manifest,
    write_dataset_registry,
    write_transition_csv,
)
from snn_world_model.datasets import file_sha256


class DatasetTests(unittest.TestCase):
    def make_trajectory_rows(self) -> list[dict[str, str]]:
        return [
            {
                "episode_id": "ep-001",
                "step": str(step),
                "time_s": f"{step * 0.05:.2f}",
                "command_left": "0.62",
                "command_right": "0.63",
                "state_x": f"{step * 0.01:.3f}",
                "state_y": "0.000",
                "state_theta": f"{step * 0.002:.3f}",
                "encoder_forward": "0.010",
                "encoder_turn": "0.001",
                "imu_yaw": f"{step * 0.002:.3f}",
                "motor_load": "0.310",
                "contact": "1.0",
                "dataset_version": "easy-slip-v1",
                "robot_id": "bench-bot-01",
                "routine": "straight-repeatability",
                "battery_voltage": "12.2",
                "floor_patch_id": "floor-a",
                "payload_kg": "0.0",
                "wheel_condition": "clean",
                "sensor_rate_hz": "20",
            }
            for step in range(3)
        ]

    def test_transition_csv_round_trips_sensor_residuals(self):
        transitions = generate_transitions(SimConfig(seed=3, steps=24))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transitions.csv"
            write_transition_csv(transitions, path)
            loaded = read_transition_csv(path)
        self.assertEqual(len(loaded), len(transitions))
        self.assertAlmostEqual(loaded[5].sensor_residual.motor_load, transitions[5].sensor_residual.motor_load)
        self.assertAlmostEqual(loaded[5].state.x, transitions[5].state.x)

    def test_manifest_records_dataset_fingerprint(self):
        transitions = generate_transitions(SimConfig(seed=4, steps=12))
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "transitions.csv"
            manifest_path = Path(tmp) / "manifest.json"
            write_transition_csv(transitions, dataset)
            manifest = write_dataset_manifest(
                path=manifest_path,
                dataset_csv=dataset,
                config=SimConfig(seed=4, steps=12),
                train_count=8,
                test_count=4,
                split_ratio=8 / 12,
            )
            fingerprint = file_sha256(dataset)
        self.assertEqual(manifest["dataset_sha256"], fingerprint)
        self.assertEqual(manifest["schema"], "snn-transition-v1")

    def test_dataset_registry_round_trips_versions(self):
        transitions = generate_transitions(SimConfig(seed=5, steps=16))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "base.csv"
            manifest_path = root / "base.manifest.json"
            registry_path = root / "dataset_registry.json"
            write_transition_csv(transitions, dataset)
            manifest = write_dataset_manifest(
                path=manifest_path,
                dataset_csv=dataset,
                config=SimConfig(seed=5, steps=16),
                train_count=12,
                test_count=4,
                split_ratio=12 / 16,
            )
            version = DatasetVersion.from_manifest("base-slip-v1", manifest)
            write_dataset_registry(registry_path, [version])
            registry = read_dataset_registry(registry_path)
        self.assertEqual(registry.by_name("base-slip-v1").dataset_sha256, manifest["dataset_sha256"])
        self.assertEqual(registry.to_dict()["schema"], "snn-dataset-registry-v1")

    def test_dataset_registry_rejects_duplicate_names(self):
        version = DatasetVersion(
            name="duplicate",
            dataset_sha256="abc",
            dataset_csv="dataset.csv",
            sim_config={"seed": 1},
            train_count=1,
            test_count=1,
            total_count=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_dataset_registry(Path(tmp) / "registry.json", [version, version])

    def test_trajectory_log_rows_validate_schema_and_episode_summary(self):
        rows = self.make_trajectory_rows()
        summary = validate_trajectory_log_rows(rows)
        self.assertEqual(summary.schema, "snn-trajectory-log-v1")
        self.assertEqual(summary.rows, 3)
        self.assertEqual(summary.episodes, 1)
        self.assertEqual(summary.routines, ["straight-repeatability"])

    def test_trajectory_log_rows_build_neighbor_transitions(self):
        transitions = trajectory_log_rows_to_transitions(self.make_trajectory_rows())
        self.assertEqual(len(transitions), 2)
        self.assertAlmostEqual(transitions[0].state.x, 0.0)
        self.assertAlmostEqual(transitions[0].action.left, 0.62)
        self.assertAlmostEqual(transitions[0].action.right, 0.63)
        self.assertAlmostEqual(transitions[0].next_state.x, 0.01)
        self.assertAlmostEqual(transitions[0].sensor_residual.encoder_turn, 0.001)

    def test_trajectory_log_csv_builds_transition_dataset(self):
        rows = self.make_trajectory_rows()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "trajectory_log.csv"
            transition_path = root / "transitions.csv"
            with log_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            summary = build_transition_dataset_from_trajectory_log(log_path, transition_path)
            loaded = read_transition_csv(transition_path)
        self.assertEqual(summary.transition_count, 2)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(summary.dataset_versions, ["easy-slip-v1"])

    def test_trajectory_log_rows_reject_missing_required_column(self):
        row = {
            "episode_id": "ep-001",
            "step": "0",
            "time_s": "0.00",
            "command_left": "0.62",
            "command_right": "0.62",
            "state_x": "0.000",
            "state_y": "0.000",
            "state_theta": "0.000",
            "encoder_forward": "0.010",
            "encoder_turn": "0.000",
            "motor_load": "0.310",
            "contact": "1.0",
            "dataset_version": "easy-slip-v1",
            "robot_id": "bench-bot-01",
            "routine": "straight-repeatability",
            "battery_voltage": "12.2",
            "floor_patch_id": "floor-a",
            "payload_kg": "0.0",
            "wheel_condition": "clean",
            "sensor_rate_hz": "20",
        }
        with self.assertRaisesRegex(ValueError, "imu_yaw"):
            validate_trajectory_log_rows([row])


if __name__ == "__main__":
    unittest.main()
