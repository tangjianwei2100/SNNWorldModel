import tempfile
import unittest
from pathlib import Path

from snn_world_model import (
    LoggerRunConfig,
    RealLogMetadata,
    make_dry_run_trajectory_rows,
    read_transition_csv,
    build_transition_dataset_from_trajectory_log,
    validate_trajectory_log_csv,
    write_dry_run_trajectory_log_csv,
)


class LoggingTests(unittest.TestCase):
    def make_config(self) -> LoggerRunConfig:
        metadata = RealLogMetadata(
            dataset_version="real-bench-v1",
            robot_id="robot-bench-01",
            routine="straight-slow",
            battery_voltage=12.1,
            floor_patch_id="floor-a",
            payload_kg=0.0,
            wheel_condition="clean",
            sensor_rate_hz=10,
        )
        return LoggerRunConfig(
            episode_id="real-bench-v1-straight-slow-01",
            duration_s=1.2,
            command_left=0.08,
            command_right=0.08,
            metadata=metadata,
        )

    def test_dry_run_rows_match_trajectory_schema(self):
        rows = make_dry_run_trajectory_rows(self.make_config())
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[0]["step"], "0")
        self.assertEqual(rows[-1]["step"], "11")
        self.assertEqual(rows[0]["dataset_version"], "real-bench-v1")

    def test_dry_run_csv_validates_and_builds_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "dry_run.csv"
            transition_path = root / "transitions.csv"
            summary = write_dry_run_trajectory_log_csv(self.make_config(), log_path)
            validation = validate_trajectory_log_csv(log_path)
            build = build_transition_dataset_from_trajectory_log(log_path, transition_path)
            transitions = read_transition_csv(transition_path)

        self.assertEqual(summary.rows, 12)
        self.assertEqual(validation.episodes, 1)
        self.assertEqual(build.transition_count, 11)
        self.assertEqual(len(transitions), 11)
        self.assertAlmostEqual(transitions[0].action.left, 0.08)


if __name__ == "__main__":
    unittest.main()
