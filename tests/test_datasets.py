import tempfile
import unittest
from pathlib import Path

from snn_world_model import (
    DatasetVersion,
    SimConfig,
    generate_transitions,
    read_dataset_registry,
    read_transition_csv,
    write_dataset_manifest,
    write_dataset_registry,
    write_transition_csv,
)
from snn_world_model.datasets import file_sha256


class DatasetTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
