"""Build a small dataset-version registry for world-model experiments.

Run:
    python3 examples/build_dataset_registry.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snn_world_model import (  # noqa: E402
    DatasetVersion,
    SimConfig,
    generate_transitions,
    write_dataset_manifest,
    write_dataset_registry,
    write_transition_csv,
)


OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "dataset_versions"
REGISTRY_PATH = OUT_DIR / "dataset_registry.json"
SPLIT_RATIO = 0.78


DATASET_CONFIGS = {
    "easy-slip-v1": SimConfig(seed=101, steps=900, slip_min=0.86, slip_max=1.02, pose_noise=0.0015),
    "base-slip-v1": SimConfig(seed=102, steps=900, slip_min=0.72, slip_max=1.03, pose_noise=0.0030),
    "hard-slip-v1": SimConfig(seed=103, steps=900, slip_min=0.60, slip_max=1.04, pose_noise=0.0060),
}


def build_version(name: str, config: SimConfig) -> DatasetVersion:
    transitions = generate_transitions(config)
    cut = int(len(transitions) * SPLIT_RATIO)
    dataset_path = OUT_DIR / f"{name}.csv"
    manifest_path = OUT_DIR / f"{name}.manifest.json"
    write_transition_csv(transitions, dataset_path)
    manifest = write_dataset_manifest(
        path=manifest_path,
        dataset_csv=dataset_path,
        config=config,
        train_count=cut,
        test_count=len(transitions) - cut,
        split_ratio=SPLIT_RATIO,
    )
    return DatasetVersion.from_manifest(name, manifest)


def main() -> None:
    versions = [build_version(name, config) for name, config in DATASET_CONFIGS.items()]
    write_dataset_registry(REGISTRY_PATH, versions)
    print(f"Dataset versions: {len(versions)}")
    for version in versions:
        print(f"{version.name}: {version.dataset_sha256[:12]}... {version.total_count} transitions")
    print(f"Registry saved: {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
