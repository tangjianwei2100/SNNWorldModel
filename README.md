# snn_world_model

SNN-style robot world model engineering prototype.

Detailed tutorial: http://tangjian.me

## Modules

- `snn_world_model.neurons`: LIF neuron state update and spike simulation.
- `snn_world_model.robot`: simple differential-drive robot dynamics.
- `snn_world_model.environment`: slip-aware robot simulation and transition dataset generation.
- `snn_world_model.datasets`: transition CSV, dataset manifest, and dataset-version registry helpers.
- `snn_world_model.models`: trainable LIF reservoir world model with a linear readout.
- `snn_world_model.training`: rollout augmentation, hard-window mining, sample weighting, and early stopping.
- `snn_world_model.evaluation`: one-step, multi-step, closed-loop, and experiment-registry metrics.
- `snn_world_model.planning`: candidate-action rollout planner and safety shield.
- `examples/safety_shield_demo.py`: a small end-to-end control-loop demo.
- `examples/build_dataset_registry.py`: generate easy/base/hard dataset versions and a registry file.
- `examples/train_world_model.py`: train, save, load, and evaluate the first SNN-style world model.

## Quick Start

Install the package in editable mode:

```bash
python -m pip install -e .
```

Run the safety-shield demo:

```bash
python3 examples/safety_shield_demo.py
```

Build a small dataset-version registry:

```bash
python3 examples/build_dataset_registry.py
```

## Train Stage-2 World Model

```bash
python3 examples/train_world_model.py
```

The first engineering model learns residual motion over a no-slip differential-drive baseline. It also consumes sensor residual features: encoder forward error, encoder turn residual, IMU yaw residual, motor-load estimate, and contact/reliability evidence.

Expected output includes:

- `outputs/stage2_lif_reservoir_world_model.json`: saved model parameters.
- `outputs/stage2_training_report.json`: training, test, and closed-loop metrics.
- `outputs/stage2_closed_loop_trajectory.csv`: robot trajectory from the learned model loop.

## Run Tests

```bash
python3 -m unittest discover -s tests
```
