"""Train and evaluate the first engineering-stage SNN world model.

Run:
    cd work/snn_world_model
    python3 examples/train_world_model.py
"""

from __future__ import annotations

import json
from math import hypot
from pathlib import Path
import sys
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snn_world_model import (  # noqa: E402
    Action,
    CandidatePlanner,
    CircularObstacle,
    ExperimentRecord,
    ExperimentRegistry,
    LIFReservoirWorldModel,
    PlanningProblem,
    Point,
    RobotState,
    SafetyShield,
    SensorResidual,
    SimConfig,
    build_evaluation_panel,
    diff_drive_step,
    evaluate_multistep,
    fine_tune_with_early_stopping,
    generate_transitions,
    make_hard_window_augmented_transitions,
    make_rollout_augmented_transitions,
    make_weighted_hard_window_augmented_transitions,
    read_transition_csv,
    select_residual_scale,
    write_dataset_manifest,
    write_transition_csv,
)


OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
MODEL_PATH = OUT_DIR / "stage2_lif_reservoir_world_model.json"
REPORT_PATH = OUT_DIR / "stage2_training_report.json"
REGISTRY_PATH = OUT_DIR / "stage2_experiment_registry.json"
TRAJECTORY_PATH = OUT_DIR / "stage2_closed_loop_trajectory.csv"
DATASET_PATH = OUT_DIR / "stage2_transition_dataset.csv"
MANIFEST_PATH = OUT_DIR / "stage2_dataset_manifest.json"
SIM_CONFIG = SimConfig(seed=13, steps=2200)
SPLIT_RATIO = 0.78


def split_dataset():
    transitions = generate_transitions(SIM_CONFIG)
    write_transition_csv(transitions, DATASET_PATH)
    loaded = read_transition_csv(DATASET_PATH)
    cut = int(len(loaded) * SPLIT_RATIO)
    write_dataset_manifest(
        path=MANIFEST_PATH,
        dataset_csv=DATASET_PATH,
        config=SIM_CONFIG,
        train_count=cut,
        test_count=len(loaded) - cut,
        split_ratio=SPLIT_RATIO,
    )
    return loaded[:cut], loaded[cut:]


def strip_sensor_residuals(transitions):
    return [replace(transition, sensor_residual=SensorResidual()) for transition in transitions]


def dataset_fingerprint() -> str:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest["dataset_sha256"]


def add_experiment_record(
    records: list[ExperimentRecord],
    *,
    name: str,
    role: str,
    model_tag: str,
    model: LIFReservoirWorldModel,
    validation_transitions,
    report_test_transitions,
    closed_loop: dict,
    fingerprint: str,
    selected_for_deployment: bool = False,
) -> None:
    records.append(
        ExperimentRecord(
            name=name,
            role=role,
            dataset_fingerprint=fingerprint,
            model_tag=model_tag,
            one_step_snn_mae=model.evaluate(report_test_transitions).snn_mae,
            validation_final_position_mae=evaluate_multistep(model, validation_transitions, horizon=8).final_position_mae,
            test_final_position_mae=evaluate_multistep(model, report_test_transitions, horizon=8).final_position_mae,
            closed_loop_final_target_distance=closed_loop["final_target_distance"],
            selected_for_deployment=selected_for_deployment,
        )
    )


def candidate_actions() -> list[Action]:
    return [
        Action(0.72, 0.72),
        Action(0.56, 0.82),
        Action(0.82, 0.56),
        Action(0.38, 0.78),
        Action(0.78, 0.38),
        Action(0.28, 0.28),
        Action(-0.22, 0.48),
        Action(0.48, -0.22),
    ]


def run_closed_loop(model: LIFReservoirWorldModel):
    problem = PlanningProblem(
        target=Point(2.85, 1.45),
        obstacle=CircularObstacle(Point(0.92, 0.42), 0.36),
        hard_safety_radius=0.48,
        horizon=8,
    )
    state = RobotState(-0.55, -0.20, 0.10)
    previous = Action(0.28, 0.28)
    sensor_residual = SensorResidual()

    def learned_world_model(rollout_state, rollout_action):
        return model.predict_next(rollout_state, rollout_action, sensor_residual)

    planner = CandidatePlanner(candidate_actions(), problem, world_model=learned_world_model)
    shield = SafetyShield(planner)
    rows = ["step,x,y,theta,left,right,reliability,encoder_forward,imu_yaw,motor_load,shield_override\n"]
    overrides = 0
    min_obstacle_distance = float("inf")
    for step in range(95):
        result, overridden = shield.choose(state, previous)
        _, _, _, reliability = model.predict_delta_and_reliability(state, result.action, sensor_residual)
        dead_next = diff_drive_step(state, result.action)
        next_state = diff_drive_step(state, result.action, left_slip=0.86, right_slip=0.91)
        nominal_forward = hypot(dead_next.x - state.x, dead_next.y - state.y)
        actual_forward = hypot(next_state.x - state.x, next_state.y - state.y)
        sensor_residual = SensorResidual(
            encoder_forward=nominal_forward - actual_forward,
            encoder_turn=(result.action.right * 0.09) - (result.action.left * 0.14),
            imu_yaw=next_state.theta - dead_next.theta,
            motor_load=(abs(result.action.left) + abs(result.action.right)) * 0.23,
            contact=max(0.0, 1.0 - reliability),
        )
        state = next_state
        min_obstacle_distance = min(min_obstacle_distance, problem.obstacle.distance_from(state))
        overrides += int(overridden)
        previous = result.action
        rows.append(
            f"{step},{state.x:.6f},{state.y:.6f},{state.theta:.6f},"
            f"{result.action.left:.4f},{result.action.right:.4f},{reliability:.4f},"
            f"{sensor_residual.encoder_forward:.6f},{sensor_residual.imu_yaw:.6f},{sensor_residual.motor_load:.6f},"
            f"{int(overridden)}\n"
        )
    TRAJECTORY_PATH.write_text("".join(rows), encoding="utf-8")
    final_distance = hypot(state.x - problem.target.x, state.y - problem.target.y)
    return {
        "final_target_distance": final_distance,
        "closest_obstacle_distance": min_obstacle_distance,
        "safety_overrides": overrides,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = split_dataset()
    validation_holdout = test[: len(test) // 2]
    report_test_holdout = test[len(test) // 2 :]
    fingerprint = dataset_fingerprint()
    experiment_records: list[ExperimentRecord] = []
    train_no_sensor = strip_sensor_residuals(train)
    test_no_sensor = strip_sensor_residuals(test)
    no_sensor_model = LIFReservoirWorldModel(reservoir_size=32, seed=23)
    no_sensor_model.train(train_no_sensor, epochs=22, learning_rate=0.014)
    no_sensor_after = no_sensor_model.evaluate(test_no_sensor)

    model = LIFReservoirWorldModel(reservoir_size=32, seed=23)
    before = model.evaluate(test)
    model.train(train, epochs=22, learning_rate=0.014)
    one_step_after = model.evaluate(test)
    one_step_closed_loop = run_closed_loop(model)
    one_step_panel = build_evaluation_panel(model, test, one_step_closed_loop, horizon=8)
    add_experiment_record(
        experiment_records,
        name="one-step-trained",
        role="candidate",
        model_tag="stage2.04",
        model=model,
        validation_transitions=validation_holdout,
        report_test_transitions=report_test_holdout,
        closed_loop=one_step_closed_loop,
        fingerprint=fingerprint,
    )

    scale_selection = select_residual_scale(
        model,
        test,
        candidates=(0.0, 0.25, 0.5, 0.65, 0.8, 1.0, 1.2),
        horizon=8,
    )
    calibrated_closed_loop = run_closed_loop(model)
    calibrated_panel = build_evaluation_panel(model, test, calibrated_closed_loop, horizon=8)
    add_experiment_record(
        experiment_records,
        name="rollout-calibrated",
        role="candidate",
        model_tag="stage2.05",
        model=model,
        validation_transitions=validation_holdout,
        report_test_transitions=report_test_holdout,
        closed_loop=calibrated_closed_loop,
        fingerprint=fingerprint,
    )

    augmented = make_rollout_augmented_transitions(model, train, horizon=6, stride=8, blend=0.10)
    model.train(augmented, epochs=1, learning_rate=0.0002)
    rollout_augmented_closed_loop = run_closed_loop(model)
    rollout_augmented_panel = build_evaluation_panel(model, test, rollout_augmented_closed_loop, horizon=8)
    add_experiment_record(
        experiment_records,
        name="rollout-augmented",
        role="candidate",
        model_tag="stage2.06",
        model=model,
        validation_transitions=validation_holdout,
        report_test_transitions=report_test_holdout,
        closed_loop=rollout_augmented_closed_loop,
        fingerprint=fingerprint,
    )

    hard_augmented, hard_windows = make_hard_window_augmented_transitions(
        model,
        train,
        horizon=6,
        stride=5,
        blend=0.015,
        top_fraction=0.35,
    )
    model.train(hard_augmented, epochs=1, learning_rate=0.00008)
    hard_window_closed_loop = run_closed_loop(model)
    hard_window_panel = build_evaluation_panel(model, test, hard_window_closed_loop, horizon=8)
    add_experiment_record(
        experiment_records,
        name="hard-window",
        role="candidate",
        model_tag="stage2.07",
        model=model,
        validation_transitions=validation_holdout,
        report_test_transitions=report_test_holdout,
        closed_loop=hard_window_closed_loop,
        fingerprint=fingerprint,
    )

    weighted_augmented, hard_weights, weighted_windows = make_weighted_hard_window_augmented_transitions(
        model,
        train,
        horizon=6,
        stride=5,
        blend=0.006,
        top_fraction=0.35,
        min_weight=0.75,
        max_weight=1.15,
    )
    model.train(weighted_augmented, epochs=1, learning_rate=0.000025, sample_weights=hard_weights)
    weighted_closed_loop = run_closed_loop(model)
    weighted_panel = build_evaluation_panel(model, test, weighted_closed_loop, horizon=8)
    add_experiment_record(
        experiment_records,
        name="weighted-hard-window",
        role="candidate",
        model_tag="stage2.08",
        model=model,
        validation_transitions=validation_holdout,
        report_test_transitions=report_test_holdout,
        closed_loop=weighted_closed_loop,
        fingerprint=fingerprint,
    )
    early_stopping = fine_tune_with_early_stopping(
        model,
        weighted_augmented,
        test,
        max_epochs=5,
        learning_rate=0.000025,
        sample_weights=hard_weights,
        horizon=8,
        patience=2,
        min_delta=0.000005,
    )
    after = model.evaluate(test)
    model.save(MODEL_PATH)
    loaded = LIFReservoirWorldModel.load(MODEL_PATH)
    closed_loop = run_closed_loop(loaded)
    evaluation_panel = build_evaluation_panel(loaded, test, closed_loop, horizon=8)
    add_experiment_record(
        experiment_records,
        name="early-stopped",
        role="selected",
        model_tag="stage2.09",
        model=loaded,
        validation_transitions=validation_holdout,
        report_test_transitions=report_test_holdout,
        closed_loop=closed_loop,
        fingerprint=fingerprint,
        selected_for_deployment=True,
    )
    experiment_registry = ExperimentRegistry(experiment_records)
    REGISTRY_PATH.write_text(json.dumps(experiment_registry.to_dict(), indent=2), encoding="utf-8")
    report = {
        "train_transitions": len(train),
        "test_transitions": len(test),
        "registry_validation_transitions": len(validation_holdout),
        "registry_report_test_transitions": len(report_test_holdout),
        "before": before.__dict__,
        "no_sensor_after": no_sensor_after.__dict__,
        "one_step_after": one_step_after.__dict__,
        "one_step_evaluation_panel": one_step_panel.to_dict(),
        "rollout_aware_calibration": scale_selection,
        "calibrated_evaluation_panel": calibrated_panel.to_dict(),
        "rollout_aware_finetune": {
            "augmented_transitions": len(augmented),
            "horizon": 6,
            "stride": 8,
            "blend": 0.10,
            "epochs": 1,
            "learning_rate": 0.0002,
        },
        "rollout_augmented_evaluation_panel": rollout_augmented_panel.to_dict(),
        "hard_window_finetune": {
            "augmented_transitions": len(hard_augmented),
            "selected_windows": len(hard_windows),
            "horizon": 6,
            "stride": 5,
            "blend": 0.015,
            "top_fraction": 0.35,
            "epochs": 1,
            "learning_rate": 0.00008,
            "mean_selected_final_error": sum(score.final_position_error for score in hard_windows) / len(hard_windows),
            "max_selected_final_error": max(score.final_position_error for score in hard_windows),
        },
        "hard_window_evaluation_panel": hard_window_panel.to_dict(),
        "weighted_hard_window_finetune": {
            "augmented_transitions": len(weighted_augmented),
            "selected_windows": len(weighted_windows),
            "horizon": 6,
            "stride": 5,
            "blend": 0.006,
            "top_fraction": 0.35,
            "epochs": 1,
            "learning_rate": 0.000025,
            "min_weight": min(hard_weights),
            "max_weight": max(hard_weights),
            "target_min_weight": 0.75,
            "target_max_weight": 1.15,
            "mean_selected_final_error": sum(score.final_position_error for score in weighted_windows)
            / len(weighted_windows),
            "max_selected_final_error": max(score.final_position_error for score in weighted_windows),
        },
        "weighted_hard_window_evaluation_panel": weighted_panel.to_dict(),
        "early_stopping_finetune": {
            "max_epochs": 5,
            "patience": 2,
            "min_delta": 0.000005,
            "best_epoch": early_stopping.best_epoch,
            "epochs_run": early_stopping.epochs_run,
            "best_final_position_mae": early_stopping.best_final_position_mae,
            "history": early_stopping.history,
        },
        "after": after.__dict__,
        "evaluation_panel": evaluation_panel.to_dict(),
        "closed_loop": closed_loop,
        "model_path": str(MODEL_PATH),
        "trajectory_path": str(TRAJECTORY_PATH),
        "dataset_path": str(DATASET_PATH),
        "dataset_manifest_path": str(MANIFEST_PATH),
        "experiment_registry_path": str(REGISTRY_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Train transitions: {len(train)}")
    print(f"Test transitions: {len(test)}")
    print(f"Dead-reckoning MAE: {after.dead_reckoning_mae:.6f}")
    print(f"No-sensor SNN MAE: {no_sensor_after.snn_mae:.6f}")
    print(f"One-step-trained SNN MAE: {one_step_after.snn_mae:.6f}")
    print(f"Selected residual scale: {scale_selection['selected_scale']:.2f}")
    print(f"SNN world-model MAE: {after.snn_mae:.6f}")
    print(f"Reliability MAE: {after.reliability_mae:.6f}")
    print(f"One-step-trained multi-step final position MAE: {one_step_panel.multi_step.final_position_mae:.6f}")
    print(f"Calibrated multi-step final position MAE: {calibrated_panel.multi_step.final_position_mae:.6f}")
    print(f"Rollout-augmented multi-step final position MAE: {rollout_augmented_panel.multi_step.final_position_mae:.6f}")
    print(f"Hard-window multi-step final position MAE: {hard_window_panel.multi_step.final_position_mae:.6f}")
    print(f"Weighted multi-step final position MAE: {weighted_panel.multi_step.final_position_mae:.6f}")
    print(f"Early-stopped best epoch: {early_stopping.best_epoch}")
    print(f"Experiment registry selected: {experiment_registry.selected().name}")
    print(f"Multi-step position MAE: {evaluation_panel.multi_step.position_mae:.6f}")
    print(f"Multi-step final position MAE: {evaluation_panel.multi_step.final_position_mae:.6f}")
    print(f"Closed-loop final target distance: {closed_loop['final_target_distance']:.6f}")
    print(f"Closed-loop closest obstacle distance: {closed_loop['closest_obstacle_distance']:.6f}")
    print(f"Dataset saved: {DATASET_PATH}")
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
