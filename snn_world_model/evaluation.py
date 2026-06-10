"""Evaluation panels for one-step, multi-step, and closed-loop metrics."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import hypot

from .environment import SensorResidual, Transition
from .models import LIFReservoirWorldModel, WorldModelMetrics
from .robot import RobotState, wrap_angle


@dataclass(frozen=True)
class MultiStepMetrics:
    horizon: int
    windows: int
    position_mae: float
    heading_mae: float
    final_position_mae: float


@dataclass(frozen=True)
class EvaluationPanel:
    one_step: WorldModelMetrics
    multi_step: MultiStepMetrics
    closed_loop: dict

    def to_dict(self) -> dict:
        return {
            "one_step": asdict(self.one_step),
            "multi_step": asdict(self.multi_step),
            "closed_loop": self.closed_loop,
        }


@dataclass(frozen=True)
class ExperimentRecord:
    name: str
    role: str
    dataset_fingerprint: str
    model_tag: str
    one_step_snn_mae: float
    validation_final_position_mae: float
    test_final_position_mae: float
    closed_loop_final_target_distance: float
    selected_for_deployment: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentRegistry:
    records: list[ExperimentRecord]

    def selected(self) -> ExperimentRecord:
        selected_records = [record for record in self.records if record.selected_for_deployment]
        if len(selected_records) != 1:
            raise ValueError("experiment registry must contain exactly one selected record")
        return selected_records[0]

    def ranked_by_validation(self) -> list[ExperimentRecord]:
        return sorted(self.records, key=lambda record: record.validation_final_position_mae)

    def to_dict(self) -> dict:
        selected = self.selected()
        return {
            "selected_model_tag": selected.model_tag,
            "selected_name": selected.name,
            "records": [record.to_dict() for record in self.records],
            "validation_ranking": [record.name for record in self.ranked_by_validation()],
        }


def position_error(predicted: RobotState, target: RobotState) -> float:
    return hypot(predicted.x - target.x, predicted.y - target.y)


def evaluate_multistep(
    model: LIFReservoirWorldModel,
    transitions: list[Transition],
    horizon: int = 8,
    stride: int = 5,
) -> MultiStepMetrics:
    """Roll the learned model forward across logged action windows."""
    if len(transitions) < horizon + 1:
        raise ValueError("not enough transitions for multi-step evaluation")
    position_error_sum = 0.0
    heading_error_sum = 0.0
    final_position_error_sum = 0.0
    sample_count = 0
    windows = 0
    for start in range(0, len(transitions) - horizon, stride):
        predicted = transitions[start].state
        sensor_residual = transitions[start].sensor_residual
        windows += 1
        for offset in range(horizon):
            transition = transitions[start + offset]
            predicted = model.predict_next(predicted, transition.action, sensor_residual)
            target = transition.next_state
            position_error_sum += position_error(predicted, target)
            heading_error_sum += abs(wrap_angle(predicted.theta - target.theta))
            sample_count += 1
            sensor_residual = transition.sensor_residual
        final_position_error_sum += position_error(predicted, transitions[start + horizon - 1].next_state)
    return MultiStepMetrics(
        horizon=horizon,
        windows=windows,
        position_mae=position_error_sum / sample_count,
        heading_mae=heading_error_sum / sample_count,
        final_position_mae=final_position_error_sum / windows,
    )


def build_evaluation_panel(
    model: LIFReservoirWorldModel,
    test_transitions: list[Transition],
    closed_loop: dict,
    horizon: int = 8,
) -> EvaluationPanel:
    return EvaluationPanel(
        one_step=model.evaluate(test_transitions),
        multi_step=evaluate_multistep(model, test_transitions, horizon=horizon),
        closed_loop=closed_loop,
    )


def select_residual_scale(
    model: LIFReservoirWorldModel,
    validation_transitions: list[Transition],
    candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.65, 0.8, 1.0, 1.2),
    horizon: int = 8,
) -> dict:
    """Choose a residual scale using multi-step final position error."""
    original = model.residual_scale
    scores = []
    for scale in candidates:
        model.residual_scale = scale
        metrics = evaluate_multistep(model, validation_transitions, horizon=horizon)
        scores.append(
            {
                "scale": scale,
                "multi_step_position_mae": metrics.position_mae,
                "multi_step_final_position_mae": metrics.final_position_mae,
            }
        )
    best = min(scores, key=lambda item: item["multi_step_final_position_mae"])
    model.residual_scale = best["scale"]
    return {
        "original_scale": original,
        "selected_scale": best["scale"],
        "scores": scores,
    }
