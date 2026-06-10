"""Training helpers for rollout-aware data augmentation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace
from math import hypot

from .environment import Transition
from .evaluation import evaluate_multistep
from .models import LIFReservoirWorldModel
from .robot import RobotState, wrap_angle


@dataclass(frozen=True)
class RolloutWindowScore:
    start: int
    mean_position_error: float
    final_position_error: float


@dataclass(frozen=True)
class EarlyStoppingResult:
    best_epoch: int
    epochs_run: int
    best_final_position_mae: float
    history: list[dict[str, float | int]]


def blend_state(true_state: RobotState, predicted_state: RobotState, blend: float) -> RobotState:
    return RobotState(
        true_state.x * (1.0 - blend) + predicted_state.x * blend,
        true_state.y * (1.0 - blend) + predicted_state.y * blend,
        wrap_angle(true_state.theta * (1.0 - blend) + predicted_state.theta * blend),
    )


def score_rollout_windows(
    model: LIFReservoirWorldModel,
    transitions: list[Transition],
    horizon: int = 8,
    stride: int = 5,
) -> list[RolloutWindowScore]:
    """Score logged windows by how much the model drifts during rollout."""
    if len(transitions) < horizon + 1:
        raise ValueError("not enough transitions for rollout window scoring")
    scores: list[RolloutWindowScore] = []
    for start in range(0, len(transitions) - horizon, stride):
        predicted = transitions[start].state
        error_sum = 0.0
        for offset in range(horizon):
            transition = transitions[start + offset]
            predicted = model.predict_next(predicted, transition.action, transition.sensor_residual)
            target = transition.next_state
            error_sum += hypot(predicted.x - target.x, predicted.y - target.y)
        final_target = transitions[start + horizon - 1].next_state
        scores.append(
            RolloutWindowScore(
                start=start,
                mean_position_error=error_sum / horizon,
                final_position_error=hypot(predicted.x - final_target.x, predicted.y - final_target.y),
            )
        )
    return scores


def make_rollout_augmented_transitions(
    model: LIFReservoirWorldModel,
    transitions: list[Transition],
    horizon: int = 6,
    stride: int = 8,
    blend: float = 0.10,
) -> list[Transition]:
    """Create short-window samples from states nudged toward model rollout."""
    if len(transitions) < horizon + 1:
        raise ValueError("not enough transitions for rollout augmentation")
    augmented: list[Transition] = []
    for start in range(0, len(transitions) - horizon, stride):
        predicted = transitions[start].state
        for offset in range(horizon):
            transition = transitions[start + offset]
            mixed_state = blend_state(transition.state, predicted, blend)
            augmented.append(replace(transition, state=mixed_state))
            predicted = model.predict_next(mixed_state, transition.action, transition.sensor_residual)
    return augmented


def make_hard_window_augmented_transitions(
    model: LIFReservoirWorldModel,
    transitions: list[Transition],
    horizon: int = 6,
    stride: int = 5,
    blend: float = 0.08,
    top_fraction: float = 0.35,
) -> tuple[list[Transition], list[RolloutWindowScore]]:
    """Create augmented samples only from the highest-drift rollout windows."""
    if top_fraction <= 0.0 or top_fraction > 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    scores = score_rollout_windows(model, transitions, horizon=horizon, stride=stride)
    selected_count = max(1, int(len(scores) * top_fraction))
    selected = sorted(scores, key=lambda item: item.final_position_error, reverse=True)[:selected_count]
    selected_starts = {score.start for score in selected}
    augmented: list[Transition] = []
    for score in scores:
        if score.start not in selected_starts:
            continue
        predicted = transitions[score.start].state
        for offset in range(horizon):
            transition = transitions[score.start + offset]
            mixed_state = blend_state(transition.state, predicted, blend)
            augmented.append(replace(transition, state=mixed_state))
            predicted = model.predict_next(mixed_state, transition.action, transition.sensor_residual)
    return augmented, selected


def make_weighted_hard_window_augmented_transitions(
    model: LIFReservoirWorldModel,
    transitions: list[Transition],
    horizon: int = 6,
    stride: int = 5,
    blend: float = 0.012,
    top_fraction: float = 0.35,
    min_weight: float = 0.75,
    max_weight: float = 1.45,
) -> tuple[list[Transition], list[float], list[RolloutWindowScore]]:
    """Create hard-window samples with weights proportional to rollout drift."""
    if min_weight <= 0.0 or max_weight < min_weight:
        raise ValueError("weights must satisfy 0 < min_weight <= max_weight")
    augmented, selected = make_hard_window_augmented_transitions(
        model,
        transitions,
        horizon=horizon,
        stride=stride,
        blend=blend,
        top_fraction=top_fraction,
    )
    max_error = max(score.final_position_error for score in selected)
    weights: list[float] = []
    for score in selected:
        if max_error <= 0.0:
            weight = min_weight
        else:
            weight = min_weight + (max_weight - min_weight) * (score.final_position_error / max_error)
        weights.extend([weight] * horizon)
    return augmented, weights, selected


def fine_tune_with_early_stopping(
    model: LIFReservoirWorldModel,
    transitions: list[Transition],
    validation_transitions: list[Transition],
    max_epochs: int = 5,
    learning_rate: float = 0.000025,
    sample_weights: list[float] | None = None,
    horizon: int = 8,
    patience: int = 2,
    min_delta: float = 0.0,
) -> EarlyStoppingResult:
    """Fine-tune one epoch at a time and restore the best rollout model."""
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    best_readout = deepcopy(model.readout)
    best_epoch = 0
    best_score = evaluate_multistep(model, validation_transitions, horizon=horizon).final_position_mae
    history: list[dict[str, float | int]] = [
        {"epoch": 0, "final_position_mae": best_score},
    ]
    stale_epochs = 0
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        model.train(
            transitions,
            epochs=1,
            learning_rate=learning_rate,
            sample_weights=sample_weights,
        )
        score = evaluate_multistep(model, validation_transitions, horizon=horizon).final_position_mae
        history.append({"epoch": epoch, "final_position_mae": score})
        epochs_run = epoch
        if score < best_score - min_delta:
            best_score = score
            best_epoch = epoch
            best_readout = deepcopy(model.readout)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.readout = best_readout
    return EarlyStoppingResult(
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        best_final_position_mae=best_score,
        history=history,
    )
