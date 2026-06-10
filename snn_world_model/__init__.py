"""Reusable SNN world-model engineering primitives."""

from .neurons import LIFParams, LIFState, lif_step
from .environment import SensorResidual, SimConfig, SlipRobotEnvironment, Transition, generate_transitions
from .datasets import (
    DatasetRegistry,
    DatasetVersion,
    read_dataset_registry,
    read_transition_csv,
    write_dataset_manifest,
    write_dataset_registry,
    write_transition_csv,
)
from .evaluation import (
    EvaluationPanel,
    ExperimentRecord,
    ExperimentRegistry,
    MultiStepMetrics,
    build_evaluation_panel,
    evaluate_multistep,
    select_residual_scale,
)
from .models import LIFReservoirWorldModel, WorldModelMetrics
from .planning import CandidatePlanner, CircularObstacle, PlanningProblem, SafetyShield
from .robot import DifferentialDriveParams, RobotState, diff_drive_step
from .training import (
    EarlyStoppingResult,
    RolloutWindowScore,
    blend_state,
    fine_tune_with_early_stopping,
    make_hard_window_augmented_transitions,
    make_rollout_augmented_transitions,
    make_weighted_hard_window_augmented_transitions,
    score_rollout_windows,
)
from .types import Action, Point

__all__ = [
    "Action",
    "CandidatePlanner",
    "CircularObstacle",
    "DifferentialDriveParams",
    "DatasetRegistry",
    "DatasetVersion",
    "EvaluationPanel",
    "ExperimentRecord",
    "ExperimentRegistry",
    "LIFParams",
    "LIFReservoirWorldModel",
    "LIFState",
    "MultiStepMetrics",
    "PlanningProblem",
    "Point",
    "RobotState",
    "SafetyShield",
    "SensorResidual",
    "SimConfig",
    "SlipRobotEnvironment",
    "Transition",
    "WorldModelMetrics",
    "diff_drive_step",
    "generate_transitions",
    "lif_step",
    "read_dataset_registry",
    "read_transition_csv",
    "write_dataset_manifest",
    "write_dataset_registry",
    "write_transition_csv",
    "build_evaluation_panel",
    "EarlyStoppingResult",
    "RolloutWindowScore",
    "blend_state",
    "evaluate_multistep",
    "fine_tune_with_early_stopping",
    "make_hard_window_augmented_transitions",
    "make_rollout_augmented_transitions",
    "make_weighted_hard_window_augmented_transitions",
    "select_residual_scale",
    "score_rollout_windows",
]
