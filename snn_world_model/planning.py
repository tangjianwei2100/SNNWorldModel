"""Candidate rollout planner and hard safety shield."""

from dataclasses import dataclass
from math import atan2, hypot
from typing import Callable, Iterable

from .robot import DifferentialDriveParams, RobotState, diff_drive_step, wrap_angle
from .types import Action, Point

WorldModel = Callable[[RobotState, Action], RobotState]


@dataclass(frozen=True)
class CircularObstacle:
    center: Point
    radius: float

    def distance_from(self, state: RobotState) -> float:
        return hypot(state.x - self.center.x, state.y - self.center.y)

    def clearance_from(self, state: RobotState) -> float:
        return self.distance_from(state) - self.radius


@dataclass(frozen=True)
class PlanningProblem:
    target: Point
    obstacle: CircularObstacle
    hard_safety_radius: float
    obstacle_cost_radius: float | None = None
    horizon: int = 7
    target_weight: float = 1.0
    heading_weight: float = 0.22
    obstacle_weight: float = 1.0
    smoothness_weight: float = 0.42
    effort_weight: float = 0.04

    def cost_radius(self) -> float:
        return self.obstacle.radius if self.obstacle_cost_radius is None else self.obstacle_cost_radius


@dataclass(frozen=True)
class RolloutResult:
    action: Action
    cost: float
    endpoint: RobotState
    min_obstacle_distance: float
    trajectory: tuple[RobotState, ...]


def default_world_model(params: DifferentialDriveParams | None = None) -> WorldModel:
    params = params or DifferentialDriveParams()
    return lambda state, action: diff_drive_step(state, action, params)


def target_distance(state: RobotState, target: Point) -> float:
    return hypot(target.x - state.x, target.y - state.y)


def heading_error(state: RobotState, target: Point) -> float:
    desired = atan2(target.y - state.y, target.x - state.x)
    return abs(wrap_angle(state.theta - desired))


def obstacle_cost(distance: float, obstacle_radius: float, safety_radius: float) -> float:
    if distance >= safety_radius:
        return 0.10 / max(distance - safety_radius + 0.08, 0.08)
    if distance >= obstacle_radius:
        return 8.0 + 28.0 * (safety_radius - distance)
    return 80.0 + 90.0 * (obstacle_radius - distance)


class CandidatePlanner:
    """Roll out a small set of candidate actions and choose the lowest-cost one."""

    def __init__(self, candidates: Iterable[Action], problem: PlanningProblem, world_model: WorldModel | None = None):
        self.candidates = tuple(candidates)
        self.problem = problem
        self.world_model = world_model or default_world_model()

    def rollout(self, state: RobotState, action: Action, previous_action: Action) -> RolloutResult:
        current = state
        trajectory = []
        cost = 0.0
        min_obstacle_distance = float("inf")
        for h in range(self.problem.horizon):
            current = self.world_model(current, action)
            trajectory.append(current)
            distance = target_distance(current, self.problem.target)
            obstacle_distance = self.problem.obstacle.distance_from(current)
            min_obstacle_distance = min(min_obstacle_distance, obstacle_distance)
            cost += self.problem.target_weight * distance * (1.0 + 0.08 * h)
            cost += self.problem.heading_weight * heading_error(current, self.problem.target)
            cost += self.problem.obstacle_weight * obstacle_cost(
                obstacle_distance,
                self.problem.obstacle.radius,
                self.problem.cost_radius(),
            )
        cost += self.problem.smoothness_weight * action.smoothness_from(previous_action)
        cost += self.problem.effort_weight * action.effort()
        return RolloutResult(action, cost, current, min_obstacle_distance, tuple(trajectory))

    def choose(self, state: RobotState, previous_action: Action) -> RolloutResult:
        return min(
            (self.rollout(state, action, previous_action) for action in self.candidates),
            key=lambda result: result.cost,
        )


class SafetyShield:
    """Enforce hard clearance before action execution."""

    def __init__(self, planner: CandidatePlanner):
        self.planner = planner

    def choose(self, state: RobotState, previous_action: Action) -> tuple[RolloutResult, bool]:
        nominal = self.planner.choose(state, previous_action)
        if nominal.min_obstacle_distance >= self.planner.problem.hard_safety_radius:
            return nominal, False
        rollouts = [self.planner.rollout(state, action, previous_action) for action in self.planner.candidates]
        safe = [result for result in rollouts if result.min_obstacle_distance >= self.planner.problem.hard_safety_radius]
        if safe:
            return min(safe, key=lambda result: result.cost), True
        return max(rollouts, key=lambda result: result.min_obstacle_distance), True
