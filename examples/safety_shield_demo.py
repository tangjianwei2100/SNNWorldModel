"""Run a small planner + safety shield demo."""

from math import hypot
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snn_world_model import (  # noqa: E402
    Action,
    CandidatePlanner,
    CircularObstacle,
    PlanningProblem,
    Point,
    RobotState,
    SafetyShield,
    diff_drive_step,
)


def simulate(use_shield: bool):
    candidates = [
        Action(0.72, 0.72),
        Action(0.56, 0.80),
        Action(0.80, 0.56),
        Action(0.42, 0.76),
        Action(0.76, 0.42),
        Action(0.35, 0.35),
        Action(-0.20, 0.45),
        Action(0.45, -0.20),
        Action(-0.32, 0.58),
        Action(0.58, -0.32),
    ]
    problem = PlanningProblem(
        target=Point(2.85, 1.45),
        obstacle=CircularObstacle(Point(0.92, 0.42), 0.36),
        hard_safety_radius=0.48,
    )
    planner = CandidatePlanner(candidates, problem)
    shield = SafetyShield(planner)
    state = RobotState(-0.55, -0.20, 0.10)
    previous = Action(0.35, 0.35)
    trajectory = [state]
    overrides = 0
    for _ in range(120):
        if use_shield:
            result, overridden = shield.choose(state, previous)
            overrides += int(overridden)
        else:
            result = planner.choose(state, previous)
        state = diff_drive_step(state, result.action)
        previous = result.action
        trajectory.append(state)
    return trajectory, overrides


def final_distance(trajectory):
    target = Point(2.85, 1.45)
    return hypot(trajectory[-1].x - target.x, trajectory[-1].y - target.y)


def closest_obstacle_distance(trajectory):
    obstacle = Point(0.92, 0.42)
    return min(hypot(state.x - obstacle.x, state.y - obstacle.y) for state in trajectory)


def main():
    unshielded, _ = simulate(use_shield=False)
    shielded, overrides = simulate(use_shield=True)
    print(f"Unshielded final target distance: {final_distance(unshielded):.6f}")
    print(f"Shielded final target distance: {final_distance(shielded):.6f}")
    print(f"Unshielded closest obstacle distance: {closest_obstacle_distance(unshielded):.6f}")
    print(f"Shielded closest obstacle distance: {closest_obstacle_distance(shielded):.6f}")
    print(f"Safety overrides: {overrides}")


if __name__ == "__main__":
    main()
