import unittest

from snn_world_model import Action, CandidatePlanner, CircularObstacle, PlanningProblem, Point, RobotState, SafetyShield


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.problem = PlanningProblem(
            target=Point(2.85, 1.45),
            obstacle=CircularObstacle(Point(0.92, 0.42), 0.36),
            hard_safety_radius=0.48,
        )
        self.candidates = [
            Action(0.72, 0.72),
            Action(0.56, 0.80),
            Action(0.80, 0.56),
            Action(0.35, 0.35),
            Action(-0.20, 0.45),
            Action(0.45, -0.20),
        ]

    def test_planner_returns_candidate(self):
        planner = CandidatePlanner(self.candidates, self.problem)
        result = planner.choose(RobotState(-0.55, -0.20, 0.10), Action(0.35, 0.35))
        self.assertIn(result.action, self.candidates)
        self.assertGreater(result.cost, 0.0)

    def test_safety_shield_returns_override_flag(self):
        planner = CandidatePlanner(self.candidates, self.problem)
        shield = SafetyShield(planner)
        result, overridden = shield.choose(RobotState(0.50, 0.20, 0.10), Action(0.72, 0.72))
        self.assertIn(result.action, self.candidates)
        self.assertIsInstance(overridden, bool)


if __name__ == "__main__":
    unittest.main()
