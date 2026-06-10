import unittest

from snn_world_model import (
    ExperimentRecord,
    ExperimentRegistry,
    LIFReservoirWorldModel,
    SimConfig,
    build_evaluation_panel,
    evaluate_multistep,
    generate_transitions,
    select_residual_scale,
)


class EvaluationTests(unittest.TestCase):
    def test_multistep_evaluation_returns_rollout_metrics(self):
        transitions = generate_transitions(SimConfig(seed=17, steps=160))
        train = transitions[:120]
        test = transitions[120:]
        model = LIFReservoirWorldModel(reservoir_size=18, seed=12)
        model.train(train, epochs=6, learning_rate=0.012)
        metrics = evaluate_multistep(model, test, horizon=6, stride=3)
        self.assertEqual(metrics.horizon, 6)
        self.assertGreater(metrics.windows, 0)
        self.assertGreater(metrics.position_mae, 0.0)
        self.assertGreater(metrics.final_position_mae, 0.0)

    def test_evaluation_panel_serializes_three_metric_groups(self):
        transitions = generate_transitions(SimConfig(seed=19, steps=180))
        train = transitions[:130]
        test = transitions[130:]
        model = LIFReservoirWorldModel(reservoir_size=18, seed=14)
        model.train(train, epochs=6, learning_rate=0.012)
        panel = build_evaluation_panel(
            model,
            test,
            {"final_target_distance": 0.1, "closest_obstacle_distance": 0.5, "safety_overrides": 2},
            horizon=6,
        )
        payload = panel.to_dict()
        self.assertIn("one_step", payload)
        self.assertIn("multi_step", payload)
        self.assertIn("closed_loop", payload)

    def test_residual_scale_selection_sets_model_scale(self):
        transitions = generate_transitions(SimConfig(seed=21, steps=180))
        train = transitions[:130]
        test = transitions[130:]
        model = LIFReservoirWorldModel(reservoir_size=18, seed=16)
        model.train(train, epochs=6, learning_rate=0.012)
        selection = select_residual_scale(model, test, candidates=(0.0, 0.8, 1.0), horizon=6)
        self.assertIn(selection["selected_scale"], (0.0, 0.8, 1.0))
        self.assertEqual(model.residual_scale, selection["selected_scale"])
        self.assertEqual(len(selection["scores"]), 3)

    def test_experiment_registry_selects_and_ranks_records(self):
        records = [
            ExperimentRecord(
                name="baseline",
                role="candidate",
                dataset_fingerprint="abc123",
                model_tag="m0",
                one_step_snn_mae=0.004,
                validation_final_position_mae=0.42,
                test_final_position_mae=0.44,
                closed_loop_final_target_distance=0.16,
            ),
            ExperimentRecord(
                name="early-stopped",
                role="deploy",
                dataset_fingerprint="abc123",
                model_tag="m1",
                one_step_snn_mae=0.003,
                validation_final_position_mae=0.32,
                test_final_position_mae=0.35,
                closed_loop_final_target_distance=0.11,
                selected_for_deployment=True,
            ),
        ]
        registry = ExperimentRegistry(records)
        payload = registry.to_dict()
        self.assertEqual(registry.selected().name, "early-stopped")
        self.assertEqual(payload["selected_model_tag"], "m1")
        self.assertEqual(payload["validation_ranking"], ["early-stopped", "baseline"])


if __name__ == "__main__":
    unittest.main()
