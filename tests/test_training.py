import unittest

from snn_world_model import (
    LIFReservoirWorldModel,
    SimConfig,
    fine_tune_with_early_stopping,
    generate_transitions,
    make_hard_window_augmented_transitions,
    make_rollout_augmented_transitions,
    make_weighted_hard_window_augmented_transitions,
    score_rollout_windows,
)


class TrainingHelperTests(unittest.TestCase):
    def test_rollout_augmented_transitions_are_generated(self):
        transitions = generate_transitions(SimConfig(seed=31, steps=80))
        model = LIFReservoirWorldModel(reservoir_size=12, seed=3)
        model.train(transitions[:50], epochs=3, learning_rate=0.01)
        augmented = make_rollout_augmented_transitions(model, transitions[:60], horizon=5, stride=5, blend=0.1)
        self.assertGreater(len(augmented), 0)
        self.assertEqual(augmented[0].action, transitions[0].action)
        self.assertTrue(any(sample.state != transitions[index].state for index, sample in enumerate(augmented[:10])))

    def test_hard_window_augmentation_selects_high_error_windows(self):
        transitions = generate_transitions(SimConfig(seed=32, steps=90))
        model = LIFReservoirWorldModel(reservoir_size=12, seed=4)
        model.train(transitions[:55], epochs=3, learning_rate=0.01)
        scores = score_rollout_windows(model, transitions[:70], horizon=5, stride=5)
        augmented, selected = make_hard_window_augmented_transitions(
            model,
            transitions[:70],
            horizon=5,
            stride=5,
            blend=0.08,
            top_fraction=0.4,
        )
        self.assertGreater(len(scores), 0)
        self.assertEqual(len(augmented), len(selected) * 5)
        cutoff = sorted((score.final_position_error for score in scores), reverse=True)[len(selected) - 1]
        self.assertTrue(all(score.final_position_error >= cutoff for score in selected))

    def test_weighted_hard_window_samples_align_weights(self):
        transitions = generate_transitions(SimConfig(seed=33, steps=90))
        model = LIFReservoirWorldModel(reservoir_size=12, seed=5)
        model.train(transitions[:55], epochs=3, learning_rate=0.01)
        augmented, weights, selected = make_weighted_hard_window_augmented_transitions(
            model,
            transitions[:70],
            horizon=5,
            stride=5,
            blend=0.05,
            top_fraction=0.4,
            min_weight=0.7,
            max_weight=1.4,
        )
        self.assertEqual(len(augmented), len(weights))
        self.assertEqual(len(augmented), len(selected) * 5)
        self.assertGreaterEqual(min(weights), 0.7)
        self.assertLessEqual(max(weights), 1.4)
        with self.assertRaises(ValueError):
            model.train(augmented, epochs=1, learning_rate=0.001, sample_weights=weights[:-1])

    def test_early_stopping_records_history_and_best_epoch(self):
        transitions = generate_transitions(SimConfig(seed=34, steps=110))
        train = transitions[:70]
        validation = transitions[70:95]
        model = LIFReservoirWorldModel(reservoir_size=12, seed=6)
        model.train(train, epochs=3, learning_rate=0.01)
        augmented = make_rollout_augmented_transitions(model, train, horizon=5, stride=5, blend=0.05)
        result = fine_tune_with_early_stopping(
            model,
            augmented,
            validation,
            max_epochs=3,
            learning_rate=0.0001,
            horizon=5,
            patience=2,
        )
        self.assertGreaterEqual(result.best_epoch, 0)
        self.assertGreaterEqual(result.epochs_run, result.best_epoch)
        self.assertGreaterEqual(len(result.history), 2)
        self.assertLessEqual(result.epochs_run, 3)
        with self.assertRaises(ValueError):
            fine_tune_with_early_stopping(model, augmented, validation, max_epochs=0)


if __name__ == "__main__":
    unittest.main()
