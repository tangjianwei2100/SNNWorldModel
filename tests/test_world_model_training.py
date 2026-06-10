import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from snn_world_model import LIFReservoirWorldModel, SensorResidual, SimConfig, generate_transitions


class WorldModelTrainingTests(unittest.TestCase):
    def test_training_improves_snn_mae_and_model_round_trips(self):
        transitions = generate_transitions(SimConfig(seed=41, steps=260))
        train = transitions[:200]
        test = transitions[200:]
        model = LIFReservoirWorldModel(reservoir_size=18, seed=5)
        before = model.evaluate(test)
        model.train(train, epochs=10, learning_rate=0.012)
        after = model.evaluate(test)
        self.assertLess(after.snn_mae, before.snn_mae)
        self.assertLess(after.reliability_mae, before.reliability_mae)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            loaded = LIFReservoirWorldModel.load(path)
            state = test[0].state
            action = test[0].action
            self.assertEqual(loaded.predict_next(state, action), model.predict_next(state, action))

    def test_sensor_residuals_improve_reliability_estimate(self):
        transitions = generate_transitions(SimConfig(seed=13, steps=420))
        train = transitions[:320]
        test = transitions[320:]
        train_no_sensor = [replace(t, sensor_residual=SensorResidual()) for t in train]
        test_no_sensor = [replace(t, sensor_residual=SensorResidual()) for t in test]

        no_sensor_model = LIFReservoirWorldModel(reservoir_size=20, seed=9)
        no_sensor_model.train(train_no_sensor, epochs=8, learning_rate=0.012)
        sensor_model = LIFReservoirWorldModel(reservoir_size=20, seed=9)
        sensor_model.train(train, epochs=8, learning_rate=0.012)

        self.assertLess(sensor_model.evaluate(test).reliability_mae, no_sensor_model.evaluate(test_no_sensor).reliability_mae)


if __name__ == "__main__":
    unittest.main()
