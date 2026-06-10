import unittest

from snn_world_model.neurons import LIFParams, LIFState, lif_step, simulate_lif


class LIFTests(unittest.TestCase):
    def test_lif_spikes_and_resets(self):
        state = LIFState(0.99)
        next_state, spike = lif_step(state, input_current=10.0, dt_ms=1.0, params=LIFParams())
        self.assertEqual(spike, 1)
        self.assertEqual(next_state.voltage, 0.0)

    def test_simulate_lif_produces_spikes_for_sustained_current(self):
        voltages, spikes = simulate_lif([1.25] * 120, dt_ms=1.0)
        self.assertEqual(len(voltages), 120)
        self.assertGreater(sum(spikes), 0)


if __name__ == "__main__":
    unittest.main()
