"""Focused checks for the channel and observation boundary; no engine imitation."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import numpy as np

from run_demo import Heuristic, burst_mask


class DemoChecks(unittest.TestCase):
    def test_channel_starts_available_and_is_reproducible(self):
        a = burst_mask(1000, 20271125)
        self.assertTrue(a[0])
        self.assertEqual(a, burst_mask(1000, 20271125))
        self.assertEqual(burst_mask(100, 3, p=0), [True]*100)

    def test_long_channel_matches_specified_transition_probabilities(self):
        flags = np.array(burst_mask(200000, 19))
        missing = ~flags
        recovery = np.mean(flags[1:][missing[:-1]])
        onset = np.mean(missing[1:][flags[:-1]])
        self.assertLess(abs(missing.mean()-.25), .015)
        self.assertLess(abs(recovery-.25), .01)
        self.assertLess(abs(onset-1/12), .005)

    def test_outage_never_updates_target_position_velocity_or_frame(self):
        policy = Heuristic([0, 0])
        label = SimpleNamespace(object_position_x=10, object_position_y=20, x=1, y=2, width=3, height=4)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        policy.observe(delivered=True, frame=frame, label=label, tic=16)
        hidden = SimpleNamespace(object_position_x=999, object_position_y=999)
        policy.observe(delivered=False, frame=None, label=hidden, tic=20)
        np.testing.assert_array_equal(policy.position, [10, 20])
        np.testing.assert_array_equal(policy.velocity_estimate, [0, 0])
        self.assertEqual(policy.last_frame_tic, 16)
        self.assertEqual(policy.last_detection_tic, 16)
        self.assertIsNone(policy.box)
        np.testing.assert_array_equal(policy.delivered_frame, frame)

    def test_delivered_without_target_differs_from_outage(self):
        policy = Heuristic([0, 0])
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        label = SimpleNamespace(object_position_x=10, object_position_y=20, x=1, y=2, width=3, height=4)
        policy.observe(delivered=True, frame=frame, label=label, tic=16)
        policy.observe(delivered=True, frame=frame+1, label=None, tic=20)
        self.assertEqual(policy.last_frame_tic, 20)
        self.assertEqual(policy.last_detection_tic, 16)
        np.testing.assert_array_equal(policy.position, [10, 20])
        np.testing.assert_array_equal(policy.delivered_frame, frame+1)


if __name__ == "__main__":
    unittest.main()
