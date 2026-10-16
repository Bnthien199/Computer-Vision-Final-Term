"""Proven runtime regressions. Synthetic cases are not rep/timer benchmarks."""
import unittest
from unittest.mock import patch
import numpy as np
import mediapipe as mp
from posefit_runtime import TimestampedPose, Tracker


class VideoRuntimeTests(unittest.TestCase):
    def test_cancelled_action_drops_angle_history(self):
        for exercise, rejected_label in [('squat', 'pushup'), ('pushup', 'squat')]:
            with self.subTest(exercise=exercise):
                tracker = Tracker()
                def body(angle):
                    return {exercise: dict(angle=angle, posture=True, side=11)}
                tracker.update(body(170), exercise, 0)
                tracker.update(body(170), exercise, .11)
                counter = getattr(tracker, exercise)
                self.assertEqual(counter.state, 'UP')
                tracker.update(body(149.9), rejected_label, .12)
                self.assertEqual(counter.state, 'READY')
                self.assertNotIn(exercise, tracker.angles)
                for now in [.13, .17, .21, .25]:
                    tracker.update(body(149.9), exercise, now)
                self.assertEqual(counter.state, 'READY')
                self.assertEqual(counter.count, 0)
                self.assertAlmostEqual(tracker.angles[exercise], 149.9)
                # A bottom -> top after cancellation is still a partial cycle;
                # the stale angle must not have armed a top beforehand.
                for now, angle in [(.29, 80), (.40, 80), (.48, 80),
                                   (.60, 170), (.72, 170), (.84, 170), (.95, 170)]:
                    tracker.update(body(angle), exercise, now)
                self.assertEqual(counter.count, 0)

    def test_real_graph_packet_timestamps_and_reset(self):
        # Intercept packets entering the actual native graph, not just the
        # Python clock variable. No image labels or benchmark counts involved.
        pose = TimestampedPose(model_complexity=1, smooth_landmarks=True)
        packets = []
        original = pose._graph
        class ObservedGraph:
            def __getattr__(self, name): return getattr(original, name)
            def add_packet_to_input_stream(self, *, stream, packet):
                packets.append(packet.timestamp.value)
                return original.add_packet_to_input_stream(stream=stream, packet=packet)
        pose._graph = ObservedGraph()
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        try:
            for now in [0, 1 / 60, 2 / 60, .10, .14]:
                pose.process(image, now)
            self.assertEqual(packets, [0, 16667, 33333, 100000, 140000])
            for now in [.14, .13, float('nan'), float('inf'), -1, .1400001]:
                with self.assertRaises(ValueError): pose.process(image, now)
            self.assertEqual(len(packets), 5)
            pose.reset()
            pose.process(image, 0)
            pose.process(image, 1 / 25)
            self.assertEqual(packets[-2:], [0, 40000])
        finally:
            pose.close()

    def test_unverified_mediapipe_version_rejected(self):
        with patch.object(mp, '__version__', 'unverified'):
            with self.assertRaisesRegex(RuntimeError, '0.10.21'):
                TimestampedPose()


if __name__ == '__main__':
    unittest.main()
