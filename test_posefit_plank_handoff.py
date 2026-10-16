"""Plank confirmation must release stale DOWN without cancelling brief label noise."""
import json
from pathlib import Path
import unittest

from posefit_runtime import Tracker


def body(angle=65, posture=True, signature=None):
    return dict(pushup=dict(angle=angle, side=12, posture=True),
                plank=dict(angle=170, side=12, posture=posture,
                           signature=[.6, .7, .6, .9, .4, .7, .2, .7]
                           if signature is None else signature))


class PlankHandoffTests(unittest.TestCase):
    def arm_down(self):
        tracker = Tracker()
        for i in range(11):
            tracker.update(body(175 if i < 5 else 65), 'pushup', i / 10)
        self.assertEqual(tracker.pushup.state, 'DOWN')
        return tracker

    @unittest.skipUnless((Path(__file__).parent / 'runs/plank_trace_session_01/trace.jsonl').is_file(),
                         'Optional historical replay requires a private webcam trace')
    def test_recorded_trace_starts_plank(self):
        tracker = Tracker()
        trace = Path(__file__).parent / 'runs/plank_trace_session_01/trace.jsonl'
        first_hold = first_increment = None
        with trace.open(encoding='utf8') as stream:
            for line in stream:
                row = json.loads(line)
                tracker.update(row['body'], row['classifier_label'], row['timestamp_seconds'])
                if tracker.plank.active and first_hold is None:
                    first_hold = row['frame']
                    self.assertEqual(tracker.pushup.state, 'READY')
                if tracker.plank.total > 0 and first_increment is None:
                    first_increment = row['frame']
                if row['frame'] == 402:
                    self.assertEqual(tracker.plank.total, 0)
        self.assertIsNotNone(first_hold)
        self.assertTrue(907 < first_hold < first_increment <= 1048)
        self.assertGreater(tracker.plank.total, 0)
        self.assertEqual(tracker.pushup.count, 0)

    def test_brief_plank_noise_preserves_down_and_rep(self):
        tracker = self.arm_down()
        started = tracker.pushup.started
        for i in range(11, 14):
            display = tracker.update(body(), 'plank', i / 10)
            self.assertEqual(display, 'pushup')
            self.assertEqual(tracker.pushup.state, 'DOWN')
            self.assertEqual(tracker.pushup.started, started)
            self.assertFalse(tracker.plank.active)
        for i in range(14, 18):
            tracker.update(body(), 'pushup', i / 10)
        for i in range(18, 24):
            tracker.update(body(175), 'pushup', i / 10)
        self.assertEqual(tracker.pushup.count, 1)
        self.assertEqual(tracker.plank.total, 0)

    def test_real_pushups_complete_with_plank_labels_at_top(self):
        # Include a long top hold: confirmed plank must preserve an armed UP.
        for cycle_frames in (16, 20):
            with self.subTest(cycle_frames=cycle_frames):
                tracker = Tracker()
                for i in range(cycle_frames * 3):
                    down = 5 <= i % cycle_frames < 12
                    tracker.update(body(65 if down else 175), 'pushup' if down else 'plank', i / 10)
                self.assertEqual(tracker.pushup.count, 3)
                if cycle_frames == 16:
                    self.assertEqual(tracker.plank.total, 0)

    def test_invalid_posture_geometry_or_signature_cannot_confirm(self):
        for variant in ('posture', 'geometry', 'signature'):
            with self.subTest(variant=variant):
                tracker = self.arm_down()
                for i in range(11, 41):
                    sample = body(posture=variant != 'posture')
                    if variant == 'geometry':
                        sample['plank'] = None
                    elif variant == 'signature':
                        sample['plank']['signature'] = [float('nan')] * 8
                    tracker.update(sample, 'plank', i / 10)
                self.assertEqual(tracker.pushup.state, 'DOWN')
                self.assertEqual(tracker.plank.total, 0)
                self.assertFalse(tracker.plank.active)

    def test_confirmation_uses_full_settle_and_does_not_backfill(self):
        tracker = self.arm_down()
        for i in range(11, 21):
            tracker.update(body(), 'plank', i / 10)
            self.assertEqual(tracker.pushup.state, 'DOWN')
            self.assertEqual(tracker.plank.total, 0)
        tracker.update(body(), 'plank', 2.1)
        self.assertTrue(tracker.plank.active)
        self.assertEqual(tracker.pushup.state, 'READY')
        self.assertEqual(tracker.pushup.count, 0)
        self.assertEqual(tracker.plank.total, 0)
        tracker.update(body(), 'plank', 2.2)
        self.assertAlmostEqual(tracker.plank.total, .1)

    def test_interrupted_candidates_must_settle_again(self):
        for interruption in ('posture', 'geometry', 'label', 'motion', 'side', 'gap'):
            with self.subTest(interruption=interruption):
                tracker = self.arm_down()
                for i in range(11, 19):
                    tracker.update(body(), 'plank', i / 10)
                sample = body()
                label, timestamp = 'plank', 1.9
                if interruption == 'posture':
                    sample['plank']['posture'] = False
                elif interruption == 'geometry':
                    sample['plank'] = None
                elif interruption == 'label':
                    label = 'pushup'
                elif interruption == 'motion':
                    sample['plank']['signature'][0] += .04
                elif interruption == 'side':
                    sample['plank']['side'] = 11
                elif interruption == 'gap':
                    timestamp = 2.1
                tracker.update(sample, label, timestamp)
                tracker.update(body(), 'plank', timestamp + .1)
                tracker.update(body(), 'plank', timestamp + .2)
                self.assertFalse(tracker.plank.active)
                self.assertEqual(tracker.plank.total, 0)
                self.assertEqual(tracker.pushup.state, 'DOWN')


if __name__ == '__main__':
    unittest.main(verbosity=2)
