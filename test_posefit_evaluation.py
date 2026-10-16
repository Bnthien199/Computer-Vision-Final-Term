"""Technical evaluator regressions; synthetic inputs are not accuracy measurements."""
import csv
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from evaluate_posefit import process_video, video_timestamp


class EvaluationTests(unittest.TestCase):
    def test_clock(self):
        self.assertEqual(video_timestamp(0, 0, 30, None), (0., 'decoder_pts'))
        self.assertEqual(video_timestamp(1, 0, 30, 0), (1 / 30, 'frame_over_fps'))
        self.assertEqual(video_timestamp(2, 80, 30, .04), (.08, 'decoder_pts'))
        with self.assertRaises(ValueError):
            video_timestamp(1, np.nan, np.nan, 0)
        with self.assertRaises(ValueError):
            video_timestamp(1, 0, 30, 10)

    def test_truncated_video_and_reset(self):
        class Capture:
            def __init__(self): self.n, self.released = 0, False
            def isOpened(self): return True
            def get(self, key):
                return {cv2.CAP_PROP_FRAME_COUNT: 3, cv2.CAP_PROP_FPS: 30,
                        cv2.CAP_PROP_POS_MSEC: max(0, self.n - 1) * 1000 / 30}[key]
            def read(self):
                self.n += 1
                return (True, np.zeros((10, 10, 3), np.uint8)) if self.n <= 2 else (False, None)
            def release(self): self.released = True
        class Classifier:
            def __init__(self): self.resets = 0
            def reset(self): self.resets += 1
        class Pose:
            def __init__(self): self.closed = False
            def process(self, rgb, now):
                self.last_timestamp = now
                return SimpleNamespace(pose_landmarks=None)
            def close(self): self.closed = True
        classifier = Classifier()
        poses = []
        def factory():
            poses.append(Pose())
            return poses[-1]
        for _ in range(2):
            cap = Capture()
            with patch('evaluate_posefit.cv2.VideoCapture', return_value=cap):
                result = process_video('unused', classifier, 640, factory, csv.writer(io.StringIO()))
            self.assertEqual(result['status'], 'error')
            self.assertIn('Incomplete decode', result['error'])
            self.assertEqual(result['frames_read'], 2)
            self.assertEqual(result['squat_reps'], 0)
            self.assertIsNone(result['rep_ground_truth'])
            self.assertTrue(cap.released and poses[-1].closed)
            self.assertAlmostEqual(poses[-1].last_timestamp, 1 / 30)
        self.assertEqual(len(poses), 2)
        self.assertEqual(classifier.resets, 6)


if __name__ == '__main__':
    unittest.main()
