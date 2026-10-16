"""Regression: unseen intervals cannot satisfy rep stability; reject bad landmarks."""
import math
from types import SimpleNamespace
import numpy as np
from posefit_runtime import Classifier, RepCounter, PlankTimer, Tracker, calculate_angle, geometry


def main():
    counter = RepCounter(100, 160)
    counter.update(170, 0)
    counter.update(170, .4)
    assert counter.state == 'READY', 'A stream gap is not observed stability'
    counter.update(170, .51)
    assert counter.state == 'UP'
    counter.update(80, .6)
    counter.update(80, .71)
    assert counter.state == 'DOWN'
    counter.update(170, 2)
    counter.update(170, 2.11)
    assert counter.count == 0, 'A long stream stall cancels the unfinished rep'
    assert calculate_angle([np.nan, 0], [0, 0], [1, 1]) is None
    lm = [SimpleNamespace(x=.5, y=.5, z=0, visibility=math.nan) for _ in range(33)]
    assert all(v is None for v in geometry(lm, 640, 360).values())
    # Validate input before calling the real model/scaler; a bad frame clears cached confidence.
    classifier = Classifier.__new__(Classifier)
    classifier.probabilities = np.array([1., 0., 0., 0.])
    classifier.last_inference = 0
    lm[0].x = math.nan
    assert classifier.update(lm, .05) == ('Unknown', 0.)
    assert classifier.probabilities is None
    timer = PlankTimer()
    for t in np.arange(0, 2, .1):
        timer.update(True, float(t), None)
    assert timer.total == 0 and not timer.active, 'Invalid signatures cannot establish a hold'
    signature = np.zeros(8)
    for t in np.arange(0, 2, .1):
        timer.update(True, float(t), signature)
    assert timer.active and timer.total > 0
    total = timer.total
    timer.update(True, 2.3, signature)
    assert not timer.active and timer.total == total, 'A stream stall requires settling again'
    timer.update(True, 0, signature)
    assert timer.current == 0 and timer.total == total, 'Clock regression cancels only current hold'
    timer.update(True, math.nan, signature)
    assert timer.last_tick is None
    counter.update(80, math.nan)
    assert counter.state == 'READY'
    assert all(v is None for v in geometry(lm[:10], 640, 360).values())
    tracker = Tracker()
    tracker.update(None, 'Unknown', 10)
    tracker.squat.state = 'DOWN'
    tracker.update(None, 'Unknown', 0)
    assert tracker.squat.state == 'READY'
    # Regressed video time must trigger fresh inference, not stale probabilities.
    classifier = Classifier.__new__(Classifier)
    classifier.reset()
    classifier.classes = np.array(['plank', 'pushup', 'squat', 'standing'])
    classifier.threshold, classifier.smoothing_seconds = .65, .2
    classifier.scaler = SimpleNamespace(transform=lambda x: x)
    calls = []
    def infer(x):
        calls.append(x)
        return SimpleNamespace(numpy=lambda: np.array([[0., 0., 1., 0.]]))
    classifier.infer = infer
    good = [SimpleNamespace(x=.5, y=.5, z=0, visibility=1) for _ in range(33)]
    classifier.update(good, 10)
    classifier.update(good, 0)
    assert len(calls) == 2
    assert classifier.update(good, math.inf) == ('Unknown', 0.)
    print('PASS: stream stalls, rep cancellation, nonfinite geometry and classifier reset')


if __name__ == '__main__':
    main()
