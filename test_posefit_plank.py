"""Deterministic plank timing, posture and entry-point regression checks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from posefit_runtime import PlankTimer, Tracker, geometry, draw_panel


def test_timer():
    timer = PlankTimer()
    signature = np.zeros(8)
    for i in range(31):
        timer.update(True, i / 10, signature)
    assert timer.active and abs(timer.current - 2.0) < 1e-6
    assert timer.total == timer.best == timer.current
    before = timer.total
    timer.update(False, 3.1)
    timer.update(False, 3.2)
    assert timer.total == before and not timer.active
    timer.update(True, 3.3, signature)
    assert timer.current == before and timer.total == before
    # Reacquisition requires stability again and must not add the missing time.
    for i in range(34, 46):
        timer.update(True, i / 10, signature)
    assert before < timer.total < before + .3
    timer.update(False, 5.2)
    assert timer.current == 0 and timer.total > 0
    saved = timer.total
    for i in range(60, 90):
        timer.update(True, i / 10, signature + (i % 2) * .1)
    assert timer.total == saved, 'Moving poses must not accrue hold time'
    timer = PlankTimer()
    timer.update(True, 0, signature)
    timer.update(True, 10, signature)
    assert timer.total == 0, 'A stalled stream must not add ten seconds'


def plank_landmarks():
    lm = [SimpleNamespace(x=.5, y=.5, visibility=0) for _ in range(33)]
    for i, xy in {11:(.25,.35), 13:(.25,.55), 23:(.5,.37), 25:(.75,.39)}.items():
        lm[i] = SimpleNamespace(x=xy[0], y=xy[1], visibility=.95)
    return lm


def test_tracking():
    lm = plank_landmarks()
    body = geometry(lm, 640, 360)
    assert body['plank']['posture'], 'Plank must work with ankles/wrists out of frame'
    tracker = Tracker()
    for i in range(31):
        assert tracker.update(body, 'plank', i / 10) == 'plank'
    assert abs(tracker.plank.current - 2) < 1e-6
    assert tracker.pushup.count == tracker.squat.count == 0
    previous = tracker.plank.total
    tracker.update(body, 'pushup', 3.1)
    assert tracker.plank.total == previous and not tracker.plank.active
    tracker.update(None, 'Unknown', 4)
    assert tracker.plank.current == 0
    lm[23].y = .15
    assert not geometry(lm, 640, 360)['plank']['posture'], 'Bent hips must pause timer'
    for label in ('standing', 'pushup', 'squat', 'Unknown'):
        t = Tracker()
        for i in range(31):
            t.update(body, label, i / 10)
        assert t.plank.total == 0
    frame = np.full((360, 640, 3), 160, dtype=np.uint8)
    draw_panel(frame, 'plank', .99, tracker, 30)
    assert np.all(frame[140:, :, :] == 160), 'Compact panel must not cover lower frame'
    assert np.all(frame[:, 288:, :] == 160)


def test_entrypoints():
    expected = {'11_realtime_posefit.py':'realtime', '12_temporal_posefit.py':'temporal',
                '13_rep_counter.py':'counter', '13.1_rep_counter_pro.py':'pro',
                '14_final_posefit.py':'final'}
    for filename, stage in expected.items():
        spec = importlib.util.spec_from_file_location('demo', Path(__file__).parent / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = []
        module.run_demo = lambda **kwargs: calls.append(kwargs)
        module.main()
        assert calls == [{'stage':stage}]


if __name__ == '__main__':
    test_timer()
    test_tracking()
    test_entrypoints()
    print('PASS: plank timer, missing frames, motion, posture, exercise separation, compact UI, all five demos')
