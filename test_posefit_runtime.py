import importlib.util
import time
from types import SimpleNamespace
import numpy as np

spec = importlib.util.spec_from_file_location('posefit', '14_final_posefit.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert abs(m.calculate_angle([0, 1], [0, 0], [1, 0]) - 90) < 1e-5
assert m.calculate_angle([0, 0], [0, 0], [1, 0]) is None
c = m.RepCounter(100, 160)
for t, angle in [(0, 80), (.2, 80), (.4, 170), (.6, 170)]:
    c.update(angle, t)
assert c.count == 0, 'Starting at bottom must not count'
for t, angle in [(.8, 80), (1, 80), (1.4, 170), (1.6, 170)]:
    c.update(angle, t)
assert c.count == 1
c.update(80, 2)
c.update(80, 2.2)
c.reset_motion()
c.update(170, 3)
c.update(170, 3.2)
assert c.count == 1, 'Lost tracking must cancel unfinished rep'
tracker = m.Tracker()
def body_data(knee=175, elbow=175, upright=True):
    return dict(squat=dict(angle=knee, side=11, posture=upright),
                pushup=dict(angle=elbow, side=11, posture=not upright))

for i in range(40):
    body = body_data(knee=170, elbow=60 if 10 <= i < 25 else 175)
    tracker.update(body, 'standing', i * .1)
assert tracker.pushup.count == 0, 'Standing arm bends must not count'
tracker = m.Tracker()
for i in range(40):
    down = 10 <= i < 25
    body = body_data(knee=80 if down else 175)
    tracker.update(body, 'squat' if down else 'standing', i * .1)
assert tracker.squat.count == 1, tracker.squat.count
tracker = m.Tracker()
for i in range(40):
    down = 10 <= i < 25
    body = body_data(elbow=65 if down else 175, upright=False)
    tracker.update(body, 'pushup' if down else 'plank', i * .1)
assert tracker.pushup.count == 1, tracker.pushup.count
for dropout_frames, expected in (({18, 19, 20}, 1), (set(range(15, 27)), 0)):
    tracker = m.Tracker()
    for i in range(40):
        down = 10 <= i < 25
        body = None if i in dropout_frames else body_data(knee=95 if down else 160)
        label = 'Unknown' if i in dropout_frames or i in (21, 22) else ('squat' if down else 'standing')
        tracker.update(body, label, i * .1)
    assert tracker.squat.count == expected, (dropout_frames, tracker.squat.count)

# Missing irrelevant joints no longer invalidates the relevant exercise.
lm = [SimpleNamespace(x=.5, y=.5, visibility=0) for _ in range(33)]
for i, point in {11:(.4,.2), 23:(.4,.45), 25:(.4,.65), 27:(.4,.9)}.items():
    lm[i] = SimpleNamespace(x=point[0], y=point[1], visibility=.9)
body = m.geometry(lm, 1280, 720)
assert body['squat'] is not None and body['pushup'] is None
assert body['squat']['posture']
for i in (11, 23, 25, 27):
    lm[i+1] = SimpleNamespace(x=lm[i].x+.1, y=lm[i].y, visibility=.99)
assert m.geometry(lm, 1280, 720, {'squat':11})['squat']['side'] == 11
lm = [SimpleNamespace(x=.5, y=.5, visibility=0) for _ in range(33)]
for i, point in {11:(.3,.3), 13:(.3,.5), 15:(.3,.7), 23:(.65,.35)}.items():
    lm[i] = SimpleNamespace(x=point[0], y=point[1], visibility=.9)
body = m.geometry(lm, 1280, 720)
assert body['pushup'] is not None and body['squat'] is None
assert body['pushup']['posture']
print('PASS: brief occlusion and label jitter, long tracking loss, partial framing, stable side selection')
print('PASS: angles, full cycles, tracking loss, wrong-action rejection, transition labels')
classifier = m.Classifier()
# Synthetic input verifies the shipped bundle without distributing held-out data.
x = np.zeros((1, 99), dtype=np.float32)
raw = classifier.scaler.inverse_transform(x)
np.testing.assert_allclose(classifier.scaler.transform(raw), x, atol=1e-5)
lm = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in raw[0].reshape(33, 3)]
classifier.update(lm, 0)
assert classifier.last_inference == 0
classifier.update(lm, .05)
assert classifier.last_inference == 0
classifier.update(lm, .11)
assert classifier.last_inference == .11
classifier.reset()
assert classifier.probabilities is None
start = time.perf_counter()
for _ in range(30):
    classifier.model.predict(x[:1], verbose=0)
old_ms = (time.perf_counter() - start) * 1000
start = time.perf_counter()
for _ in range(100):
    classifier.infer(x[:1]).numpy()
new_ms = (time.perf_counter() - start) * 10
print(f'Classifier only: old 30 predict calls = {old_ms:.1f} ms; new single call = {new_ms:.2f} ms')
print('PASS: real model, scaler, inference throttle and reset')
