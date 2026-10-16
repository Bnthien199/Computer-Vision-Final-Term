"""PoseFit: scaled MLP inference and posture-gated repetition counting."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import math
import time
from pathlib import Path
import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
WINDOW = 'PoseFit AI Fitness'


class TimestampedPose(mp.solutions.pose.Pose):
    """Keep the existing Pose graph/model, but stamp packets with source time.

    Legacy SolutionBase.process simulates 30 FPS by adding 33333 microseconds.
    The pinned 0.10.21 adapter compensates that increment before every call;
    graph filters then receive the same clock as the classifier and counters.
    This uses a private legacy field: reject unverified versions rather than
    silently reverting to a synthetic clock. Do not change the graph or ROI.
    """
    def __init__(self, **kwargs):
        if mp.__version__ != '0.10.21':
            raise RuntimeError('TimestampedPose requires the verified mediapipe==0.10.21 legacy implementation')
        if kwargs.get('model_complexity', 1) != 1:
            raise ValueError('TimestampedPose uses the installed full Pose model (model_complexity=1)')
        super().__init__(**kwargs)
        self._last_source_timestamp_us = None

    def process(self, image, now):
        if not math.isfinite(now) or not 0 <= now < (2**63 - 2) / 1_000_000:
            raise ValueError('Pose timestamp must be finite, nonnegative and within int64 microseconds')
        timestamp_us = round(now * 1_000_000)
        if self._last_source_timestamp_us is not None and timestamp_us <= self._last_source_timestamp_us:
            raise ValueError('Pose timestamps must strictly increase at microsecond resolution; reset between streams')
        self._simulated_timestamp = timestamp_us - 33333
        result = super().process(image)
        if self._simulated_timestamp != timestamp_us:
            raise RuntimeError('Legacy Pose timestamp implementation changed; packet clock could not be verified')
        self._last_source_timestamp_us = timestamp_us
        return result

    def reset(self):
        super().reset()
        self._last_source_timestamp_us = None


def calculate_angle(a, b, c):
    # Coordinates must be pixels: normalized x and y have different scales.
    u, v = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    length = np.linalg.norm(u) * np.linalg.norm(v)
    if not np.isfinite(length) or length < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / length, -1, 1))))


def geometry(lm, width, height, preferred=None):
    # Each exercise needs its own joints. A hidden wrist must not stop a squat;
    # a hidden ankle must not stop a push-up.
    if len(lm) != 33 or not np.isfinite([width, height]).all() or min(width, height) <= 0:
        return dict.fromkeys(('squat', 'pushup', 'plank'))
    preferred = preferred or {}
    body = {}
    for name, left in (('squat', (11, 23, 25, 27)),
                       ('pushup', (11, 13, 15, 23)),
                       ('plank', (11, 13, 23, 25))):
        sides = (left, tuple(i + 1 for i in left))
        def score(side):
            return min(lm[i].visibility if np.isfinite([lm[i].x, lm[i].y, lm[i].visibility]).all() and 0 <= lm[i].visibility <= 1 and 0 <= lm[i].x <= 1 and 0 <= lm[i].y <= 1
                       else 0 for i in side)
        ids = max(sides, key=score)
        old = next((side for side in sides if side[0] == preferred.get(name)), None)
        # Keep the selected side while it is usable, even if the other gets clearer.
        if old is not None and score(old) >= 0.5:
            ids = old
        if score(ids) < 0.5:
            body[name] = None
            continue
        points = [np.array([lm[i].x * width, lm[i].y * height]) for i in ids]
        if name == 'squat':
            shoulder, hip, knee, ankle = points
            angle = calculate_angle(hip, knee, ankle)
            posture = hip[1] < knee[1] + 0.12 * height
        elif name == 'pushup':
            shoulder, elbow, wrist, hip = points
            angle = calculate_angle(shoulder, elbow, wrist)
            posture = wrist[1] > shoulder[1]
        else:
            shoulder, elbow, hip, knee = points
            angle = calculate_angle(shoulder, hip, knee)
            # Side-view alignment check; not a clinical form-quality assessment.
            posture = angle is not None and angle >= 155 and elbow[1] > shoulder[1]
        torso = shoulder - hip
        tilt = math.degrees(math.atan2(abs(torso[0]), abs(torso[1])))
        posture = posture and (tilt < 65 and shoulder[1] < hip[1] if name == 'squat' else tilt > 50)
        body[name] = (dict(angle=angle, side=ids[0], posture=bool(posture))
                      if angle is not None and np.linalg.norm(torso) >= 10 else None)
        if name == 'plank' and body[name] is not None:
            body[name]['signature'] = np.array(points).flatten() / np.tile([width, height], 4)
    return body


class RepCounter:
    """Require top -> bottom -> top, with stable thresholds and a timeout."""
    def __init__(self, down, up):
        self.down, self.up = down, up
        self.count = 0
        self.reset_motion()

    def reset_motion(self):
        self.state = 'READY'
        self.candidate = None
        self.candidate_since = 0.0
        self.started = None
        self.last_sample = None

    def pause(self):
        # Missing frames must not count as time spent beyond a threshold.
        self.candidate = None

    def update(self, angle, now, can_start=True):
        if not math.isfinite(now):
            self.reset_motion()
            return
        if self.last_sample is not None:
            gap = now - self.last_sample
            if gap < 0 or gap > 0.6:
                self.reset_motion()
            elif gap > 0.25:
                self.pause()
        self.last_sample = now
        if angle is None or not math.isfinite(angle):
            self.pause()
            return
        if self.started is not None and now - self.started > 10:
            self.reset_motion()
        target = 'UP' if angle > self.up else 'DOWN' if angle < self.down else None
        if target != self.candidate:
            self.candidate, self.candidate_since = target, now
        if target is None or now - self.candidate_since < 0.10:
            return
        if target == 'UP':
            if self.state == 'DOWN':
                if now - self.started < 0.35:
                    return
                self.count += 1
            self.state, self.started = 'UP', None
        elif self.state == 'UP' and can_start:
            self.state, self.started = 'DOWN', now


class PlankTimer:
    """Count observed stable hold time; exclude preparation and unseen intervals."""
    def __init__(self, settle_seconds=1.0, grace_seconds=0.6):
        self.settle_seconds, self.grace_seconds = settle_seconds, grace_seconds
        self.total = self.current = self.best = 0.0
        self.reset_motion()

    def reset_motion(self):
        self.current = 0.0
        self.anchor = self.stable_since = self.last_valid = self.last_tick = None
        self.active = False
        self.state = 'READY'

    def update(self, valid, now, signature=None):
        if not math.isfinite(now):
            self.reset_motion()
            return
        if self.last_tick is not None and now < self.last_tick:
            self.reset_motion()
        elapsed = 0.0 if self.last_tick is None else max(0.0, now - self.last_tick)
        was_active = self.active
        if self.last_valid is not None and now - self.last_valid > self.grace_seconds:
            self.reset_motion()
            was_active = False
        elif elapsed > 0.25:
            # Unobserved time cannot establish that the person stayed still.
            self.anchor = self.stable_since = None
            self.active = was_active = False
        self.last_tick = now
        signature = np.asarray(signature, dtype=float)
        valid = valid and signature.shape == (8,) and np.isfinite(signature).all()
        if not valid:
            self.active = False
            self.anchor = self.stable_since = None
            self.state = 'PAUSED' if self.current > 0 else 'READY'
            return
        self.last_valid = now
        moved = self.anchor is not None and np.max(np.abs(signature - self.anchor)) > 0.035
        if self.anchor is None or moved:
            if moved:
                self.current = 0.0
            self.anchor = signature.copy()
            self.stable_since = now
            self.active = False
            was_active = False
        self.active = now - self.stable_since >= self.settle_seconds
        self.state = 'HOLD' if self.active else 'GET STABLE'
        if self.active and was_active and elapsed <= 0.25:
            self.current += elapsed
            self.total += elapsed
            self.best = max(self.best, self.current)


class Classifier:
    def __init__(self, confidence=0.65, smoothing_seconds=0.20, model_path=None, data_dir=None):
        if not 0 < confidence <= 1 or not math.isfinite(smoothing_seconds) or smoothing_seconds < 0:
            raise ValueError('Confidence must be in (0, 1]; smoothing seconds must be finite and nonnegative')
        data_dir = Path(data_dir) if data_dir is not None else ROOT / 'training_data'
        model_path = Path(model_path) if model_path is not None else ROOT / 'models/mlp_posefit_final.keras'
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.scaler = joblib.load(data_dir / 'scaler.pkl')
        self.classes = joblib.load(data_dir / 'label_encoder.pkl').classes_
        if self.model.input_shape[-1] != 99 or self.model.output_shape[-1] != len(self.classes) or self.scaler.n_features_in_ != 99:
            raise ValueError('Model, scaler and encoder dimensions do not match the 99 xyz features')
        self.threshold = confidence
        self.smoothing_seconds = smoothing_seconds
        if set(self.classes) != {'plank', 'pushup', 'squat', 'standing'}:
            raise ValueError(f'Unexpected model labels: {self.classes}')
        self.infer = tf.function(lambda x: self.model(x, training=False), autograph=False,
                                 input_signature=[tf.TensorSpec([None, 99], tf.float32)])
        self.infer(tf.zeros((1, 99)))
        self.reset()

    def reset(self):
        self.probabilities = None
        self.last_inference = None
        self.last_sample = None

    def update(self, landmarks, now):
        if not math.isfinite(now):
            self.reset()
            return 'Unknown', 0.0
        last_sample = getattr(self, 'last_sample', None)
        if last_sample is not None and (now < last_sample or now - last_sample > 0.6):
            self.reset()
        self.last_sample = now
        raw = np.array([[v for lm in landmarks for v in (lm.x, lm.y, lm.z)]], dtype=np.float32)
        if raw.shape != (1, 99) or not np.isfinite(raw).all():
            self.reset()
            return 'Unknown', 0.0
        # Classify only a new sample, at most 10 times per second.
        if self.last_inference is None or now - self.last_inference >= 0.10:
            features = self.scaler.transform(raw).astype(np.float32)
            if not np.isfinite(features).all():
                self.reset()
                return 'Unknown', 0.0
            prediction = self.infer(features).numpy()[0]
            if (prediction.shape != (len(self.classes),) or not np.isfinite(prediction).all()
                    or np.any(prediction < 0) or np.any(prediction > 1)
                    or not np.isclose(prediction.sum(), 1, atol=1e-4)):
                raise ValueError('Model returned invalid class probabilities')
            alpha = (1.0 if self.last_inference is None or self.smoothing_seconds <= 0
                     else 1 - math.exp(-(now - self.last_inference) / self.smoothing_seconds))
            self.probabilities = prediction if self.probabilities is None else (
                alpha * prediction + (1 - alpha) * self.probabilities)
            self.last_inference = now
        index = int(np.argmax(self.probabilities))
        confidence = float(self.probabilities[index])
        label = str(self.classes[index]) if confidence >= self.threshold else 'Unknown'
        return label, confidence


class Tracker:
    def __init__(self, squat_down=110, squat_up=150, pushup_down=100, pushup_up=150):
        self.squat = RepCounter(squat_down, squat_up)
        self.pushup = RepCounter(pushup_down, pushup_up)
        self.plank = PlankTimer()
        self.reset_motion()

    def reset_motion(self):
        self.squat.reset_motion()
        self.pushup.reset_motion()
        self.plank.reset_motion()
        self.sides = {}
        self.angles = {}
        self.last_seen = {}
        self.last_label = {}
        self.last_tick = None
        self.hint = 'Start with legs / arms extended'

    def update(self, body, label, now):
        if not math.isfinite(now):
            self.reset_motion()
            return 'Unknown'
        if self.last_tick is not None and (now < self.last_tick or now - self.last_tick > 0.6):
            self.reset_motion()
        self.last_tick = now
        body = body or {}
        displayed = label
        self.hint = 'Show hips, knees, ankles OR shoulders, elbows, wrists, hips'
        for name, family in (('squat', ('squat', 'standing')),
                             ('pushup', ('pushup', 'plank'))):
            counter = getattr(self, name)
            joint = body.get(name)
            seen = self.last_seen.get(name)
            if seen is not None and now - seen > 0.6:
                counter.reset_motion()
                self.angles.pop(name, None)
            if joint is None:
                counter.pause()
                continue
            if not joint['posture']:
                counter.reset_motion()
                self.angles.pop(name, None)
                continue
            if label in family:
                self.last_label[name] = now
            allowed = label in family or (label == 'Unknown' and
                        now - self.last_label.get(name, -math.inf) <= 0.7)
            if not allowed:
                counter.reset_motion()
                # Cancelled motion must not contribute old angles to a new one.
                self.angles.pop(name, None)
                continue
            if name in self.sides and self.sides[name] != joint['side']:
                counter.reset_motion()
                self.angles.pop(name, None)
            dt = 0.1 if seen is None else now - seen
            alpha = 1 - math.exp(-dt / 0.06)
            old = self.angles.get(name, joint['angle'])
            angle = old + alpha * (joint['angle'] - old)
            self.angles[name] = angle
            self.sides[name], self.last_seen[name] = joint['side'], now
            # A family label may arm the top; require the actual exercise to
            # start the bottom. Once started, brief label uncertainty is tolerated.
            counter.update(angle, now, can_start=label == name)
            if counter.state == 'DOWN':
                displayed = name
            self.hint = f'{name}: {angle:.0f} deg | {counter.state} | down <{counter.down}, up >{counter.up}'
        plank = body.get('plank')
        # Probe stability independently of an unfinished push-up. Otherwise DOWN
        # blocks HOLD, while HOLD is the only place that releases the old DOWN.
        # Keep counting the push-up normally until the existing timer confirms
        # a stable plank; a few misclassified frames must not cancel a real rep.
        valid_plank = label == 'plank' and plank is not None and plank['posture']
        if valid_plank:
            if self.sides.get('plank', plank['side']) != plank['side']:
                self.plank.reset_motion()
            self.sides['plank'] = plank['side']
        self.plank.update(valid_plank, now, plank['signature'] if valid_plank else None)
        if label == 'plank':
            if valid_plank:
                if self.pushup.state != 'DOWN' or self.plank.active:
                    displayed = 'plank'
                self.hint = f'Plank: {self.plank.state} | keep body straight and still'
                # Release only an unfinished bottom. Preserve an armed UP so a
                # real push-up can start after a confirmed straight-arm hold.
                if self.plank.active and self.pushup.state == 'DOWN':
                    self.pushup.reset_motion()
            elif self.pushup.state != 'DOWN':
                self.hint = 'Plank timer: show shoulder, elbow, hip, knee in side view'
        return displayed


def draw_panel(frame, exercise, confidence, tracker, fps, show_reps=True, *, classifier_label=None):
    h, w = frame.shape[:2]
    panel = frame[8:min(h, 140 if show_reps else 120), 8:min(w, 288)]
    panel[:] = (panel * 0.50).astype(np.uint8)
    hint = tracker.hint
    if hint.startswith('Show hips'):
        hint = 'Keep the joints of your exercise in view'
    # Confidence belongs to the classifier's accepted EMA prediction, not a
    # latched counter activity returned by Tracker (e.g. pushup still DOWN).
    prediction = exercise if classifier_label is None else classifier_label
    lines = ['POSEFIT', f'{prediction}  |  Confidence: {confidence:.0%}']
    if show_reps:
        lines.append(f'Squat: {tracker.squat.count}   Push-up: {tracker.pushup.count}')
    lines.extend([f'Plank: {tracker.plank.current:.1f}s  Total: {tracker.plank.total:.1f}s',
                  f'FPS: {fps:.1f}   Q: quit   R: reset', hint])
    for i, line in enumerate(lines):
        scale = 0.44 if i == 0 else 0.32 if i == len(lines) - 1 else 0.38
        text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        scale *= min(1.0, max(1, panel.shape[1] - 16) / max(1, text_width))
        cv2.putText(frame, line, (16, 26 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 255, 255) if i == 0 else (255, 255, 255), 1, cv2.LINE_AA)


def main(stage='final'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/mlp_posefit_final.keras')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'training_data', help='Scaler and encoder belonging to the selected model')
    parser.add_argument('--width', type=int, default=640, help='Processing width, minimum 480')
    parser.add_argument('--confidence', type=float, default=0.65)
    parser.add_argument('--capture-width', type=int, default=1280)
    parser.add_argument('--capture-height', type=int, default=720)
    parser.add_argument('--zoom', type=float, default=None, help='Optional hardware zoom; supported range depends on camera')
    parser.add_argument('--squat-down', type=float, default=110)
    parser.add_argument('--squat-up', type=float, default=150)
    parser.add_argument('--pushup-down', type=float, default=100)
    parser.add_argument('--pushup-up', type=float, default=150)
    parser.add_argument('--plank-trace', type=Path, default=None,
                        help='Optional diagnostic run under runs/: xyz/probabilities/timer gates; no images saved')
    args = parser.parse_args()
    if args.width < 480 or not 0 < args.confidence <= 1:
        parser.error('width must be >= 480; confidence must be in (0, 1]')
    if min(args.capture_width, args.capture_height) <= 0:
        parser.error('capture dimensions must be positive')
    if not (0 < args.squat_down < args.squat_up < 180 and 0 < args.pushup_down < args.pushup_up < 180):
        parser.error('angle thresholds must satisfy 0 < down < up < 180')
    classifier = Classifier(args.confidence, smoothing_seconds=0 if stage == 'realtime' else 0.20,
                            model_path=args.model, data_dir=args.data_dir)
    show_reps = stage not in ('realtime', 'temporal')
    window = f'{WINDOW} - {stage}'
    tracker = Tracker(args.squat_down, args.squat_up, args.pushup_down, args.pushup_up)
    cap = cv2.VideoCapture(args.camera)
    pose = None
    trace = None
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera. Close other camera apps or try --camera 1.')
        # Capture and processing resolutions are independent. Preserve the entire
        # camera frame, then shrink for inference without cropping or stretching.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
        if args.zoom is not None and not cap.set(cv2.CAP_PROP_ZOOM, args.zoom):
            print('Camera did not accept the requested hardware zoom.')
        print(f'Camera capture: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}')
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if args.plank_trace is not None:
            from posefit_plank_trace import PlankTrace
            trace = PlankTrace(args.plank_trace, classifier, args.model, args.data_dir,
                               input_config=dict(stage=stage, camera=args.camera, processing_width_cap=args.width,
                                   capture_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), capture_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                   mediapipe_version=mp.__version__, pose_model_complexity=1,
                                   pose_smooth_landmarks=True, detection_confidence=.6, tracking_confidence=.6))
        # Use the installed full model; avoid downloading another pose model.
        pose = TimestampedPose(model_complexity=1, smooth_landmarks=True,
                                      enable_segmentation=False,
                                      min_detection_confidence=0.6, min_tracking_confidence=0.6)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window, 960, 540)
        print('PoseFit started. Focus camera window: Q = quit, R = reset counts.')
        previous, fps = time.perf_counter(), 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Camera stopped delivering frames.')
                break
            now = time.perf_counter()
            if frame.shape[1] > args.width:
                height = round(frame.shape[0] * args.width / frame.shape[1])
                frame = cv2.resize(frame, (args.width, height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb, now)
            exercise, confidence = 'No Pose', 0.0
            prediction_label = 'No Pose'
            lm, body = None, None
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                body = geometry(lm, frame.shape[1], frame.shape[0], tracker.sides)
                label, confidence = classifier.update(lm, now)
                prediction_label = label
                exercise = tracker.update(body, label, now)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            else:
                classifier.reset()
                tracker.update(None, 'Unknown', now)
            instant_fps = 1 / max(now - previous, 1e-6)
            fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
            previous = now
            draw_panel(frame, exercise, confidence, tracker, fps, show_reps=show_reps,
                       classifier_label=prediction_label)
            if trace is not None:
                trace.write(now, lm, body, prediction_label, confidence, exercise, tracker,
                            image_size=(frame.shape[1], frame.shape[0]))
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('r'), ord('R')):
                tracker = Tracker(args.squat_down, args.squat_up, args.pushup_down, args.pushup_up)
                classifier.reset()
    finally:
        cap.release()
        try:
            if pose is not None:
                pose.close()
        finally:
            try:
                if trace is not None:
                    trace.close()
            finally:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
