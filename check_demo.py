"""Check the shipped inference bundle without training or needing a dataset."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--webcam', action='store_true', help='Also read one camera frame; nothing saved')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    # Check bytes before loading trusted project pickle/Keras files.
    hashes = json.loads((ROOT / 'artifacts.sha256.json').read_text(encoding='utf-8'))
    for name, expected in hashes.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Missing or changed artifact: {name}. Restore from the trusted repository.')
        print(f'OK SHA256 {name}')

    import cv2
    import numpy as np
    import mediapipe as mp
    from posefit_runtime import Classifier, TimestampedPose

    classifier = Classifier()
    landmarks = [SimpleNamespace(x=.5, y=.5, z=0., visibility=1.) for _ in range(33)]
    label, confidence = classifier.update(landmarks, 1.)
    assert label in {*classifier.classes, 'Unknown'} and np.isfinite(confidence)
    print(f'OK scaler -> MLP -> encoder: {classifier.classes.tolist()}')
    pose_model = Path(mp.__file__).parent / 'modules/pose_landmark/pose_landmark_full.tflite'
    if not pose_model.is_file():
        raise RuntimeError('MediaPipe wheel is missing its full pose model; reinstall the pinned package.')
    with TimestampedPose(model_complexity=1) as pose:
        pose.process(np.zeros((480, 640, 3), dtype=np.uint8), 1.)
        pose.process(np.zeros((480, 640, 3), dtype=np.uint8), 1.1)
    print('OK installed MediaPipe graph/model and source timestamps (blank frames)')
    if args.webcam:
        cap = cv2.VideoCapture(args.camera)
        try:
            if not cap.isOpened():
                raise RuntimeError('Cannot open camera. Check Windows privacy settings or try --camera 1.')
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError('Camera opened but did not deliver a frame.')
            print(f'OK camera {args.camera}: {frame.shape[1]}x{frame.shape[0]}; frame not saved')
        finally:
            cap.release()
    print('PASS. Synthetic inference is a compatibility check, not an accuracy evaluation.')


if __name__ == '__main__':
    main()
