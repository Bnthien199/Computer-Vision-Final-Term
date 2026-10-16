"""Read-only evaluation of saved features, or sequential local-video diagnostics.

No fitting, downloads, relabeling or rep/timing scores without ground truth.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from posefit_console import configure_console
from posefit_pipeline import ROOT, FEATURE_COLUMNS, LABELS, new_output, validate_features, validate_metadata
from posefit_runtime import Classifier, Tracker, TimestampedPose, geometry


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def feature_evaluation(args, classifier, output):
    df = pd.read_csv(args.features)
    validate_features(df, require_all_splits=False)
    subset = df.loc[df.split.eq(args.split)]
    if subset.empty:
        raise ValueError(f'No feature rows for split {args.split}')
    x = classifier.scaler.transform(subset[FEATURE_COLUMNS].to_numpy()).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features after applying the saved scaler')
    encoder = joblib.load(args.data_dir / 'label_encoder.pkl')
    y = encoder.transform(subset.label)
    probabilities = np.concatenate([classifier.infer(x[i:i + 256]).numpy() for i in range(0, len(x), 256)])
    if (probabilities.shape != (len(x), 4) or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)):
        raise ValueError('Invalid model probabilities')
    pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    accepted = confidence >= args.confidence
    report = classification_report(y, pred, labels=np.arange(4), target_names=list(classifier.classes),
                                   zero_division=0, output_dict=True)
    bins = []
    ece = 0.0
    for i in range(10):
        mask = (confidence >= i / 10) & (confidence < (i + 1) / 10 if i < 9 else confidence <= 1)
        if mask.any():
            acc, conf = float(np.mean(pred[mask] == y[mask])), float(confidence[mask].mean())
            ece += mask.mean() * abs(acc - conf)
            bins.append(dict(lower=i / 10, upper=(i + 1) / 10, n=int(mask.sum()), accuracy=acc, confidence=conf))
    report.update(split=args.split, n=len(x), accuracy=float(accuracy_score(y, pred)),
                  balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                  confusion_matrix=confusion_matrix(y, pred, labels=np.arange(4)).tolist(),
                  classes=classifier.classes.tolist(), confidence_threshold=args.confidence,
                  coverage=float(accepted.mean()), accepted_n=int(accepted.sum()),
                  accepted_accuracy=float(np.mean(pred[accepted] == y[accepted])) if accepted.any() else None,
                  correct_and_accepted_fraction=float(np.mean(accepted & (pred == y))),
                  negative_log_likelihood=float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1)).mean()),
                  multiclass_brier=float(np.mean(np.sum((probabilities - np.eye(4)[y]) ** 2, axis=1))),
                  ece_10_bins=float(ece), reliability_bins=bins,
                  evaluation_unit='stored feature row; no temporal smoothing or geometry gating',
                  limitations=['No rep/hold/form ground truth', 'Frames from the same clip are correlated',
                               'No legacy feature filepath: clip-level grouping is unavailable'] if 'filepath' not in df else
                               ['No rep/hold/form ground truth', 'Frames from the same clip are correlated'])
    write_json(output / 'metrics.json', report)
    records = subset[[c for c in ('filepath', 'frame', 'subject', 'label', 'split') if c in subset]].copy()
    records.insert(0, 'feature_row', subset.index)
    records['top1_prediction'] = classifier.classes[pred]
    records['confidence'] = confidence
    records['accepted_prediction'] = np.where(accepted, classifier.classes[pred], 'Unknown')
    for i, label in enumerate(classifier.classes):
        records[f'p_{label}'] = probabilities[:, i]
    with (output / 'predictions.csv').open('x', encoding='utf-8', newline='') as stream:
        records.to_csv(stream, index=False)
    print(f'{args.split}: {len(x)} rows; accuracy={report["accuracy"]:.4%}; '
          f'macro-F1={report["macro avg"]["f1-score"]:.4f}; coverage={report["coverage"]:.4%}')
    return 0


def video_timestamp(index, milliseconds, fps, previous):
    """Prefer decoder PTS. Declare every use of the constant-FPS fallback."""
    pts = milliseconds / 1000
    if math.isfinite(pts) and pts >= 0 and (previous is None or pts > previous):
        return float(pts), 'decoder_pts'
    if math.isfinite(fps) and fps > 0:
        fallback = index / fps
        if previous is None or fallback > previous:
            return float(fallback), 'frame_over_fps'
    raise ValueError(f'No strictly increasing video timestamp at frame {index}: PTS={pts}, FPS={fps}')


def process_video(path, classifier, width, pose_factory, timeline, *, tracker_factory=None):
    # A fresh tracker and MediaPipe graph for every video; never feed the true
    # annotation label to the classifier/counter.
    started_at = time.perf_counter()
    classifier.reset()
    # Evaluation-only injection. With no factory, keep the existing runtime
    # defaults and instrumentation wrappers used by validation capture tools.
    tracker = Tracker() if tracker_factory is None else tracker_factory()
    cap = None
    pose = None
    counts = Counter()
    result = dict(status='error', error=None, frames_read=0, expected_frames=None,
                  pose_frames=0, accepted_frames=0, timestamp_fallback_frames=0,
                  dominant_prediction=None, squat_reps=0, pushup_reps=0, plank_seconds=0.0,
                  pose_process_calls=0, classifier_update_calls=0,
                  classifier_inference_updates=0, geometry_calls=0,
                  counter_thresholds={f'{name}_{phase}': getattr(getattr(tracker, name), phase)
                                      for name in ('squat', 'pushup') for phase in ('down', 'up')},
                  rep_ground_truth=None, plank_ground_truth_seconds=None,
                  geometry_frames={name: 0 for name in ('squat', 'pushup', 'plank')},
                  posture_frames={name: 0 for name in ('squat', 'pushup', 'plank')},
                  angle_range={name: [None, None] for name in ('squat', 'pushup', 'plank')})
    previous = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError('Cannot open video')
        count, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        expected = int(round(count)) if math.isfinite(count) and count >= 1 else None
        result['expected_frames'] = expected
        pose = pose_factory()
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            index = result['frames_read']
            result['frames_read'] += 1
            now, clock = video_timestamp(index, cap.get(cv2.CAP_PROP_POS_MSEC), fps, previous)
            previous = now
            result['timestamp_fallback_frames'] += int(clock == 'frame_over_fps')
            if frame.shape[1] > width:
                h = max(1, round(frame.shape[0] * width / frame.shape[1]))
                frame = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result['pose_process_calls'] += 1
            detection = pose.process(rgb, now)
            label, confidence = 'Unknown', 0.0
            body = {}
            if detection.pose_landmarks:
                landmarks = detection.pose_landmarks.landmark
                result['pose_frames'] += 1
                result['classifier_update_calls'] += 1
                previous_inference = getattr(classifier, 'last_inference', None)
                label, confidence = classifier.update(landmarks, now)
                current_inference = getattr(classifier, 'last_inference', None)
                result['classifier_inference_updates'] += int(current_inference is not None and
                                                               current_inference != previous_inference)
                result['geometry_calls'] += 1
                body = geometry(landmarks, frame.shape[1], frame.shape[0], tracker.sides)
                for name, joint in body.items():
                    if joint is not None:
                        result['geometry_frames'][name] += 1
                        result['posture_frames'][name] += int(joint['posture'])
                        low, high = result['angle_range'][name]
                        angle = float(joint['angle'])
                        result['angle_range'][name] = [angle if low is None else min(low, angle),
                                                      angle if high is None else max(high, angle)]
                tracker.update(body, label, now)
            else:
                classifier.reset()
                tracker.update(None, 'Unknown', now)
            if label in LABELS:
                counts[label] += 1
                result['accepted_frames'] += 1
            timeline.writerow([index, now, clock, label, confidence, tracker.squat.count,
                               tracker.pushup.count, tracker.plank.total,
                               (body.get('squat') or {}).get('angle'), (body.get('pushup') or {}).get('angle'),
                               (body.get('squat') or {}).get('posture'), (body.get('pushup') or {}).get('posture'),
                               tracker.squat.state, tracker.pushup.state, tracker.plank.state])
        if not result['frames_read']:
            raise RuntimeError('No decoded frames')
        if expected is None:
            raise RuntimeError('Frame-count metadata unavailable: complete decoding cannot be verified')
        if result['frames_read'] != expected:
            raise RuntimeError(f'Incomplete decode: expected {expected}, read {result["frames_read"]}')
        result['status'] = 'ok'
    except Exception as exc:
        # Errors remain visible in the report; partial results are diagnostics.
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if cap is not None:
            cap.release()
        if pose is not None:
            try:
                pose.close()
            except Exception as exc:
                result['status'] = 'error'
                result['error'] = f'Pose cleanup failed: {exc}; earlier error: {result["error"]}'
        result.update(squat_reps=tracker.squat.count, pushup_reps=tracker.pushup.count,
                      plank_seconds=float(tracker.plank.total),
                      elapsed_seconds=time.perf_counter() - started_at,
                      dominant_prediction=counts.most_common(1)[0][0] if counts else None)
    return result


def video_evaluation(args, classifier, output):
    df = pd.read_csv(args.metadata)
    validate_metadata(df)
    selected = df.loc[df.split.eq(args.split) & df.source.eq('gym') & df.label.isin(args.labels)]
    if args.limit is not None:
        selected = selected.head(args.limit)
    if selected.empty:
        raise ValueError('No videos match the requested split and labels')
    # Full model is bundled in the current installation. Fail before constructing
    # MediaPipe if it is absent; Pose would otherwise try to download a model.
    bundled = Path(mp.__file__).parent / 'modules/pose_landmark/pose_landmark_full.tflite'
    if not bundled.is_file():
        raise RuntimeError(f'Installed MediaPipe full model is missing: {bundled}; no download attempted')
    def pose_factory():
        return TimestampedPose(model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                                      min_detection_confidence=0.6, min_tracking_confidence=0.6)
    results = []
    for i, (_, row) in enumerate(selected.iterrows()):
        with (output / f'video_{i:04d}_timeline.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['frame', 'timestamp_seconds', 'timestamp_source', 'classifier_label', 'confidence',
                             'squat_reps', 'pushup_reps', 'plank_total_seconds', 'squat_angle', 'pushup_angle',
                             'squat_posture', 'pushup_posture', 'squat_state', 'pushup_state', 'plank_state'])
            record = process_video(ROOT / row.filepath, classifier, args.width, pose_factory, writer)
        record.update(filepath=row.filepath, label=row.label, split=row.split, timeline=f'video_{i:04d}_timeline.csv')
        results.append(record)
        write_json(output / f'video_{i:04d}.json', record)
        print(f'[{i+1}/{len(selected)}] {record["status"]}: {row.filepath}; '
              f'frames={record["frames_read"]}; fallback={record["timestamp_fallback_frames"]}; {record["error"] or ""}')
    errors = sum(r['status'] != 'ok' for r in results)
    write_json(output / 'video_summary.json', dict(split=args.split, labels=args.labels, explicit_limit=args.limit,
               selected_videos=len(results), successful_videos=len(results)-errors, error_videos=errors, videos=results,
               rep_mae=None, exact_rep_accuracy=None, within_one_rep_accuracy=None, plank_timing_error=None,
               limitation='Local metadata has action labels only. Counts/times are diagnostics without ground truth. '
                          'FPS fallback assumes constant frame rate; every affected frame is marked.'))
    return int(errors > 0)


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['features', 'videos'])
    parser.add_argument('--features', type=Path, default=ROOT / 'pose_features.csv')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'metadata.csv')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/mlp_posefit_final.keras')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'training_data')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--labels', nargs='+', choices=sorted(LABELS - {'standing'}), default=['squat', 'pushup', 'plank'])
    parser.add_argument('--limit', type=int, help='Explicit video subset for technical smoke checks only')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--confidence', type=float, default=.65)
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory under runs/')
    args = parser.parse_args()
    if args.mode == 'features' and args.limit is not None:
        parser.error('--limit applies only to videos')
    if args.width < 480 or not 0 < args.confidence <= 1 or (args.limit is not None and args.limit <= 0):
        parser.error('width >= 480, confidence in (0,1], limit > 0 required')
    output = new_output(args.output, directory=True)
    config = vars(args).copy()
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in config.items()}
    config['artifact_sha256'] = {str(p.resolve()): sha256(p) for p in
                                 [args.model, args.data_dir / 'scaler.pkl', args.data_dir / 'label_encoder.pkl',
                                  args.features if args.mode == 'features' else args.metadata]}
    config['preprocessing'] = '33 MediaPipe landmarks, ordered x/y/z -> f_0..f_98 -> saved StandardScaler'
    config['pose_packet_clock'] = 'source timestamp rounded to microseconds; verified mediapipe 0.10.21 adapter' if args.mode == 'videos' else None
    write_json(output / 'config.json', config)
    try:
        classifier = Classifier(args.confidence, model_path=args.model, data_dir=args.data_dir)
        code = feature_evaluation(args, classifier, output) if args.mode == 'features' else video_evaluation(args, classifier, output)
    except Exception as exc:
        write_json(output / 'failure.json', dict(error=f'{type(exc).__name__}: {exc}'))
        raise
    print(f'Report: {output}')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
