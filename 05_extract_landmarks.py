import argparse
from posefit_pipeline import new_output, validate_metadata, validate_features, FEATURE_COLUMNS
from pathlib import Path
from posefit_console import configure_console
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

METADATA_FILE = (
    ROOT / "metadata.csv"
)

OUTPUT_FILE = (
    ROOT / "runs/pose_features.csv"
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

# Số frame lấy từ mỗi video
FRAMES_PER_VIDEO = 5

# ============================================================
# IMAGE READER
# ============================================================

def read_image_unicode(path):

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, cv2.error):
        return None
    if not data.size:
        return None

    try:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error:
        return None

    return image

# ============================================================
# EXTRACT LANDMARK
# ============================================================

def extract_pose(image, pose):

    if image is None:
        return None

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    result = pose.process(
        rgb
    )

    if not result.pose_landmarks:
        return None

    landmarks = (
        result
        .pose_landmarks
        .landmark
    )

    features = []

    for lm in landmarks:

        features.extend(
            [
                lm.x,
                lm.y,
                lm.z
            ]
        )

    if len(features) != 99 or not np.isfinite(features).all():
        return None
    visibility = np.array([lm.visibility for lm in landmarks])
    if not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)):
        return None
    return features, visibility.tolist()

# ============================================================
# VIDEO FRAME SAMPLING
# ============================================================

def extract_video_frames(video_path):

    cap = cv2.VideoCapture(str(video_path))
    try:
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not np.isfinite(count) or count < 1:
            return []
        total_frames = int(count)
        indices = np.unique(np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        previous_timestamp = None
        for idx in indices:
            if cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)) is False:
                raise RuntimeError(f'Decoder could not seek to frame {idx}')
            success, frame = cap.read()
            if not success or frame is None:
                frames.append((int(idx), None, None))
                continue
            position = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not np.isfinite(position) or abs(position - (idx + 1)) > 0.5:
                raise RuntimeError(f'Decoder position {position} does not match requested frame {idx}')
            milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = float(milliseconds / 1000) if np.isfinite(milliseconds) and milliseconds >= 0 else None
            if timestamp is None or (idx > 0 and timestamp == 0) or (previous_timestamp is not None and timestamp <= previous_timestamp):
                timestamp = float(idx / fps) if np.isfinite(fps) and fps > 0 else None
            previous_timestamp = timestamp
            frames.append((int(idx), timestamp, frame))
        return frames
    finally:
        cap.release()

# ============================================================
# INITIALIZE MEDIAPIPE
# ============================================================

def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    print("="*70)
    print("POSEFIT - EXTRACT LANDMARKS")
    print("="*70)

    print("\nLoading metadata...")

    df = pd.read_csv(
        args.metadata
    )

    print(
        "Samples in metadata:",
        len(df)
    )

    # ============================================================
    # PROCESS DATA
    # ============================================================

    results = []

    success = 0
    failed = 0

    validate_metadata(df)
    output = new_output(args.output)
    log_path = new_output(output.with_suffix(".extraction.csv"))
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    )

    attempts = []

    def checked_pose(image):
        try:
            return extract_pose(image, pose), None
        except (ValueError, RuntimeError, cv2.error) as exc:
            return None, str(exc)


    try:
        for _, row in tqdm(
            df.iterrows(),
            total=len(df)
        ):

            file_path = (
                ROOT
                /
                row["filepath"]
            )

            label = row["label"]

            source = row["source"]

            # --------------------------------------------------------
            # VIDEO
            # --------------------------------------------------------

            if source == "gym":

                try:
                    frames = extract_video_frames(file_path)
                except (OSError, cv2.error, RuntimeError) as exc:
                    attempts.append(dict(filepath=row['filepath'], label=label, split=row['split'], frame=None,
                                         status='video_error', error=str(exc)))
                    failed += 1
                    continue

                if not frames:
                    failed += 1
                    attempts.append(dict(filepath=row["filepath"], label=label, split=row["split"], frame=None, status="unreadable_video"))
                for frame_id, timestamp, frame in frames:

                    features, error = checked_pose(frame)
                    attempts.append(dict(filepath=row["filepath"], label=label, split=row["split"], frame=frame_id,
                                         status="pose_error" if error else "ok" if features is not None else "decode_failed" if frame is None else "no_valid_pose", error=error))

                    if features is None:

                        failed += 1

                        continue

                    features, visibility = features
                    row_data = {

                        "label":
                            label,

                        "source":
                            source,

                        "split":
                            row["split"],

                        "subject":
                            row["subject"],

                        "filepath": row["filepath"],
                        "timestamp_seconds": timestamp,
                        "image_width": frame.shape[1],
                        "image_height": frame.shape[0],
                        "frame":
                            frame_id
                    }

                    row_data.update({f'visibility_{i}': value for i, value in enumerate(visibility)})
                    for i, value in enumerate(features):

                        row_data[
                            f"f_{i}"
                        ] = value

                    results.append(
                        row_data
                    )

                    success += 1

            # --------------------------------------------------------
            # IMAGE
            # --------------------------------------------------------

            else:

                image = read_image_unicode(
                    file_path
                )

                features, error = checked_pose(image)
                attempts.append(dict(filepath=row["filepath"], label=label, split=row["split"], frame=0,
                                     status="pose_error" if error else "ok" if features is not None else "decode_failed" if image is None else "no_valid_pose", error=error))

                if features is None:

                    failed += 1

                    continue

                features, visibility = features
                row_data = {

                    "label":
                        label,

                    "source":
                        source,

                    "split":
                        row["split"],

                    "subject":
                        row["subject"],

                    "filepath": row["filepath"],
                    "timestamp_seconds": None,
                    "image_width": image.shape[1],
                    "image_height": image.shape[0],
                    "frame":
                        0
                }

                row_data.update({f'visibility_{i}': value for i, value in enumerate(visibility)})
                for i, value in enumerate(features):

                    row_data[
                        f"f_{i}"
                    ] = value

                results.append(
                    row_data
                )

                success += 1

        # ============================================================
        # SAVE
        # ============================================================

    finally:
        pose.close()

    feature_df = pd.DataFrame(
        results
    )

    with log_path.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(attempts).to_csv(stream, index=False)
    if feature_df.empty:
        raise RuntimeError(f"No valid poses; see {log_path}")
    with output.open("x", encoding="utf-8", newline="") as stream:
        feature_df.to_csv(stream, index=False)

    print("\n" + "="*70)

    print("DONE")

    print("="*70)

    print(
        "Successful:",
        success
    )

    print(
        "Failed:",
        failed
    )

    print(
        "Output:"
    )

    print(
        output
    )

    print(
        "\nFeature shape:"
    )

    print(
        feature_df.shape
    )
    errors = sum(a['status'] in {'unreadable_video', 'video_error', 'decode_failed', 'pose_error'} for a in attempts)
    print(f'Attempt log: {log_path}; decoding/processing errors: {errors}')
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
