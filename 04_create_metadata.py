import argparse
from posefit_pipeline import new_output, validate_metadata, validate_features, FEATURE_COLUMNS
from pathlib import Path
from posefit_console import configure_console
import pandas as pd
import random

# =========================================================
# CONFIG
# =========================================================

ROOT = Path(__file__).resolve().parent

GYM_DIR = (
    ROOT
    / "archive"
    / "verified_data"
)

STANDING_DIR = (
    ROOT
    / "standing_raw"
)

OUTPUT_FILE = (
    ROOT
    / "runs" / "metadata.csv"
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

# =========================================================
# LABEL MAPPING
# =========================================================

GYM_CLASSES = {

    "squat": [
        "squat"
    ],

    "pushup": [
        "push-up",
        "pushup",
        "push_up"
    ],

    "plank": [
        "plank"
    ]
}

# =========================================================
# FIND GYM VIDEOS
# =========================================================

def create_gym_metadata():

    data = []

    print("\n========== GYM DATA ==========")

    videos = []

    for file in sorted(GYM_DIR.rglob("*")):

        if (
            file.is_file()
            and file.suffix.lower()
            in VIDEO_EXTENSIONS
        ):

            videos.append(file)

    for label, names in GYM_CLASSES.items():

        class_files = []

        for file in videos:

            folder_name = (
                file.parent.name.lower()
            )

            if folder_name in names:

                class_files.append(file)

        random.shuffle(class_files)

        n = len(class_files)
        if n < 10:
            raise ValueError(f'Not enough videos to populate all splits: {label}')

        train_end = int(
            n * 0.8
        )

        val_end = int(
            n * 0.9
        )

        for i, file in enumerate(class_files):

            if i < train_end:

                split = "train"

            elif i < val_end:

                split = "val"

            else:

                split = "test"

            data.append({

                "filepath":
                    str(file.relative_to(ROOT)),

                "label":
                    label,

                "source":
                    "gym",

                "subject":
                    None,

                "split":
                    split
            })

        print(
            f"{label}: {n} videos"
        )

    return data

# =========================================================
# FIND STANDING IMAGES
# =========================================================

def create_standing_metadata():

    data = []

    print("\n========== STANDING DATA ==========")

    subjects = sorted(
        [
            folder
            for folder in STANDING_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    for subject_folder in subjects:

        subject_name = (
            subject_folder.name
        )

        if subject_name.upper() not in {f'SUBJECT{i}' for i in range(1, 6)}:
            raise ValueError(f'Unknown subject folder: {subject_name}')
        subject_id = int(
            ''.join(
                filter(
                    str.isdigit,
                    subject_name
                )
            )
        )

        images = []

        for file in sorted(subject_folder.rglob("*")):

            if (
                file.is_file()
                and file.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                images.append(file)

        # lấy mẫu đều
        random.shuffle(images)

        SAMPLE_PER_SUBJECT = 100

        if len(images) > SAMPLE_PER_SUBJECT:

            step = (
                len(images)
                //
                SAMPLE_PER_SUBJECT
            )

            images = images[
                ::step
            ][:SAMPLE_PER_SUBJECT]

        # split theo subject

        if subject_id in [1,2,3]:

            split = "train"

        elif subject_id == 4:

            split = "val"

        else:

            split = "test"

        for file in images:

            data.append({

                "filepath":
                    str(file.relative_to(ROOT)),

                "label":
                    "standing",

                "source":
                    "standing",

                "subject":
                    subject_id,

                "split":
                    split
            })

        print(
            f"Subject {subject_id}: "
            f"{len(images)} images "
            f"-> {split}"
        )

    return data

# =========================================================
# MAIN
# =========================================================

def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    print("="*70)
    print("POSEFIT - CREATE METADATA")
    print("="*70)

    metadata = []

    metadata.extend(
        create_gym_metadata()
    )

    metadata.extend(
        create_standing_metadata()
    )

    df = pd.DataFrame(
        metadata
    )

    df = df.sample(
        frac=1,
        random_state=args.seed
    ).reset_index(
        drop=True
    )

    validate_metadata(df)
    output = new_output(args.output)
    with output.open("x", encoding="utf-8", newline="") as stream:
        df.to_csv(stream, index=False)

    print("\n" + "="*70)

    print(
        "Saved:"
    )

    print(
        output
    )

    print("\nDATASET SUMMARY")
    print(
        df.groupby(
            [
                "label",
                "split"
            ]
        ).size()
    )

    print("\nTotal samples:")
    print(
        len(df)
    )

    print("\nHoàn thành.")

if __name__ == "__main__":
    main()
