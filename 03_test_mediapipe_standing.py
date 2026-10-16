import argparse
from posefit_pipeline import new_output
from pathlib import Path
from posefit_console import configure_console

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parent

STANDING_DIR = ROOT / "standing_raw"

OUTPUT_DIR = ROOT / "runs/standing_preview"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# UNICODE-SAFE IMAGE READ
# ============================================================

def imread_unicode(path):

    try:

        data = np.fromfile(
            str(path),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception as e:

        print(
            "Lỗi đọc ảnh:",
            e
        )

        return None


# ============================================================
# UNICODE-SAFE IMAGE SAVE
# ============================================================

def imwrite_unicode(path, image):

    try:

        extension = Path(path).suffix

        success, encoded = cv2.imencode(
            extension,
            image
        )

        if not success:
            return False

        encoded.tofile(
            str(path)
        )

        return True

    except Exception as e:

        print(
            "Lỗi lưu ảnh:",
            e
        )

        return False


# ============================================================
# FIND IMAGES
# ============================================================

def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    images = sorted([
        file
        for file in STANDING_DIR.rglob("*")
        if (
            file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ])


    print("=" * 70)
    print("POSEFIT - TEST MEDIAPIPE STANDING")
    print("=" * 70)

    print(
        "\nMediaPipe version:",
        mp.__version__
    )

    print(
        "Có mp.solutions:",
        hasattr(mp, "solutions")
    )

    print(
        "\nTổng ảnh tìm thấy:",
        len(images)
    )


    if not images:

        print(
            "❌ Không tìm thấy ảnh."
        )

        raise SystemExit


    # ============================================================
    # MEDIAPIPE POSE
    # ============================================================

    output_dir = new_output(args.output, directory=True)
    mp_pose = mp.solutions.pose

    mp_drawing = (
        mp.solutions.drawing_utils
    )


    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )


    # ============================================================
    # SAMPLE EVENLY
    # ============================================================

    TEST_COUNT = min(
        20,
        len(images)
    )


    if len(images) > TEST_COUNT:

        indices = np.linspace(
            0,
            len(images) - 1,
            TEST_COUNT,
            dtype=int
        )

        test_images = [
            images[i]
            for i in indices
        ]

    else:

        test_images = images


    # ============================================================
    # TEST
    # ============================================================

    success = 0
    failed = 0


    for i, image_path in enumerate(
        test_images,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"[{i}/{len(test_images)}]"
        )

        print(
            image_path.relative_to(ROOT)
        )


        image = imread_unicode(
            image_path
        )


        if image is None:

            print(
                "❌ Không đọc được ảnh."
            )

            failed += 1

            continue


        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        results = pose.process(
            rgb
        )


        if not results.pose_landmarks:

            print(
                "❌ MediaPipe không phát hiện pose."
            )

            failed += 1

            continue


        landmarks = (
            results
            .pose_landmarks
            .landmark
        )


        print(
            "✅ Phát hiện:",
            len(landmarks),
            "landmarks"
        )


        if len(landmarks) == 33:

            success += 1

        else:

            failed += 1


        # ========================================================
        # PRINT FIRST LANDMARKS
        # ========================================================

        print(
            "Landmark 0:"
        )

        lm = landmarks[0]

        print(
            f"x={lm.x:.4f}, "
            f"y={lm.y:.4f}, "
            f"z={lm.z:.4f}, "
            f"visibility="
            f"{lm.visibility:.4f}"
        )


        # ========================================================
        # DRAW SKELETON
        # ========================================================

        annotated = image.copy()


        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        output_file = (
            output_dir
            / f"standing_pose_{i:02d}.jpg"
        )


        saved = imwrite_unicode(
            output_file,
            annotated
        )


        if saved:

            print(
                "✅ Saved:",
                output_file.name
            )

        else:

            print(
                "❌ Không lưu được ảnh."
            )


    # ============================================================
    # CLOSE
    # ============================================================

    pose.close()


    # ============================================================
    # RESULT
    # ============================================================

    print("\n" + "=" * 70)
    print("KẾT QUẢ")
    print("=" * 70)

    print(
        "Ảnh test  :",
        len(test_images)
    )

    print(
        "Thành công:",
        success
    )

    print(
        "Thất bại  :",
        failed
    )


    if test_images:

        detection_rate = (
            success
            / len(test_images)
            * 100
        )

        print(
            f"Detection rate: "
            f"{detection_rate:.2f}%"
        )


    print(
        "\nẢnh skeleton:"
    )

    print(
        output_dir
    )

if __name__ == "__main__":
    main()
