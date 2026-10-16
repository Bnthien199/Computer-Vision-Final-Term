from pathlib import Path
from posefit_console import configure_console
configure_console()
from collections import defaultdict
import re

import cv2
import numpy as np


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parent
STANDING_DIR = ROOT / "standing_raw"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# READ IMAGE - SUPPORT UNICODE WINDOWS PATH
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
        print("Lỗi đọc ảnh:", e)
        return None


# ============================================================
# METADATA
# ============================================================

def extract_number(pattern, text):

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


def get_metadata(path):

    text = str(path)

    return {
        "subject": extract_number(
            r"subject[_\-\s]?(\d+)",
            text
        ),

        "activity": extract_number(
            r"activity[_\-\s]?(\d+)",
            text
        ),

        "trial": extract_number(
            r"trial[_\-\s]?(\d+)",
            text
        ),

        "camera": extract_number(
            r"camera[_\-\s]?(\d+)",
            text
        )
    }


# ============================================================
# START
# ============================================================

print("=" * 70)
print("POSEFIT - INSPECT STANDING RGB")
print("=" * 70)

print("\nStanding folder:")
print(STANDING_DIR)


if not STANDING_DIR.exists():

    print("❌ Không tìm thấy standing_raw")
    raise SystemExit


# ============================================================
# FIND IMAGES
# ============================================================

images = sorted([
    file
    for file in STANDING_DIR.rglob("*")
    if (
        file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    )
])


print("\nTổng số ảnh:", len(images))


if not images:

    print("❌ Không tìm thấy ảnh.")
    raise SystemExit


# ============================================================
# DISTRIBUTION
# ============================================================

groups = defaultdict(list)


for image in images:

    meta = get_metadata(image)

    key = (
        meta["subject"],
        meta["activity"],
        meta["trial"],
        meta["camera"]
    )

    groups[key].append(image)


print("\n" + "=" * 70)
print("PHÂN BỐ DỮ LIỆU")
print("=" * 70)


for key, files in sorted(
    groups.items(),
    key=lambda x: str(x[0])
):

    subject, activity, trial, camera = key

    print(
        f"Subject={subject} | "
        f"Activity={activity} | "
        f"Trial={trial} | "
        f"Camera={camera} | "
        f"Images={len(files)}"
    )


# ============================================================
# SAMPLE IMAGE
# ============================================================

sample = images[0]

print("\n" + "=" * 70)
print("ẢNH MẪU")
print("=" * 70)

print(sample)


image = imread_unicode(sample)


if image is None:

    print("❌ Không đọc được ảnh.")

else:

    height, width = image.shape[:2]

    print("✅ Đọc ảnh thành công")
    print("Width :", width)
    print("Height:", height)
    print("Shape :", image.shape)


print("\n5 file đầu:")

for file in images[:5]:

    print(
        " ",
        file.relative_to(ROOT)
    )


print("\nHoàn thành.")
