from pathlib import Path
from posefit_console import configure_console
configure_console()
import csv

# =====================================================
# 1. ĐƯỜNG DẪN
# =====================================================

ROOT = Path(__file__).resolve().parent

GYM_DIR = ROOT / "archive" / "verified_data"
STANDING_DIR = ROOT / "12773013"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}

print("=" * 70)
print("POSEFIT - DATASET INSPECTION")
print("=" * 70)

print("\nROOT:")
print(ROOT)

print("\nGYM:")
print(GYM_DIR)

print("\nSTANDING:")
print(STANDING_DIR)


# =====================================================
# 2. KIỂM TRA GYM DATASET
# =====================================================

print("\n")
print("=" * 70)
print("1. GYM DATASET")
print("=" * 70)

if not GYM_DIR.exists():
    print("❌ Không tìm thấy verified_data")
else:
    total_videos = 0

    for folder in sorted(GYM_DIR.rglob("*")):

        if not folder.is_dir():
            continue

        videos = [
            f for f in folder.iterdir()
            if f.is_file()
            and f.suffix.lower() in VIDEO_EXTENSIONS
        ]

        if videos:
            relative_folder = folder.relative_to(GYM_DIR)

            print(
                f"{relative_folder}: "
                f"{len(videos)} videos"
            )

            total_videos += len(videos)

    print("-" * 70)
    print("Tổng số video:", total_videos)


# =====================================================
# 3. TÌM VIDEO LIÊN QUAN ĐẾN 3 CLASS
# =====================================================

print("\n")
print("=" * 70)
print("2. TÌM CLASS SQUAT / PUSH-UP / PLANK")
print("=" * 70)

keywords = {
    "squat": ["squat"],
    "pushup": [
        "pushup",
        "push-up",
        "push_up",
        "push up"
    ],
    "plank": ["plank"]
}

class_videos = {
    "squat": [],
    "pushup": [],
    "plank": []
}

if GYM_DIR.exists():

    for file in GYM_DIR.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower() in VIDEO_EXTENSIONS
        ):

            full_name = str(file).lower()

            for class_name, words in keywords.items():

                if any(word in full_name for word in words):
                    class_videos[class_name].append(file)
                    break


for class_name, files in class_videos.items():

    print(
        f"{class_name.upper():10s}: "
        f"{len(files)} videos"
    )

    for f in files[:3]:
        print("   ", f.relative_to(GYM_DIR))

    if len(files) > 3:
        print("    ...")


# =====================================================
# 4. KIỂM TRA STANDING
# =====================================================

print("\n")
print("=" * 70)
print("3. STANDING DATASET - ACTIVITY A7")
print("=" * 70)

standing_files = []

for subject_num in range(1, 6):

    subject_dir = STANDING_DIR / f"SUBJECT{subject_num}"

    if not subject_dir.exists():

        print(
            f"❌ SUBJECT{subject_num}: "
            "không tìm thấy folder"
        )

        continue

    csv_files = list(
        subject_dir.rglob("*.csv")
    )

    # A7 = Standing
    a7_files = [
        f for f in csv_files
        if "A7" in f.name.upper()
    ]

    standing_files.extend(a7_files)

    print(
        f"SUBJECT{subject_num}: "
        f"{len(csv_files)} CSV total | "
        f"{len(a7_files)} Standing A7"
    )

    for f in a7_files[:5]:
        print("   ", f.name)

    if len(a7_files) > 5:
        print("    ...")


print("-" * 70)
print(
    "Tổng Standing A7:",
    len(standing_files)
)


# =====================================================
# 5. ĐỌC HEADER CỦA 1 FILE STANDING
# =====================================================

print("\n")
print("=" * 70)
print("4. CẤU TRÚC FILE STANDING")
print("=" * 70)

if standing_files:

    sample_file = standing_files[0]

    print("Sample file:")
    print(sample_file)

    try:

        with open(
            sample_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.reader(f)

            header = next(reader)
            first_row = next(reader, None)

        print("\nSố cột:")
        print(len(header))

        print("\nTên các cột:")
        for i, column in enumerate(header):
            print(f"{i:3d}: {column}")

        if first_row:

            print("\n10 giá trị đầu của dòng đầu tiên:")

            for i in range(
                min(10, len(first_row))
            ):
                print(
                    f"{header[i]} = "
                    f"{first_row[i]}"
                )

    except Exception as e:

        print("❌ Không đọc được CSV:")
        print(e)

else:

    print(
        "❌ Không tìm thấy file A7."
    )


# =====================================================
# 6. KẾT LUẬN NHANH
# =====================================================

print("\n")
print("=" * 70)
print("5. SUMMARY")
print("=" * 70)

print(
    "Squat videos  :",
    len(class_videos["squat"])
)

print(
    "Push-up videos:",
    len(class_videos["pushup"])
)

print(
    "Plank videos  :",
    len(class_videos["plank"])
)

print(
    "Standing files:",
    len(standing_files)
)

print("\nHoàn thành kiểm tra dataset.")
