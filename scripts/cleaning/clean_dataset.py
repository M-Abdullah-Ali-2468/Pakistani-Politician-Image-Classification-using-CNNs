# ============================================================
# CLEAN CROPPED DATASET + STANDARDIZE IMAGE SIZE
# ============================================================
#
# PURPOSE:
#
#   Load images from:
#
#       data/cropped_faces
#
#   Clean + standardize them into:
#
#       data/final_dataset
#
#
# WHAT THIS SCRIPT DOES:
#
#   ✅ keeps minimum 80 images per class
#   ✅ removes only worst images
#   ✅ consistent naming
#   ✅ resizes ALL images to 224x224
#   ✅ saves CSV cleaning report
#
#
# OUTPUT EXAMPLE:
#
#   data/final_dataset/
#
#       imran_khan/
#           imran_khan_0001.jpg
#           imran_khan_0002.jpg
#
#       maryam_nawaz/
#           maryam_nawaz_0001.jpg
#
# ============================================================

import csv
from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path.cwd()

INPUT_DIR = BASE_DIR / "data" / "cropped_faces"

OUTPUT_DIR = BASE_DIR / "data" / "final_dataset"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = OUTPUT_DIR / "cleaning_report.csv"


# ============================================================
# SETTINGS
# ============================================================

MIN_IMAGES_PER_CLASS = 80

IMAGE_SIZE = (224, 224)

BLUR_THRESHOLD = 35

MIN_BRIGHTNESS = 25
MAX_BRIGHTNESS = 235

MIN_WIDTH = 160
MIN_HEIGHT = 160

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
)


# ============================================================
# GLOBAL STATS
# ============================================================

GLOBAL_TOTAL = 0
GLOBAL_SAVED = 0
GLOBAL_REMOVED = 0

CSV_ROWS = []


# ============================================================
# UTILITIES
# ============================================================

def calculate_blur_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


def calculate_brightness(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(np.mean(gray))


# ============================================================
# START
# ============================================================

print("\n" + "=" * 70)
print("CLEANING + STANDARDIZATION")
print("=" * 70)

politicians = sorted(INPUT_DIR.iterdir())

for politician_dir in politicians:

    if not politician_dir.is_dir():
        continue

    politician_name = politician_dir.name

    print(f"\n[{politician_name}]")

    output_class_dir = (
        OUTPUT_DIR / politician_name
    )

    output_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_candidates = []

    image_files = [

        image_path

        for image_path in politician_dir.iterdir()

        if image_path.suffix.lower()
        in IMAGE_EXTENSIONS
    ]

    # ========================================================
    # ANALYZE IMAGES
    # ========================================================

    for image_path in image_files:

        GLOBAL_TOTAL += 1

        try:

            image = cv2.imread(
                str(image_path)
            )

            # ------------------------------------------------
            # HARD REJECTION
            # ------------------------------------------------

            if image is None:

                print(
                    f"  [SKIP] {image_path.name}"
                )

                continue

            h, w = image.shape[:2]

            blur_score = calculate_blur_score(
                image
            )

            brightness = calculate_brightness(
                image
            )

            # ------------------------------------------------
            # PENALTY SCORE
            # ------------------------------------------------

            penalty = 0

            # Blur penalty
            if blur_score < BLUR_THRESHOLD:

                penalty += (
                    BLUR_THRESHOLD - blur_score
                )

            # Brightness penalties
            if brightness < MIN_BRIGHTNESS:

                penalty += (
                    MIN_BRIGHTNESS - brightness
                )

            if brightness > MAX_BRIGHTNESS:

                penalty += (
                    brightness - MAX_BRIGHTNESS
                )

            # Small resolution penalty
            if (
                w < MIN_WIDTH or
                h < MIN_HEIGHT
            ):

                penalty += 15

            image_candidates.append({

                "path": image_path,

                "blur": blur_score,

                "brightness": brightness,

                "penalty": penalty,

                "width": w,

                "height": h,
            })

        except Exception as e:

            print(
                f"  [ERR] {image_path.name} | {e}"
            )

    # ========================================================
    # SORT BY QUALITY
    # ========================================================

    image_candidates.sort(
        key=lambda x: x["penalty"]
    )

    total_valid = len(image_candidates)

    removable = max(
        0,
        total_valid - MIN_IMAGES_PER_CLASS
    )

    removed_images = (
        image_candidates[-removable:]
        if removable > 0
        else []
    )

    removed_paths = set(
        x["path"]
        for x in removed_images
    )

    # ========================================================
    # SAVE BEST IMAGES
    # ========================================================

    saved_count = 0
    removed_count = 0

    for item in image_candidates:

        image_path = item["path"]

        # ----------------------------------------------------
        # REMOVE WORST ONLY
        # ----------------------------------------------------

        if image_path in removed_paths:

            removed_count += 1
            GLOBAL_REMOVED += 1

            print(
                f"  [REMOVED] {image_path.name} "
                f"| penalty={item['penalty']:.1f}"
            )

            CSV_ROWS.append({

                "politician": politician_name,

                "original_name": image_path.name,

                "new_name": "",

                "status": "removed",

                "blur": round(
                    item["blur"], 2
                ),

                "brightness": round(
                    item["brightness"], 2
                ),

                "penalty": round(
                    item["penalty"], 2
                ),
            })

            continue

        # ----------------------------------------------------
        # LOAD IMAGE AGAIN
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        # ----------------------------------------------------
        # STANDARDIZE SIZE
        # ----------------------------------------------------

        resized = cv2.resize(
            image,
            IMAGE_SIZE
        )

        # ----------------------------------------------------
        # CONSISTENT FILENAME
        # ----------------------------------------------------

        saved_count += 1
        GLOBAL_SAVED += 1

        new_filename = (
            f"{politician_name}_"
            f"{saved_count:04d}.jpg"
        )

        output_path = (
            output_class_dir / new_filename
        )

        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        cv2.imwrite(
            str(output_path),
            resized
        )

        print(
            f"  [✓] {new_filename} "
            f"| saved={saved_count}"
        )

        CSV_ROWS.append({

            "politician": politician_name,

            "original_name": image_path.name,

            "new_name": new_filename,

            "status": "saved",

            "blur": round(
                item["blur"], 2
            ),

            "brightness": round(
                item["brightness"], 2
            ),

            "penalty": round(
                item["penalty"], 2
            ),
        })

    # ========================================================
    # CLASS SUMMARY
    # ========================================================

    print("\n" + "-" * 60)

    print(f"[{politician_name}] SUMMARY")

    print(f"  Valid Images : {total_valid}")
    print(f"  Saved        : {saved_count}")
    print(f"  Removed      : {removed_count}")

    print("-" * 60)


# ============================================================
# SAVE CSV REPORT
# ============================================================

with open(
    CSV_PATH,
    mode="w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(

        csv_file,

        fieldnames=[

            "politician",

            "original_name",

            "new_name",

            "status",

            "blur",

            "brightness",

            "penalty",
        ]
    )

    writer.writeheader()

    writer.writerows(CSV_ROWS)

print(f"\nCSV report saved at:\n{CSV_PATH}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Total Images : {GLOBAL_TOTAL}")
print(f"Saved        : {GLOBAL_SAVED}")
print(f"Removed      : {GLOBAL_REMOVED}")

print(
    f"Final Size   : "
    f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
)

print("=" * 70)

print("\nCleaning completed.\n")