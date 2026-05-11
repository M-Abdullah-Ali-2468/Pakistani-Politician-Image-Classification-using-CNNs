# ============================================================
# DATASET CLEANING PIPELINE (STRICT MINIMUM 80 GUARANTEE)
# ============================================================
#
# STRATEGY:
#
#   We DO NOT aggressively remove images.
#
#   Instead:
#
#       - analyze image quality
#       - assign penalty scores
#       - remove ONLY worst images
#       - STRICTLY preserve minimum 80 images
#
# GUARANTEE:
#
#   If valid images >= 80:
#       -> exactly 80+ images saved
#
#   If valid images < 80:
#       -> keep ALL valid images
#
# IMPORTANT:
#
#   ONLY corrupted/unreadable images are rejected.
#
#   Tiny, blurry, dark images are NOT immediately removed.
#   They only receive higher penalty scores.
#
# ============================================================

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "data" / "cropped_faces"

OUTPUT_DIR = BASE_DIR / "data" / "final_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "cleaning_report.csv"


# ============================================================
# SETTINGS
# ============================================================

MIN_IMAGES_PER_CLASS = 80

MIN_WIDTH = 160
MIN_HEIGHT = 160

BLUR_THRESHOLD = 35

MIN_BRIGHTNESS = 25
MAX_BRIGHTNESS = 235

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
)


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


def resolution_valid(image):

    h, w = image.shape[:2]

    return (
        w >= MIN_WIDTH and
        h >= MIN_HEIGHT
    )


# ============================================================
# GLOBAL STATS
# ============================================================

GLOBAL_TOTAL = 0
GLOBAL_SAVED = 0
GLOBAL_REMOVED = 0

CSV_ROWS = []


# ============================================================
# START
# ============================================================

print("\n" + "=" * 70)
print("SMART DATASET CLEANING")
print("=" * 70)

politicians = sorted(os.listdir(INPUT_DIR))

for politician in politicians:

    politician_input_dir = INPUT_DIR / politician

    if not politician_input_dir.is_dir():
        continue

    politician_output_dir = OUTPUT_DIR / politician

    os.makedirs(
        politician_output_dir,
        exist_ok=True
    )

    print(f"\n[{politician}]")

    image_candidates = []

    image_files = sorted(
        os.listdir(politician_input_dir)
    )

    # ========================================================
    # ANALYZE ALL IMAGES
    # ========================================================

    for image_name in image_files:

        if not image_name.lower().endswith(
            IMAGE_EXTENSIONS
        ):
            continue

        GLOBAL_TOTAL += 1

        image_path = (
            politician_input_dir / image_name
        )

        try:

            image = cv2.imread(
                str(image_path)
            )

            # ------------------------------------------------
            # ONLY HARD REJECTION
            # ------------------------------------------------

            if image is None:

                print(
                    f"  [SKIP] {image_name} | corrupted"
                )

                continue

            if image.size == 0:

                print(
                    f"  [SKIP] {image_name} | empty"
                )

                continue

            # ------------------------------------------------
            # QUALITY METRICS
            # ------------------------------------------------

            blur_score = calculate_blur_score(
                image
            )

            brightness = calculate_brightness(
                image
            )

            h, w = image.shape[:2]

            # ------------------------------------------------
            # PENALTY SCORE
            #
            # Higher penalty = worse image
            # ------------------------------------------------

            penalty = 0

            # Blur penalty
            if blur_score < BLUR_THRESHOLD:

                penalty += (
                    BLUR_THRESHOLD - blur_score
                )

            # Dark penalty
            if brightness < MIN_BRIGHTNESS:

                penalty += (
                    MIN_BRIGHTNESS - brightness
                )

            # Bright penalty
            if brightness > MAX_BRIGHTNESS:

                penalty += (
                    brightness - MAX_BRIGHTNESS
                )

            # Tiny image penalty
            if not resolution_valid(image):

                penalty += 15

                print(
                    f"  [WARN] {image_name} "
                    f"| tiny={w}x{h}"
                )

            image_candidates.append({

                "image_name": image_name,

                "image_path": image_path,

                "blur": blur_score,

                "brightness": brightness,

                "penalty": penalty,

                "width": w,

                "height": h,
            })

        except Exception as e:

            print(
                f"  [ERR] {image_name} | {e}"
            )

    # ========================================================
    # SORT BY QUALITY
    # ========================================================

    image_candidates.sort(
        key=lambda x: x["penalty"]
    )

    total_images = len(image_candidates)

    # ========================================================
    # STRICT MINIMUM GUARANTEE
    # ========================================================

    if total_images < MIN_IMAGES_PER_CLASS:

        print(
            f"\n[WARNING] {politician} has only "
            f"{total_images} valid images."
        )

        print(
            "Keeping ALL valid images "
            "to preserve dataset size."
        )

    # ========================================================
    # DETERMINE HOW MANY TO REMOVE
    # ========================================================

    removable = max(
        0,
        total_images - MIN_IMAGES_PER_CLASS
    )

    removed_images = (
        image_candidates[-removable:]
        if removable > 0
        else []
    )

    removed_names = set(
        x["image_name"]
        for x in removed_images
    )

    # ========================================================
    # SAVE BEST IMAGES
    # ========================================================

    saved_count = 0
    removed_count = 0

    for item in image_candidates:

        image_name = item["image_name"]

        # ----------------------------------------------------
        # REMOVE WORST IMAGES ONLY
        # ----------------------------------------------------

        if image_name in removed_names:

            removed_count += 1
            GLOBAL_REMOVED += 1

            print(
                f"  [REMOVED] {image_name} "
                f"| penalty={item['penalty']:.1f} "
                f"| blur={item['blur']:.1f} "
                f"| brightness={item['brightness']:.1f}"
            )

            CSV_ROWS.append({
                "politician": politician,
                "original_name": image_name,
                "new_name": "",
                "status": "removed",
                "blur_score": round(item["blur"], 2),
                "brightness": round(item["brightness"], 2),
                "penalty": round(item["penalty"], 2),
                "width": item["width"],
                "height": item["height"],
            })

            continue

        # ----------------------------------------------------
        # CONSISTENT FILENAME
        # ----------------------------------------------------

        saved_count += 1
        GLOBAL_SAVED += 1

        new_filename = (
            f"{politician}_{saved_count:04d}.jpg"
        )

        output_path = (
            politician_output_dir / new_filename
        )

        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        shutil.copy2(
            item["image_path"],
            output_path
        )

        CSV_ROWS.append({
            "politician": politician,
            "original_name": image_name,
            "new_name": new_filename,
            "status": "saved",
            "blur_score": round(item["blur"], 2),
            "brightness": round(item["brightness"], 2),
            "penalty": round(item["penalty"], 2),
            "width": item["width"],
            "height": item["height"],
        })

        print(
            f"  [✓] {new_filename} "
            f"| blur={item['blur']:.1f} "
            f"| brightness={item['brightness']:.1f} "
            f"| saved={saved_count}"
        )

    # ========================================================
    # CLASS SUMMARY
    # ========================================================

    print("\n" + "-" * 60)

    print(f"[{politician}] SUMMARY")

    print(f"  Total Valid   : {total_images}")
    print(f"  Saved         : {saved_count}")
    print(f"  Removed       : {removed_count}")
    print(f"  Minimum Kept  : {MIN_IMAGES_PER_CLASS}")

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
            "blur_score",
            "brightness",
            "penalty",
            "width",
            "height",
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

print("=" * 70)

print("\nCleaning completed.\n")