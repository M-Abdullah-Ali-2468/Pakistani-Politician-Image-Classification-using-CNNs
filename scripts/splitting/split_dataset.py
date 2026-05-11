from __future__ import annotations

import random
import shutil
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "data" / "final_dataset"

OUTPUT_DIR = BASE_DIR / "data" / "split"


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

RANDOM_SEED = 42

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split_name in ["train", "val", "test"]:

    split_dir = OUTPUT_DIR / split_name

    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# START
# ============================================================

print("\n" + "=" * 70)
print("DATASET SPLITTING")
print("=" * 70)

politicians = sorted(INPUT_DIR.iterdir())

GLOBAL_TRAIN = 0
GLOBAL_VAL = 0
GLOBAL_TEST = 0

for politician_dir in politicians:

    if not politician_dir.is_dir():
        continue

    politician_name = politician_dir.name

    print(f"\n[{politician_name}]")

    # ========================================================
    # LOAD IMAGES
    # ========================================================

    image_files = [

        image_path

        for image_path in politician_dir.iterdir()

        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(image_files)

    total_images = len(image_files)

    # ========================================================
    # SPLIT COUNTS
    # ========================================================

    train_count = int(
        total_images * TRAIN_RATIO
    )

    val_count = int(
        total_images * VAL_RATIO
    )

    test_count = (
        total_images
        - train_count
        - val_count
    )

    # ========================================================
    # SPLIT FILES
    # ========================================================

    train_files = image_files[:train_count]

    val_files = image_files[
        train_count:
        train_count + val_count
    ]

    test_files = image_files[
        train_count + val_count:
    ]

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    train_output_dir = (
        OUTPUT_DIR
        / "train"
        / politician_name
    )

    val_output_dir = (
        OUTPUT_DIR
        / "val"
        / politician_name
    )

    test_output_dir = (
        OUTPUT_DIR
        / "test"
        / politician_name
    )

    train_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    val_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    test_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # COPY TRAIN FILES
    # ========================================================

    for image_path in train_files:

        shutil.copy2(
            image_path,
            train_output_dir / image_path.name
        )

    # ========================================================
    # COPY VAL FILES
    # ========================================================

    for image_path in val_files:

        shutil.copy2(
            image_path,
            val_output_dir / image_path.name
        )

    # ========================================================
    # COPY TEST FILES
    # ========================================================

    for image_path in test_files:

        shutil.copy2(
            image_path,
            test_output_dir / image_path.name
        )

    # ========================================================
    # STATS
    # ========================================================

    GLOBAL_TRAIN += len(train_files)
    GLOBAL_VAL += len(val_files)
    GLOBAL_TEST += len(test_files)

    print(f"  Total Images : {total_images}")
    print(f"  Train Images : {len(train_files)}")
    print(f"  Val Images   : {len(val_files)}")
    print(f"  Test Images  : {len(test_files)}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SPLIT SUMMARY")
print("=" * 70)

print(f"Train Images : {GLOBAL_TRAIN}")
print(f"Val Images   : {GLOBAL_VAL}")
print(f"Test Images  : {GLOBAL_TEST}")

print("=" * 70)

print("\nDataset splitting completed.\n")