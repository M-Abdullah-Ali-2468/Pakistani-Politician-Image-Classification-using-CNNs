# ============================================================
# IDENTITY-AWARE FACE CROPPER v2
# ============================================================
#
# CHANGES:
#   ✅ similarity threshold = 0.50
#   ✅ consistent filename counting
#   ✅ saves as:
#         politician_0001.jpg
#         politician_0002.jpg
#   ✅ restart-safe counting
#   ✅ skips already existing files
#
# ============================================================

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import insightface


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
CROPPED_DIR = BASE_DIR / "data" / "cropped_faces"

os.makedirs(CROPPED_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SIMILARITY_THRESHOLD = 0.50

FACE_PADDING = 0.30


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("\nLoading InsightFace...\n")

app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)

app.prepare(ctx_id=0)

print("InsightFace loaded.\n")


# ============================================================
# LOAD REFERENCE EMBEDDINGS
# ============================================================

REFERENCE_EMBEDDINGS = {}

for file in os.listdir(EMBEDDINGS_DIR):

    if file.endswith(".npy"):

        politician = file.replace(".npy", "")

        emb = np.load(EMBEDDINGS_DIR / file).astype(np.float32)

        emb /= (np.linalg.norm(emb) + 1e-9)

        REFERENCE_EMBEDDINGS[politician] = emb

print(f"Loaded {len(REFERENCE_EMBEDDINGS)} reference embeddings.\n")


# ============================================================
# UTILS
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def crop_face(image, bbox):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = map(int, bbox)

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * FACE_PADDING)
    pad_y = int(bh * FACE_PADDING)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return image[y1:y2, x1:x2]


def get_starting_count(folder: Path, politician: str) -> int:

    existing = []

    for file in os.listdir(folder):

        if file.startswith(politician) and file.endswith(".jpg"):

            try:
                num = int(file.split("_")[-1].replace(".jpg", ""))
                existing.append(num)

            except:
                pass

    if not existing:
        return 1

    return max(existing) + 1


# ============================================================
# MAIN
# ============================================================

total_saved = 0
total_rejected = 0

print("=" * 60)
print("IDENTITY-AWARE FACE CROPPING")
print("=" * 60)

for politician in os.listdir(RAW_DIR):

    politician_dir = RAW_DIR / politician

    if not politician_dir.is_dir():
        continue

    print(f"\n[{politician}]")

    reference_embedding = REFERENCE_EMBEDDINGS.get(politician)

    if reference_embedding is None:
        print("  No reference embedding found.")
        continue

    output_dir = CROPPED_DIR / politician
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================
    # RESTART-SAFE COUNTER
    # ========================================================

    counter = get_starting_count(output_dir, politician)

    saved_count = 0
    rejected_count = 0

    for image_name in os.listdir(politician_dir):

        image_path = politician_dir / image_name

        try:

            # ====================================================
            # LOAD IMAGE
            # ====================================================

            image = cv2.imread(str(image_path))

            if image is None:
                rejected_count += 1
                continue

            # ====================================================
            # HANDLE GRAYSCALE
            # ====================================================

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # ====================================================
            # DETECT ALL FACES
            # ====================================================

            faces = app.get(image)

            if len(faces) == 0:
                rejected_count += 1
                continue

            # ====================================================
            # FIND BEST MATCH
            # ====================================================

            best_similarity = -1
            best_face = None

            for face in faces:

                emb = face.embedding.astype(np.float32)

                emb /= (np.linalg.norm(emb) + 1e-9)

                similarity = cosine_similarity(
                    emb,
                    reference_embedding
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_face = face

            # ====================================================
            # VERIFY IDENTITY
            # ====================================================

            if best_similarity < SIMILARITY_THRESHOLD:
                rejected_count += 1
                continue

            # ====================================================
            # CROP FACE
            # ====================================================

            cropped = crop_face(
                image,
                best_face.bbox
            )

            if cropped.size == 0:
                rejected_count += 1
                continue

            # ====================================================
            # SAVE
            # ====================================================

            filename = f"{politician}_{counter:04d}.jpg"

            save_path = output_dir / filename

            cv2.imwrite(str(save_path), cropped)

            print(
                f"  [✓] {filename} "
                f"| sim={best_similarity:.3f}"
            )

            counter += 1
            saved_count += 1
            total_saved += 1

        except Exception as e:

            rejected_count += 1

            print(f"  [ERR] {image_name} | {e}")

    total_rejected += rejected_count

    print(
        f"\n[{politician}] "
        f"saved={saved_count} "
        f"| rejected={rejected_count}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Total Saved     : {total_saved}")
print(f"Total Rejected  : {total_rejected}")

print("=" * 60)

print("\nCropping completed.\n")