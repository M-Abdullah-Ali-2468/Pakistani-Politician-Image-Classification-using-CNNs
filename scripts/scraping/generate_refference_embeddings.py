from pathlib import Path
import os

# pyrefly: ignore [missing-import]
import cv2

import numpy as np

# pyrefly: ignore [missing-import]
import insightface


# =========================================================
# PROJECT PATHS
# =========================================================

# Current file:
# scripts/scraping/generate_reference_embeddings.py

BASE_DIR = Path(__file__).resolve().parents[2]

REFERENCE_DIR = BASE_DIR / "data" / "refferences"
EMBEDDING_DIR = BASE_DIR / "data" / "embeddings"

os.makedirs(EMBEDDING_DIR, exist_ok=True)

print(f"\nREFERENCE_DIR: {REFERENCE_DIR}")
print(f"EMBEDDING_DIR: {EMBEDDING_DIR}")


# =========================================================
# LOAD INSIGHTFACE MODEL
# =========================================================

app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0)


# =========================================================
# PROCESS EACH POLITICIAN
# =========================================================

for politician in os.listdir(REFERENCE_DIR):

    person_dir = REFERENCE_DIR / politician

    if not person_dir.is_dir():
        continue

    embeddings = []

    print(f"\n=================================================")
    print(f"Processing: {politician}")
    print(f"=================================================")

    for image_name in os.listdir(person_dir):

        image_path = person_dir / image_name

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[ERROR] Could not read image: {image_name}")
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print(f"[NO FACE] {image_name}")
            continue

        # =================================================
        # TAKE LARGEST FACE
        # =================================================

        face = max(
            faces,
            key=lambda x: (
                (x.bbox[2] - x.bbox[0]) *
                (x.bbox[3] - x.bbox[1])
            )
        )

        embedding = face.embedding

        # normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        embeddings.append(embedding)

        print(f"[OK] Face extracted: {image_name}")

    # =====================================================
    # CHECK IF EMBEDDINGS EXIST
    # =====================================================

    if len(embeddings) == 0:
        print(f"[WARNING] No embeddings generated for {politician}")
        continue

    # =====================================================
    # AVERAGE EMBEDDING
    # =====================================================

    avg_embedding = np.mean(embeddings, axis=0)

    # normalize final embedding
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    # =====================================================
    # SAVE EMBEDDING
    # =====================================================

    save_path = EMBEDDING_DIR / f"{politician}.npy"

    np.save(save_path, avg_embedding)

    print(f"\n[SAVED] {save_path}")
    print(f"[INFO] Total embeddings used: {len(embeddings)}")


print("\n=================================================")
print("ALL REFERENCE EMBEDDINGS GENERATED SUCCESSFULLY")
print("=================================================")