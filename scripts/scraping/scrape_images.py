# =========================================================
# ADVANCED VERIFIED POLITICIAN DATASET SCRAPER
# =========================================================
# Features:
# ---------------------------------------------------------
# ✅ Parallel scraping (3 politicians simultaneously)
# ✅ Google Images scraping
# ✅ InsightFace + ArcFace verification
# ✅ Single-face-only filtering
# ✅ Duplicate detection
# ✅ Text-overlay rejection
# ✅ Blur / quality filtering
# ✅ Similarity threshold filtering
# ✅ Save ONLY accepted images
# ✅ Automatic rejected image discard
# ✅ Accepted / rejected counters
# ✅ Async Playwright architecture
# =========================================================

from pathlib import Path
import os
import re
import random
import asyncio
import hashlib
from io import BytesIO

import requests
import numpy as np
from PIL import Image, ImageFilter
from scipy.fft import dct

# pyrefly: ignore [missing-import]
import cv2

# pyrefly: ignore [missing-import]
import insightface

# pyrefly: ignore [missing-import]
from playwright.async_api import async_playwright


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FINAL_DATASET_DIR = BASE_DIR / "data" / "final_dataset"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

os.makedirs(FINAL_DATASET_DIR, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

MAX_IMAGES_PER_PERSON = 100

MIN_WIDTH = 400
MIN_HEIGHT = 400
MIN_FILE_SIZE = 30000

PHASH_THRESHOLD = 5

SIMILARITY_THRESHOLD = 0.50

MAX_CONCURRENT_POLITICIANS = 3


# =========================================================
# POLITICIANS
# =========================================================

POLITICIANS = {

    "asif_ali_zardari": [
        "only Asif Ali Zardari solo HD face",
        "only Asif Ali Zardari close up solo pic",
        "only Asif Ali Zardari official solo image",
        "only Asif Ali Zardari high quality face",
        "only Asif Ali Zardari single person photo"
    ],

    "benazir_bhutto": [
        "only Benazir Bhutto solo HD face",
        "only Benazir Bhutto close up solo pic",
        "only Benazir Bhutto official solo image",
        "only Benazir Bhutto high quality face",
        "only Benazir Bhutto single person photo"
    ],

    "bilawal_bhutto": [
        "only Bilawal Bhutto Zardari solo HD face",
        "only Bilawal Bhutto close up solo pic",
        "only Bilawal Bhutto official solo image",
        "only Bilawal Bhutto high quality face",
        "only Bilawal Bhutto single person photo"
    ],

    "fazl_ur_rehman": [
        "only Maulana Fazlur Rehman solo HD face",
        "only Fazl ur Rehman close up solo pic",
        "only Maulana Fazlur Rehman official solo image",
        "only Fazl ur Rehman high quality face",
        "only Fazl ur Rehman single person photo"
    ],

    "hina_rabbani_khar": [
        "only Hina Rabbani Khar solo HD face",
        "only Hina Rabbani Khar close up solo pic",
        "only Hina Rabbani Khar official solo image",
        "only Hina Rabbani Khar high quality face",
        "only Hina Rabbani Khar single person photo"
    ],

    "imran_khan": [
        "only Imran Khan solo HD face",
        "only Imran Khan close up solo pic",
        "only Imran Khan official solo image",
        "only Imran Khan high quality face",
        "only Imran Khan single person photo"
    ],

    "khawaja_asif": [
        "only Khawaja Asif solo HD face",
        "only Khawaja Asif close up solo pic",
        "only Khawaja Asif official solo image",
        "only Khawaja Asif high quality face",
        "only Khawaja Asif single person photo"
    ],

    "maryam_nawaz": [
        "only Maryam Nawaz solo HD face",
        "only Maryam Nawaz close up solo pic",
        "only Maryam Nawaz official solo image",
        "only Maryam Nawaz high quality face",
        "only Maryam Nawaz single person photo"
    ],

    "nawaz_sharif": [
        "only Nawaz Sharif solo HD face",
        "only Nawaz Sharif close up solo pic",
        "only Nawaz Sharif official solo image",
        "only Nawaz Sharif high quality face",
        "only Nawaz Sharif single person photo"
    ],

    "pervez_musharraf": [
        "only Pervez Musharraf solo HD face",
        "only Pervez Musharraf close up solo pic",
        "only Pervez Musharraf official solo image",
        "only Pervez Musharraf high quality face",
        "only Pervez Musharraf single person photo"
    ],

    "shah_mahmood_qureshi": [
        "only Shah Mahmood Qureshi solo HD face",
        "only Shah Mahmood Qureshi close up solo pic",
        "only Shah Mahmood Qureshi official solo image",
        "only Shah Mahmood Qureshi high quality face",
        "only Shah Mahmood Qureshi single person photo"
    ],

    "shehbaz_sharif": [
        "only Shehbaz Sharif solo HD face",
        "only Shehbaz Sharif close up solo pic",
        "only Shehbaz Sharif official solo image",
        "only Shehbaz Sharif high quality face",
        "only Shehbaz Sharif single person photo"
    ],

    "sheikh_rasheed": [
        "only Sheikh Rasheed Ahmed solo HD face",
        "only Sheikh Rasheed close up solo pic",
        "only Sheikh Rasheed official solo image",
        "only Sheikh Rasheed high quality face",
        "only Sheikh Rasheed single person photo"
    ],

    "siraj_ul_haq": [
        "only Siraj ul Haq solo HD face",
        "only Siraj ul Haq close up solo pic",
        "only Siraj ul Haq official solo image",
        "only Siraj ul Haq high quality face",
        "only Siraj ul Haq single person photo"
    ],

    "yousuf_raza_gillani": [
        "only Yousuf Raza Gillani solo HD face",
        "only Yousuf Raza Gillani close up solo pic",
        "only Yousuf Raza Gillani official solo image",
        "only Yousuf Raza Gillani high quality face",
        "only Yousuf Raza Gillani single person photo"
    ]
}

# =========================================================
# LOAD INSIGHTFACE
# =========================================================

print("\nLoading InsightFace Model...\n")

app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0)

print("InsightFace Loaded Successfully.\n")


# =========================================================
# LOAD REFERENCE EMBEDDINGS
# =========================================================

REFERENCE_EMBEDDINGS = {}

for file in os.listdir(EMBEDDINGS_DIR):

    if not file.endswith(".npy"):
        continue

    politician = file.replace(".npy", "")

    path = EMBEDDINGS_DIR / file

    embedding = np.load(path)

    REFERENCE_EMBEDDINGS[politician] = embedding

print(f"Loaded {len(REFERENCE_EMBEDDINGS)} reference embeddings.\n")


# =========================================================
# TEXT FILTER
# =========================================================

def has_too_much_text(image_bytes):

    try:

        img = Image.open(
            BytesIO(image_bytes)
        ).convert("L")

        if img.width > 600:

            scale = 600 / img.width

            img = img.resize(
                (
                    600,
                    int(img.height * scale)
                ),
                Image.LANCZOS
            )

        arr = np.array(img)

        top = arr[:70, :]
        bottom = arr[-70:, :]

        for strip in [top, bottom]:

            edges = np.array(
                Image.fromarray(strip).filter(
                    ImageFilter.FIND_EDGES
                )
            )

            density = (
                (edges > 25).sum()
                / edges.size
            )

            if density > 0.12:
                return True

        return False

    except:
        return False


# =========================================================
# PHASH
# =========================================================

def compute_phash(image_bytes):

    try:

        img = Image.open(
            BytesIO(image_bytes)
        ).convert("L").resize(
            (32, 32),
            Image.LANCZOS
        )

        arr = np.array(
            img,
            dtype=float
        )

        dct2d = dct(
            dct(arr, axis=0, norm="ortho"),
            axis=1,
            norm="ortho"
        )

        low = dct2d[:8, :8].flatten()

        median = np.median(low)

        return "".join(
            "1" if v > median else "0"
            for v in low
        )

    except:
        return None


def hamming(a, b):

    return sum(
        x != y
        for x, y in zip(a, b)
    )


def is_duplicate(image_bytes, seen_hashes):

    md5 = hashlib.md5(
        image_bytes
    ).hexdigest()

    if md5 in seen_hashes:
        return True

    ph = compute_phash(image_bytes)

    if ph is not None:

        for existing in seen_hashes:

            if len(existing) == 64:

                dist = hamming(ph, existing)

                if dist <= PHASH_THRESHOLD:
                    return True

    seen_hashes.append(md5)

    if ph is not None:
        seen_hashes.append(ph)

    return False


# =========================================================
# FACE VERIFICATION
# =========================================================

def verify_face(image_bytes, politician):

    try:

        np_arr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return False, 0.0, "Invalid image"

        faces = app.get(image)

        # =================================================
        # ONLY SINGLE FACE
        # =================================================

        if len(faces) != 1:
            return False, 0.0, "Multiple/No faces"

        face = faces[0]

        # =================================================
        # FACE SIZE CHECK
        # =================================================

        x1, y1, x2, y2 = face.bbox

        face_area = (x2 - x1) * (y2 - y1)

        img_area = image.shape[0] * image.shape[1]

        ratio = face_area / img_area

        if ratio < 0.12:
            return False, 0.0, "Face too small"

        embedding = face.embedding

        embedding = (
            embedding /
            np.linalg.norm(embedding)
        )

        ref_embedding = REFERENCE_EMBEDDINGS[
            politician
        ]

        similarity = np.dot(
            embedding,
            ref_embedding
        )

        if similarity < SIMILARITY_THRESHOLD:

            return (
                False,
                similarity,
                "Low similarity"
            )

        return (
            True,
            similarity,
            "Accepted"
        )

    except Exception as e:

        return (
            False,
            0.0,
            str(e)
        )


# =========================================================
# DOWNLOAD + VERIFY
# =========================================================

def download_and_verify_image(
    img_url,
    save_path,
    seen_hashes,
    politician
):

    try:

        response = requests.get(
            img_url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if response.status_code != 200:
            return False, "HTTP Error"

        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        if not content_type.startswith("image/"):
            return False, "Not image"

        data = response.content

        # =================================================
        # FILE SIZE
        # =================================================

        if len(data) < MIN_FILE_SIZE:
            return False, "Small file"

        # =================================================
        # IMAGE SIZE
        # =================================================

        try:

            img = Image.open(
                BytesIO(data)
            )

            w, h = img.size

            if (
                w < MIN_WIDTH
                or
                h < MIN_HEIGHT
            ):
                return False, "Low resolution"

        except:
            return False, "Invalid image"

        # =================================================
        # TEXT FILTER
        # =================================================

        if has_too_much_text(data):
            return False, "Text overlay"

        # =================================================
        # DUPLICATE FILTER
        # =================================================

        if is_duplicate(
            data,
            seen_hashes
        ):
            return False, "Duplicate"

        # =================================================
        # FACE VERIFICATION
        # =================================================

        accepted, similarity, reason = verify_face(
            data,
            politician
        )

        if not accepted:

            return (
                False,
                f"{reason} | sim={similarity:.3f}"
            )

        # =================================================
        # SAVE IMAGE
        # =================================================

        with open(save_path, "wb") as f:
            f.write(data)

        return (
            True,
            f"sim={similarity:.3f}"
        )

    except Exception as e:

        return False, str(e)


# =========================================================
# URL EXTRACTION
# =========================================================

def extract_image_urls(html):

    urls = re.findall(
        r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp))"',
        html,
        re.IGNORECASE
    )

    seen = set()

    final = []

    BAD_WORDS = [
        "gstatic",
        "google",
        "logo",
        "icon",
        "sprite",
        "emoji",
        "sticker",
        "banner",
        "thumbnail",
        "avatar",
    ]

    for url in urls:

        url = (
            url
            .replace("\\u003d", "=")
            .replace("\\u0026", "&")
        )

        if any(
            b in url.lower()
            for b in BAD_WORDS
        ):
            continue

        if url not in seen:

            seen.add(url)

            final.append(url)

    return final


# =========================================================
# SCRAPE PERSON
# =========================================================

async def scrape_person(
    context,
    semaphore,
    person_key,
    query_list
):

    async with semaphore:

        page = await context.new_page()

        print("\n" + "=" * 80)
        print(f"STARTING: {person_key}")
        print("=" * 80)

        seen_hashes = []

        accepted_count = len([
            f for f in os.listdir(FINAL_DATASET_DIR)
            if f.startswith(person_key)
        ])

        rejected_count = 0

        while accepted_count < MAX_IMAGES_PER_PERSON:

            before = accepted_count

            for query in query_list:

                if accepted_count >= MAX_IMAGES_PER_PERSON:
                    break

                try:

                    query_encoded = query.replace(
                        " ",
                        "+"
                    )

                    url = (
                        f"https://www.google.com/search?"
                        f"tbm=isch&q={query_encoded}"
                        f"&tbs=itp:face"
                    )

                    print(f"\nQuery: {query}")

                    await page.goto(
                        url,
                        wait_until="domcontentloaded"
                    )

                    await page.wait_for_timeout(3000)

                    for _ in range(12):

                        await page.mouse.wheel(
                            0,
                            7000
                        )

                        await page.wait_for_timeout(
                            random.randint(1200, 2400)
                        )

                    html = await page.content()

                    urls = extract_image_urls(html)

                    print(f"Found URLs: {len(urls)}")

                    for img_url in urls:

                        if accepted_count >= MAX_IMAGES_PER_PERSON:
                            break

                        filename = (
                            f"{person_key}_"
                            f"{accepted_count+1:04d}.jpg"
                        )

                        save_path = (
                            FINAL_DATASET_DIR /
                            filename
                        )

                        success, reason = (
                            download_and_verify_image(
                                img_url,
                                save_path,
                                seen_hashes,
                                person_key
                            )
                        )

                        # =========================
                        # ACCEPTED
                        # =========================

                        if success:

                            accepted_count += 1

                            print(
                                f"[ACCEPTED] "
                                f"{filename} "
                                f"| {reason} "
                                f"| total={accepted_count}"
                            )

                        # =========================
                        # REJECTED
                        # =========================

                        else:

                            rejected_count += 1

                            print(
                                f"[REJECTED] "
                                f"{reason}"
                            )

                    print(
                        f"\nCurrent Count:"
                        f" {accepted_count}/"
                        f"{MAX_IMAGES_PER_PERSON}"
                    )

                    await asyncio.sleep(
                        random.randint(3, 7)
                    )

                except Exception as e:

                    print(f"Query Error: {e}")

            if accepted_count == before:

                print(
                    "\nNo new valid images found."
                )

                break

        print("\n" + "=" * 80)
        print(f"COMPLETED: {person_key}")
        print(f"Accepted: {accepted_count}")
        print(f"Rejected: {rejected_count}")
        print("=" * 80)

        await page.close()


# =========================================================
# MAIN
# =========================================================

async def main():

    print(
        "\nStarting Advanced Verified Dataset Scraper...\n"
    )

    semaphore = asyncio.Semaphore(
        MAX_CONCURRENT_POLITICIANS
    )

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        context = await browser.new_context(
            viewport={
                "width": 1400,
                "height": 900
            },
            locale="en-US",
        )

        await context.add_init_script(
            """
            Object.defineProperty(
                navigator,
                'webdriver',
                {get: () => undefined}
            )
            """
        )

        tasks = []

        for person_key, query_list in POLITICIANS.items():

            task = scrape_person(
                context,
                semaphore,
                person_key,
                query_list
            )

            tasks.append(task)

        await asyncio.gather(*tasks)

        await browser.close()

    print("\nALL SCRAPING COMPLETED.\n")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    asyncio.run(main())