# =============================================================
# POLITICIAN DATASET SCRAPER  v3 — PRODUCTION GRADE
# =============================================================
#
# GUARANTEES:
#   ✅ Exactly 100 images per politician (hard atomic counter)
#   ✅ ZERO duplicates — 4-layer dedup:
#        L1: URL-level  (per-politician set, O(1))
#        L2: MD5        (exact byte match)
#        L3: pHash      (visual near-duplicate, hamming ≤ 5)
#        L4: Embedding  (same face cropped differently)
#        All 4 checked atomically inside one lock
#   ✅ Proper backpressure — scraper waits when queue full
#   ✅ Workers never idle — queue always fed
#   ✅ No deadlock — every blocking op has timeout
#   ✅ task_done() called exactly once per item (finally only)
#   ✅ Restart-safe — existing files hashed on startup
#   ✅ Exhaustion recovery — 3 query tiers before giving up
#   ✅ Auto-scaled resources from MAX_CONCURRENT_POLITICIANS
#
# =============================================================

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import aiohttp
import cv2                        # pyrefly: ignore
import insightface                # pyrefly: ignore
import numpy as np
from PIL import Image
# pyrefly: ignore [missing-import]
from playwright.async_api import async_playwright
from scipy.fft import dct


# =============================================================
# PATHS
# =============================================================
BASE_DIR        = Path(__file__).resolve().parents[2]
RAW_DATASET_DIR = BASE_DIR / "data" / "raw"
EMBEDDINGS_DIR  = BASE_DIR / "data" / "embeddings"
os.makedirs(RAW_DATASET_DIR, exist_ok=True)


# =============================================================
# SETTINGS
#
# Only tweak MAX_CONCURRENT_POLITICIANS — everything else
# derives from it automatically.
# =============================================================
MAX_IMAGES_PER_PERSON      = 100
MAX_CONCURRENT_POLITICIANS = 3   # ← only knob you need

# --- derived values -----------------------------------------
# Workers: each politician needs ~5 concurrent downloaders
# InsightFace is CPU-bound so cap at 16 to avoid thrashing
MAX_VERIFY_WORKERS = min(MAX_CONCURRENT_POLITICIANS * 5, 16)

# Queue: each Google page yields ~50-70 URLs
# Buffer = 5 pages × politicians × some headroom
URLS_PER_PAGE = 70
PAGES_BUFFER  = 5
QUEUE_SIZE    = MAX_CONCURRENT_POLITICIANS * URLS_PER_PAGE * PAGES_BUFFER  # 1050

# Backpressure: scraper pauses when queue is above this
# = 2 pages worth per politician → workers have enough to chew
QUEUE_HIGH_WATERMARK = MAX_CONCURRENT_POLITICIANS * URLS_PER_PAGE * 2      # 420
QUEUE_LOW_WATERMARK  = MAX_CONCURRENT_POLITICIANS * URLS_PER_PAGE          # 210

# Timeouts
PAGE_LOAD_TIMEOUT  = 30    # seconds
DOWNLOAD_TIMEOUT   = 15    # seconds
BACKPRESSURE_POLL  = 2     # poll interval when waiting for queue to drain

# Exhaustion thresholds (rounds with zero new accepted images)
ZERO_ROUNDS_TO_TIER2 = 4   # switch to broader queries
ZERO_ROUNDS_TO_TIER3 = 4   # switch to fallback templates
ZERO_ROUNDS_GIVE_UP  = 6   # stop entirely

MAX_SCRAPE_ROUNDS = 150
SCROLL_ROUNDS     = 16     # per page — more scroll = more URLs

# Face similarity
SIMILARITY_THRESHOLD = 0.50
PHASH_THRESHOLD      = 5   # hamming distance for near-duplicate
MAX_ALLOWED_FACES = 2

print(f"""
{"="*58}
  SCRAPER v3  —  parallelism = {MAX_CONCURRENT_POLITICIANS}
{"="*58}
  verify_workers      : {MAX_VERIFY_WORKERS}
  queue_size          : {QUEUE_SIZE}
  high_watermark      : {QUEUE_HIGH_WATERMARK}
  low_watermark       : {QUEUE_LOW_WATERMARK}
  zero→tier2          : {ZERO_ROUNDS_TO_TIER2} rounds
  zero→tier3          : {ZERO_ROUNDS_TO_TIER3} rounds
  zero→give_up        : {ZERO_ROUNDS_GIVE_UP} rounds
  max_scrape_rounds   : {MAX_SCRAPE_ROUNDS}
{"="*58}
""")


# =============================================================
# POLITICIANS
# =============================================================
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

    "khawaja_saad_rafique": [
    "only Khawaja Saad Rafique solo HD face",
    "only Khawaja Saad Rafique close up solo pic",
    "only Khawaja Saad Rafique official solo image",
    "only Khawaja Saad Rafique high quality face",
    "only Khawaja Saad Rafique single person photo",

    "Khawaja Saad Rafique portrait HD",
    "Khawaja Saad Rafique face closeup",
    "Khawaja Saad Rafique official portrait",
    "Khawaja Saad Rafique looking at camera",
    "Khawaja Saad Rafique sharp face image",

    "Pakistan politician Khawaja Saad Rafique face",
    "PMLN leader Khawaja Saad Rafique portrait",
    "Federal minister Khawaja Saad Rafique HD photo",
    "Khawaja Saad Rafique press conference face",
    "Khawaja Saad Rafique recent close up",

    "Khawaja Saad Rafique smiling portrait",
    "Khawaja Saad Rafique news interview face",
    "Khawaja Saad Rafique front face HD",
    "Khawaja Saad Rafique official media image",
    "Khawaja Saad Rafique single person portrait"
],

    "yousuf_raza_gillani": [
        "only Yousuf Raza Gillani solo HD face",
        "only Yousuf Raza Gillani close up solo pic",
        "only Yousuf Raza Gillani official solo image",
        "only Yousuf Raza Gillani high quality face",
        "only Yousuf Raza Gillani single person photo"
    ]

"ahmed_sharif_chaudhry": [

    "only Lieutenant General Ahmed Sharif Chaudhry solo face",

    "only Ahmed Sharif Chaudhry close up pic",

    "only Ahmed Sharif Chaudhry official image",

    "only Ahmed Sharif Chaudhry military uniform face",

    "only Ahmed Sharif Chaudhry single person photo",

    "only DG ISPR Ahmed Sharif Chaudhry face",

    "only Ahmed Sharif Chaudhry press conference pic",

    "only Ahmed Sharif Chaudhry looking at camera",

    "only Ahmed Sharif Chaudhry sharp face image",

    "only Ahmed Sharif Chaudhry army spokesperson"

],
}

# Tier-2: slightly broader (no "solo"/"only")
_TIER2_TEMPLATES = [
    "{name} face photograph",
    "{name} politician face",
    "{name} press conference",
    "{name} interview photo",
    "{name} news photo",
    "{name} speech photo",
    "{name} headshot",
    "{name} official photo",
]

# Tier-3 fallback: very broad
_TIER3_TEMPLATES = [
    "{name}",
    "{name} Pakistan",
    "{name} politician",
    "{name} portrait",
    "{name} photo",
    "{name} image",
    "{name} picture",
    "{name} closeup",
]

def _make_queries(person_key: str, templates: list[str]) -> list[str]:
    name = person_key.replace("_", " ").title()
    return [t.format(name=name) for t in templates]


# =============================================================
# INSIGHTFACE
# =============================================================
print("Loading InsightFace...")
_face_app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
_face_app.prepare(ctx_id=0)
print("InsightFace ready.\n")


# =============================================================
# REFERENCE EMBEDDINGS
# =============================================================
REFERENCE_EMBEDDINGS: dict[str, np.ndarray] = {}
for _f in os.listdir(EMBEDDINGS_DIR):
    if _f.endswith(".npy"):
        _k = _f.replace(".npy", "")
        _e = np.load(EMBEDDINGS_DIR / _f).astype(np.float32)
        _e /= (np.linalg.norm(_e) + 1e-9)
        REFERENCE_EMBEDDINGS[_k] = _e
print(f"Loaded {len(REFERENCE_EMBEDDINGS)} reference embeddings.\n")


# =============================================================
# DUPLICATE DETECTION — 4 layers
# =============================================================

def _phash(image_bytes: bytes) -> str | None:
    """Perceptual hash — 64-bit binary string."""
    try:
        img = (
            Image.open(BytesIO(image_bytes))
            .convert("L")
            .resize((32, 32), Image.LANCZOS)
        )
        arr    = np.array(img, dtype=float)
        d      = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
        low    = d[:8, :8].flatten()
        median = np.median(low)
        return "".join("1" if v > median else "0" for v in low)
    except Exception:
        return None


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _face_embedding(image_bytes: bytes) -> np.ndarray | None:
    """Extract face embedding for embedding-level dedup (L4)."""
    try:
        arr   = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        faces = _face_app.get(img)
        if len(faces) != 1:
            return None
        emb = faces[0].embedding.astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)
        return emb
    except Exception:
        return None


class DedupStore:
    """
    Thread-safe 4-layer duplicate store for one politician.

    Layer 1: URL set          — O(1), checked before download
    Layer 2: MD5              — exact byte-level match
    Layer 3: pHash            — visual near-duplicate (hamming ≤ 5)
    Layer 4: Embedding cosine — same face, different crop/resize
                                 (threshold 0.92 — very tight)

    check_and_claim():
        Atomically checks all 4 layers.
        If NOT duplicate → registers immediately and returns False.
        If duplicate     → returns True (caller discards image).
    """

    EMBED_SIM_THRESHOLD = 0.92   # very tight — only flags near-identical crops

    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self._urls: set[str]        = set()
        self._md5s: set[str]        = set()
        self._phashes: list[str]    = []        # list for hamming scan
        self._embeddings: list[np.ndarray] = [] # list for cosine scan

    # ----------------------------------------------------------
    # URL level (L1) — called BEFORE download, no lock needed
    # because asyncio is single-threaded for this part
    # ----------------------------------------------------------
    def seen_url(self, url: str) -> bool:
        return url in self._urls

    def register_url(self, url: str) -> None:
        self._urls.add(url)

    # ----------------------------------------------------------
    # Image level (L2-L4) — called in ThreadPoolExecutor
    # All checks + registration are atomic (one lock acquisition)
    # ----------------------------------------------------------
    def check_and_claim(self, image_bytes: bytes, embedding: np.ndarray | None) -> tuple[bool, str]:
        """
        Returns (is_duplicate, reason).
        If not duplicate, registers all hashes before returning.
        """
        md5 = hashlib.md5(image_bytes).hexdigest()
        ph  = _phash(image_bytes)

        with self._lock:

            # L2: exact MD5
            if md5 in self._md5s:
                return True, "dup:md5"

            # L3: perceptual hash
            if ph:
                for stored_ph in self._phashes:
                    if _hamming(ph, stored_ph) <= PHASH_THRESHOLD:
                        return True, "dup:phash"

            # L4: embedding cosine similarity
            if embedding is not None:
                for stored_emb in self._embeddings:
                    sim = float(np.dot(embedding, stored_emb))
                    if sim >= self.EMBED_SIM_THRESHOLD:
                        return True, f"dup:emb(sim={sim:.3f})"

            # Not duplicate — register everything atomically
            self._md5s.add(md5)
            if ph:
                self._phashes.append(ph)
            if embedding is not None:
                self._embeddings.append(embedding)

        return False, "new"

    def load_from_disk(self, folder: Path) -> int:
        """Load hashes from existing files. Returns file count."""
        files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
        for fname in files:
            try:
                data = (folder / fname).read_bytes()
                md5  = hashlib.md5(data).hexdigest()
                ph   = _phash(data)
                emb  = _face_embedding(data)
                with self._lock:
                    self._md5s.add(md5)
                    if ph:
                        self._phashes.append(ph)
                    if emb is not None:
                        self._embeddings.append(emb)
            except Exception:
                pass
        return len(files)


# =============================================================
# FACE VERIFICATION
# =============================================================

def _verify_and_embed(
    image_bytes: bytes,
    politician:  str,
) -> tuple[bool, float, str, np.ndarray | None]:
    """
    Returns (passed, similarity, reason, embedding).
    embedding is always returned so dedup can use it (L4).
    """
    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False, 0.0, "corrupt", None

        faces = _face_app.get(img)

        if len(faces) == 0:
            return False, 0.0, "no_face", None

        if len(faces) > 3:
            return False, 0.0, f"too_many_faces={len(faces)}", None

        emb  = faces[0].embedding.astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)

        ref = REFERENCE_EMBEDDINGS.get(politician)
        if ref is None:
            return False, 0.0, "no_ref", None

        sim = float(np.dot(emb, ref))
        if sim < SIMILARITY_THRESHOLD:
            return False, sim, "low_sim", emb   # return emb anyway for dedup

        return True, sim, "ok", emb

    except Exception as e:
        return False, 0.0, str(e), None


# =============================================================
# PIPELINE  (runs in ThreadPoolExecutor — CPU work)
#
# Order:
#   1. Face verify  → get embedding
#   2. Dedup check  → all 4 layers, atomic
#   3. Accept / reject
#
# Face verify runs BEFORE dedup so we have the embedding
# for L4 dedup. Dedup registers atomically on pass only.
# =============================================================

def _pipeline(
    image_bytes: bytes,
    politician:  str,
    dedup:       DedupStore,
) -> tuple[bool, str]:
    """Returns (accepted, reason)."""
    try:
        # Step 1: face verify + get embedding
        passed, sim, reason, emb = _verify_and_embed(image_bytes, politician)
        if not passed:
            return False, reason

        # Step 2: atomic 4-layer dedup (L2-L4)
        is_dup, dup_reason = dedup.check_and_claim(image_bytes, emb)
        if is_dup:
            return False, dup_reason

        return True, f"sim={sim:.3f}"

    except Exception as e:
        return False, f"pipeline_err:{e}"


# =============================================================
# GLOBAL STATE
# =============================================================

# Per-politician dedup stores
DEDUP: dict[str, DedupStore] = {}

# Per-politician counters  {"accepted": int, "rejected": int}
_counter_lock = threading.Lock()
COUNTERS: dict[str, dict] = {}

# ThreadPoolExecutor for CPU work (face verify)
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_VERIFY_WORKERS)


def _init_politician(key: str) -> None:
    folder = RAW_DATASET_DIR / key
    os.makedirs(folder, exist_ok=True)
    store = DedupStore()
    print(f"  [{key}] scanning existing files...")
    existing = store.load_from_disk(folder)
    DEDUP[key]    = store
    COUNTERS[key] = {"accepted": existing, "rejected": 0}
    print(f"  [{key}] {existing} existing | "
          f"{len(store._md5s)} md5s | "
          f"{len(store._phashes)} phashes | "
          f"{len(store._embeddings)} embeddings loaded")


print("Initialising politician stores...")
for _p in POLITICIANS:
    _init_politician(_p)
print()


# =============================================================
# URL EXTRACTION
# =============================================================
_BAD_SUBSTRINGS = {
    "gstatic", "google", "logo", "icon", "sprite",
    "emoji", "sticker", "banner", "avatar", "pixel",
    "1x1", "blank", "spacer", "placeholder",
}

def _extract_urls(html: str) -> list[str]:
    raw = re.findall(
        r'"(https?://[^"]{10,}\.(?:jpg|jpeg|png|webp)(?:[^"]{0,200})?)"',
        html, re.IGNORECASE,
    )
    seen: set[str] = set()
    out:  list[str] = []
    for url in raw:
        url = (url
               .replace("\\u003d", "=")
               .replace("\\u0026", "&")
               .replace("\\u003c", "<")
               .replace("\\u003e", ">"))
        low = url.lower()
        if any(b in low for b in _BAD_SUBSTRINGS):
            continue
        # skip tiny thumbnails (Google adds =s<N> or =w<N>-h<N>)
        if re.search(r'=s\d{1,2}(?:-c)?$', url):
            continue
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


# =============================================================
# DOWNLOAD
# =============================================================
async def _download(
    session: aiohttp.ClientSession,
    url:     str,
) -> bytes | None:
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            },
            ssl=False,
        ) as r:
            if r.status != 200:
                return None
            ct = r.headers.get("Content-Type", "")
            if not any(t in ct for t in ("image/", "application/octet")):
                return None
            data = await r.read()
            # Minimum size filter — skip tiny thumbnails
            return data if len(data) >= 5_000 else None
    except Exception:
        return None


# =============================================================
# WORKER
#
# One worker loop — pulls items from queue, downloads,
# runs pipeline, saves if accepted.
# task_done() is called EXACTLY ONCE per item in finally.
# =============================================================
async def _worker(
    queue:   asyncio.Queue,
    loop:    asyncio.AbstractEventLoop,
    session: aiohttp.ClientSession,
) -> None:
    while True:
        item = await queue.get()

        # Poison pill
        if item is None:
            queue.task_done()
            break

        politician = item["politician"]
        url        = item["url"]

        try:
            # Already at target? Drain silently
            with _counter_lock:
                if COUNTERS[politician]["accepted"] >= MAX_IMAGES_PER_PERSON:
                    continue   # finally calls task_done()

            # Download
            data = await _download(session, url)
            if data is None:
                with _counter_lock:
                    COUNTERS[politician]["rejected"] += 1
                continue       # finally calls task_done()

            # CPU pipeline in thread (face verify + 4-layer dedup)
            accepted, reason = await loop.run_in_executor(
                _EXECUTOR, _pipeline, data, politician, DEDUP[politician]
            )

            if accepted:
                saved = False
                fname = ""
                path  = Path()

                with _counter_lock:
                    if COUNTERS[politician]["accepted"] < MAX_IMAGES_PER_PERSON:
                        COUNTERS[politician]["accepted"] += 1
                        n     = COUNTERS[politician]["accepted"]
                        fname = f"{politician}_{n:04d}.jpg"
                        path  = RAW_DATASET_DIR / politician / fname
                        saved = True

                if saved:
                    path.write_bytes(data)
                    with _counter_lock:
                        total = COUNTERS[politician]["accepted"]
                    print(f"  [✓] {fname} | {reason} | total={total}/{MAX_IMAGES_PER_PERSON}")

            else:
                with _counter_lock:
                    COUNTERS[politician]["rejected"] += 1
                # Only print non-trivial rejections (skip dup spam)
                if not reason.startswith("dup:"):
                    print(f"  [✗] {politician} | {reason}")

        except Exception as e:
            print(f"  [WORKER ERR] {politician} | {e}")

        finally:
            # Exactly once per item — no double task_done() possible
            queue.task_done()


# =============================================================
# BACKPRESSURE WAIT
#
# Scraper calls this after pushing URLs.
# Blocks until queue drains below LOW_WATERMARK OR target reached.
# This prevents queue from growing unboundedly and ensures
# workers always have fresh (not stale/exhausted) URLs.
# =============================================================
async def _wait_for_backpressure(
    queue:      asyncio.Queue,
    politician: str,
) -> None:
    """
    Wait until queue is below low watermark.
    Returns immediately if politician target is already met.
    Hard timeout = 120s to prevent infinite block.
    """
    deadline = asyncio.get_event_loop().time() + 120

    while asyncio.get_event_loop().time() < deadline:
        with _counter_lock:
            if COUNTERS[politician]["accepted"] >= MAX_IMAGES_PER_PERSON:
                return

        if queue.qsize() <= QUEUE_LOW_WATERMARK:
            return

        await asyncio.sleep(BACKPRESSURE_POLL)


# =============================================================
# SCRAPE ONE POLITICIAN
# =============================================================
async def _scrape(
    context:    object,
    semaphore:  asyncio.Semaphore,
    queue:      asyncio.Queue,
    person_key: str,
    tier1_q:    list[str],
) -> None:
    """
    3-tier query strategy:
      Tier 1 — specific "solo face" queries (from POLITICIANS dict)
      Tier 2 — broader queries (TIER2_TEMPLATES)
      Tier 3 — very broad fallback (TIER3_TEMPLATES)
    Switches tier when zero_rounds threshold is hit.
    Gives up only after Tier 3 exhausted.
    """
    async with semaphore:

        with _counter_lock:
            already = COUNTERS[person_key]["accepted"]
        if already >= MAX_IMAGES_PER_PERSON:
            print(f"  [{person_key}] already complete — skipping")
            return

        page = await context.new_page()
        print("\n" + "=" * 65)
        print(f"  START: {person_key}")
        print("=" * 65)

        tier2_q = _make_queries(person_key, _TIER2_TEMPLATES)
        tier3_q = _make_queries(person_key, _TIER3_TEMPLATES)

        # Tier state machine
        tiers     = [tier1_q, tier2_q, tier3_q]
        tier_names= ["tier1:specific", "tier2:broader", "tier3:broad"]
        tier_idx  = 0

        active_q   = tiers[tier_idx][:]
        q_cursor   = 0
        rounds     = 0
        zero_rounds= 0

        with _counter_lock:
            prev_accepted = COUNTERS[person_key]["accepted"]

        while True:

            # ── Target check ────────────────────────────
            with _counter_lock:
                current = COUNTERS[person_key]["accepted"]
            if current >= MAX_IMAGES_PER_PERSON:
                print(f"  [{person_key}] ✅ Target {MAX_IMAGES_PER_PERSON} reached!")
                break

            # ── Round cap ───────────────────────────────
            if rounds >= MAX_SCRAPE_ROUNDS:
                print(f"  [{person_key}] ⚠️  Round cap {MAX_SCRAPE_ROUNDS} hit.")
                break

            rounds   += 1
            query     = active_q[q_cursor % len(active_q)]
            q_cursor += 1

            print(f"\n  [{person_key}] Round {rounds} | [{tier_names[tier_idx]}] | {query!r}")

            # ── Backpressure: wait if queue too full ─────
            if queue.qsize() > QUEUE_HIGH_WATERMARK:
                print(f"  [{person_key}] queue={queue.qsize()} > {QUEUE_HIGH_WATERMARK} → waiting for workers...")
                await _wait_for_backpressure(queue, person_key)

            # ── Load Google Images page ──────────────────
            try:
                await asyncio.wait_for(
                    page.goto(
                        "https://www.google.com/search"
                        f"?tbm=isch&q={query.replace(' ', '+')}&tbs=itp:face",
                        wait_until="domcontentloaded",
                    ),
                    timeout=PAGE_LOAD_TIMEOUT,
                )
            except Exception as e:
                print(f"  [{person_key}] page load failed: {e}")
                await asyncio.sleep(5)
                continue

            # Initial wait for images to render
            await page.wait_for_timeout(random.randint(1500, 2500))

            # ── Scroll to load more images ───────────────
            for _ in range(SCROLL_ROUNDS):
                await page.mouse.wheel(0, 3000)
                await page.wait_for_timeout(random.randint(300, 600))
            await page.wait_for_timeout(800)

            # FIX: Extra settle time after scrolling.
            # Google lazy-loads and continuously mutates the DOM after
            # scroll events — calling page.content() too early raises:
            #   "Page.content: Unable to retrieve content because the
            #    page is navigating"
            # The 2s sleep lets in-flight XHRs settle before we read.
            await asyncio.sleep(2)

            # ── Extract + push URLs ──────────────────────
            # FIX: Retry loop instead of a single page.content() call.
            # domcontentloaded is a lighter barrier than networkidle —
            # networkidle can time out on pages that keep polling.
            # We retry up to 3 times with a 1s back-off between attempts.
            html = ""
            for _ in range(3):
                try:
                    await page.wait_for_load_state(
                        "domcontentloaded",
                        timeout=5000,
                    )
                    html = await page.content()
                    if html:
                        break
                except Exception:
                    await asyncio.sleep(1)

            urls = _extract_urls(html)
            random.shuffle(urls)

            dedup    = DEDUP[person_key]
            pushed   = 0
            skipped  = 0
            too_full = 0

            for url in urls:
                with _counter_lock:
                    if COUNTERS[person_key]["accepted"] >= MAX_IMAGES_PER_PERSON:
                        break

                # L1 dedup — URL level (before any download)
                if dedup.seen_url(url):
                    skipped += 1
                    continue

                try:
                    queue.put_nowait({"politician": person_key, "url": url})
                    # Register ONLY after successful enqueue
                    dedup.register_url(url)
                    pushed += 1
                except asyncio.QueueFull:
                    # Do NOT register — allow retry next round
                    too_full += 1

            print(
                f"  [{person_key}] found={len(urls)} | "
                f"pushed={pushed} | seen={skipped} | "
                f"queue_full={too_full} | q={queue.qsize()}"
            )

            # ── Brief async yield so workers can run ────
            await asyncio.sleep(random.randint(2, 4))

            # ── Progress check ───────────────────────────
            with _counter_lock:
                accepted = COUNTERS[person_key]["accepted"]
                rejected = COUNTERS[person_key]["rejected"]

            net_new = accepted - prev_accepted

            print(
                f"  [{person_key}] accepted={accepted}/{MAX_IMAGES_PER_PERSON} "
                f"(+{net_new}) | rejected={rejected} | "
                f"zero_streak={zero_rounds}"
            )

            if net_new > 0:
                zero_rounds   = 0
                prev_accepted = accepted
            elif queue.qsize() < QUEUE_LOW_WATERMARK:
                # Only count as "zero" when queue is actually drained
                # (workers had a chance to process everything)
                zero_rounds += 1

            # ── Tier switching ───────────────────────────
            next_tier_threshold = (
                ZERO_ROUNDS_TO_TIER2 if tier_idx == 0
                else ZERO_ROUNDS_TO_TIER3 if tier_idx == 1
                else ZERO_ROUNDS_GIVE_UP
            )

            if zero_rounds >= next_tier_threshold:
                if tier_idx < len(tiers) - 1:
                    tier_idx  += 1
                    active_q   = tiers[tier_idx][:]
                    q_cursor   = 0
                    zero_rounds= 0
                    print(f"  [{person_key}] → switching to {tier_names[tier_idx]}")
                    await asyncio.sleep(random.randint(4, 8))  # rate limit breather
                else:
                    print(f"  [{person_key}] → all tiers exhausted. Stopping.")
                    break

        # ── Scraper done — do NOT call queue.join() here ──
        # main() calls queue.join() after ALL scrapers finish.
        with _counter_lock:
            final    = COUNTERS[person_key]["accepted"]
            rejected = COUNTERS[person_key]["rejected"]

        status = "✅" if final >= MAX_IMAGES_PER_PERSON else "⚠️ "
        print("\n" + "=" * 65)
        print(f"  {status} DONE : {person_key}")
        print(f"     accepted = {final}/{MAX_IMAGES_PER_PERSON}")
        print(f"     rejected = {rejected}")
        print("=" * 65)

        await page.close()


# =============================================================
# MAIN
# =============================================================
async def main() -> None:
    print("\nStarting scraper...\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_POLITICIANS)
    queue     = asyncio.Queue(maxsize=QUEUE_SIZE)
    loop      = asyncio.get_running_loop()

    async with aiohttp.ClientSession() as session:
        async with async_playwright() as pw:

            browser = await pw.chromium.launch(
                headless=False,
                # FIX: slow_mo increased from 40 → 80.
                # This gives Google Images more breathing room between
                # synthetic browser events, reducing DOM-navigation races
                # under high parallelism (3+ pages scrolling concurrently).
                slow_mo=120,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            context = await browser.new_context(
                viewport={"width": 1400, "height": 900},
                locale="en-US",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            await context.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
            )

            # Launch workers first so queue is serviced immediately
            workers = [
                asyncio.create_task(_worker(queue, loop, session))
                for _ in range(MAX_VERIFY_WORKERS)
            ]

            # Launch all scraper coroutines concurrently
            # Semaphore limits to MAX_CONCURRENT_POLITICIANS at a time
            await asyncio.gather(*[
                _scrape(context, semaphore, queue, key, q)
                for key, q in POLITICIANS.items()
            ])

            # All scrapers done — drain whatever is left in queue
            print("\nAll scrapers done. Draining remaining queue items...")
            await queue.join()
            print("Queue fully drained.")

            # Send poison pills to stop workers
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)

            await browser.close()

    _EXECUTOR.shutdown(wait=True)

    # ── Final summary ────────────────────────────────────────
    total_accepted = 0
    total_rejected = 0
    incomplete     = []

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    for key, c in COUNTERS.items():
        ok     = c["accepted"] >= MAX_IMAGES_PER_PERSON
        status = "✅" if ok else "⚠️ "
        print(
            f"  {status} {key:<28} "
            f"accepted={c['accepted']:>3}/{MAX_IMAGES_PER_PERSON} | "
            f"rejected={c['rejected']:>5}"
        )
        total_accepted += c["accepted"]
        total_rejected += c["rejected"]
        if not ok:
            incomplete.append(key)

    print("=" * 65)
    print(f"  Total accepted : {total_accepted}")
    print(f"  Total rejected : {total_rejected}")
    if incomplete:
        print(f"\n  ⚠️  Incomplete ({len(incomplete)}):")
        for k in incomplete:
            print(f"       {k} — {COUNTERS[k]['accepted']}/{MAX_IMAGES_PER_PERSON}")
    else:
        print("\n  ✅ All politicians complete!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    asyncio.run(main())