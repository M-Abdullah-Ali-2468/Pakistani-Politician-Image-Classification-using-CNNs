import os
import re
import time
import json
import random
import asyncio
import hashlib
import requests

from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np
from scipy.fft import dct  # for perceptual hashing
from tqdm import tqdm
from playwright.async_api import async_playwright

# =========================================================
# Pakistani Politician Google Images Scraper (Fixed)
# =========================================================

POLITICIANS = {
    "imran_khan":            "Imran Khan Pakistani politician",
    "nawaz_sharif":          "Nawaz Sharif Pakistani politician",
    "shahbaz_sharif":        "Shahbaz Sharif Prime Minister Pakistan",
    "maryam_nawaz":          "Maryam Nawaz Punjab Chief Minister",
    "bilawal_bhutto":        "Bilawal Bhutto Zardari politician",
    "asif_zardari":          "Asif Ali Zardari President Pakistan",
    "fazlur_rehman":         "Maulana Fazlur Rehman politician",
    "khawaja_asif":          "Khawaja Asif politician Pakistan",
    "sheikh_rasheed":        "Sheikh Rasheed Ahmad politician Pakistan",
    "murad_ali_shah":        "Murad Ali Shah Sindh Chief Minister",
    "pervez_khattak":        "Pervez Khattak politician KPK",
    "siraj_ul_haq":          "Siraj ul Haq JI politician Pakistan",
    "hina_rabbani":          "Hina Rabbani Khar politician Pakistan",
    "sherry_rehman":         "Sherry Rehman politician Pakistan",
    "aitzaz_ahsan":          "Aitzaz Ahsan politician Pakistan",
    "ahmed_sharif_chaudhry": "Ahmed Sharif Chaudhry DG ISPR",
}

MAX_IMAGES_PER_PERSON = 100

# =========================================================
# HD QUALITY SETTINGS
# Min resolution to accept — images below this are rejected
# =========================================================
MIN_WIDTH  = 400   # pixels
MIN_HEIGHT = 400   # pixels
MIN_FILE_SIZE = 30_000   # bytes (~30 KB) — filters out tiny/blurry images

# =========================================================
# TEXT DETECTION SETTINGS (pixel-based — no OCR needed)
# =========================================================

# How much of a strip (top/bottom/sides) to scan — 0.15 = 15% of image height
TEXT_STRIP_RATIO   = 0.15

# Edge density threshold — strips above this are likely text bars
# Range 0.0–1.0. Lower = stricter. 0.08 works well for news tickers.
EDGE_DENSITY_THRESH = 0.08

# Horizontal variance threshold — text rows have alternating dark/light pixels
# i.e. high variance along horizontal axis
HORIZ_VAR_THRESH   = 18.0

# What fraction of strip rows must be "text-like" to reject the whole image
TEXT_ROW_FRACTION  = 0.35   # 35% of rows in the strip look like text → reject


# =========================================================
# TEXT DETECTION — pixel / edge based
# ─────────────────────────────────────────────────────────
# Strategy: text (news tickers, watermarks, Urdu overlays,
# captions) always creates dense horizontal edge patterns
# in the top/bottom/side strips of Pakistani news images.
# We detect those patterns without any OCR.
#
# Three signals combined:
#   1. Edge density  — Sobel edges / total pixels in strip
#   2. Horizontal variance — alternating dark↔light columns
#   3. Uniform-color band — solid-color ticker backgrounds
# =========================================================

def _strip(arr: np.ndarray, ratio: float, side: str) -> np.ndarray:
    """Extract a strip from top, bottom, left, or right of a grayscale array."""
    h, w = arr.shape
    n = max(1, int(h * ratio)) if side in ("top", "bottom") else max(1, int(w * ratio))
    if side == "top":    return arr[:n, :]
    if side == "bottom": return arr[h - n:, :]
    if side == "left":   return arr[:, :n]
    if side == "right":  return arr[:, w - n:]


def _edge_density(strip: np.ndarray) -> float:
    """Fraction of pixels that are strong edges (Sobel magnitude > 30)."""
    from PIL import Image as _Image, ImageFilter as _IF
    pil = _Image.fromarray(strip)
    edges = np.array(pil.filter(_IF.FIND_EDGES))
    return float((edges > 30).sum()) / edges.size


def _horiz_variance(strip: np.ndarray) -> float:
    """Mean variance of pixel values along each row — high in text strips."""
    if strip.shape[0] == 0:
        return 0.0
    return float(np.mean(np.var(strip.astype(float), axis=1)))


def _is_uniform_band(strip: np.ndarray, tol: int = 18) -> bool:
    """True if the strip is a near-solid colour band (common ticker background)."""
    return float(np.std(strip.astype(float))) < tol


def strip_has_text(strip: np.ndarray) -> bool:
    """
    Returns True if the strip looks like it contains overlaid text.
    Combines edge density + horizontal variance + uniform-band check.
    """
    ed  = _edge_density(strip)
    hv  = _horiz_variance(strip)
    uni = _is_uniform_band(strip)

    # Uniform colour band with moderate edges → solid ticker bar (e.g. GEO red bar)
    if uni and ed > 0.03:
        return True

    # High edge density AND high horizontal variance → text characters
    if ed > EDGE_DENSITY_THRESH and hv > HORIZ_VAR_THRESH:
        return True

    return False


def has_too_much_text(image_bytes: bytes) -> bool:
    """
    Analyses the image's border strips (top, bottom, left, right).
    Returns True if any strip is detected as a text/ticker overlay.
    No external OCR engine required — works on any font or script,
    including Urdu, Arabic, and stylised news-channel graphics.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("L")  # greyscale

        # Downscale for speed — analysis doesn't need full resolution
        max_w = 600
        if img.width > max_w:
            scale = max_w / img.width
            img = img.resize((max_w, int(img.height * scale)), Image.LANCZOS)

        arr = np.array(img)

        for side in ("top", "bottom", "left", "right"):
            strip = _strip(arr, TEXT_STRIP_RATIO, side)
            if strip_has_text(strip):
                print(f"  [text-skip] text overlay detected on {side} strip")
                return True

        return False

    except Exception as e:
        print(f"  [text-check error] {e} — allowing image")
        return False


# =========================================================
# DUPLICATE DETECTION — perceptual hash (pHash)
# ─────────────────────────────────────────────────────────
# MD5 on raw bytes catches exact duplicates (same file).
# pHash catches visually identical images even if they have
# different compression, slight crop, or resize applied.
#
# How pHash works:
#   1. Shrink image to 32×32 greyscale
#   2. Apply DCT (via numpy) — like JPEG compression
#   3. Keep top-left 8×8 = 64 low-frequency coefficients
#   4. Hash = 64-bit string: each bit = above/below median
#
# Two images are duplicates if their pHash differs by
# fewer than PHASH_THRESHOLD bits (Hamming distance).
# =========================================================

PHASH_THRESHOLD = 8   # bits difference allowed (0=exact, 10=very similar)


def compute_phash(image_bytes: bytes) -> str | None:
    """Returns a 64-char binary string (pHash) for the image, or None on error."""
    try:
        img = Image.open(BytesIO(image_bytes)).convert("L").resize((32, 32), Image.LANCZOS)
        arr = np.array(img, dtype=float)

        # 2D DCT via row-then-column 1D DCTs
        dct2d = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")

        # Top-left 8×8 block (low frequencies)
        low = dct2d[:8, :8].flatten()
        median = np.median(low)

        return "".join("1" if v > median else "0" for v in low)

    except Exception:
        return None


def hamming(a: str, b: str) -> int:
    """Bit-level Hamming distance between two equal-length binary strings."""
    return sum(x != y for x, y in zip(a, b))


def is_duplicate(image_bytes: bytes, seen_hashes: list[str]) -> bool:
    """
    Returns True if image is too similar to any previously saved image.
    Also appends the new hash to seen_hashes if it passes.
    """
    # ── Layer 1: exact byte hash (fastest) ────────────────
    md5 = hashlib.md5(image_bytes).hexdigest()
    if md5 in seen_hashes:
        print("  [dup-skip] exact duplicate (MD5 match)")
        return True

    # ── Layer 2: perceptual hash (catches resized/recompressed dupes) ──
    ph = compute_phash(image_bytes)
    if ph is not None:
        for existing_ph in seen_hashes:
            if len(existing_ph) == 64:          # pHash entries are 64 chars
                dist = hamming(ph, existing_ph)
                if dist <= PHASH_THRESHOLD:
                    print(f"  [dup-skip] visually similar image (pHash dist={dist})")
                    return True

    # Not a duplicate — register both hashes for future comparisons
    seen_hashes.append(md5)
    if ph is not None:
        seen_hashes.append(ph)

    return False


# =========================================================
# DOWNLOAD IMAGE — with content-type + HD dimension check
# =========================================================

def download_image(img_url: str, save_path: str, seen_hashes: list[str]) -> bool:
    try:
        response = requests.get(
            img_url,
            timeout=15,
            stream=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            },
        )

        if response.status_code != 200:
            return False

        # Must be an image content-type
        ct = response.headers.get("Content-Type", "")
        if not ct.startswith("image/"):
            return False

        data = response.content

        # ── HD CHECK 1: file size ──────────────────────────
        # Small files are almost always thumbnails or error pages
        if len(data) < MIN_FILE_SIZE:
            print(f"  [hd-skip] too small ({len(data)//1024} KB) — {img_url[:60]}")
            return False

        # ── HD CHECK 2: actual pixel dimensions ───────────
        try:
            img = Image.open(BytesIO(data))
            w, h = img.size
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                print(f"  [hd-skip] too small resolution ({w}×{h}) — {img_url[:60]}")
                return False
            # Log resolution for debugging
            print(f"  [hd-ok]  {w}×{h}  {len(data)//1024} KB")
        except Exception as e:
            print(f"  [hd-skip] could not read image ({e})")
            return False

        # ── TEXT CHECK: reject images with overlaid text ──
        if has_too_much_text(data):
            return False

        # ── DUPLICATE CHECK: reject visually identical images ──
        if is_duplicate(data, seen_hashes):
            return False

        with open(save_path, "wb") as f:
            f.write(data)

        return True

    except Exception as e:
        print(f"  [download error] {e}")
        return False


# =========================================================
# EXTRACT IMAGE URLS from Google's JSON payload
# Regex approach — much more reliable than DOM selectors
# =========================================================

def extract_image_urls(html: str) -> list[str]:
    """
    Google embeds full-size image URLs inside AF_initDataCallback JSON blobs.
    We regex-extract every https://... URL that looks like a real image.
    """
    # Pattern 1 — modern Google Images JSON payload
    urls = re.findall(r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp|gif))"', html, re.IGNORECASE)

    # Pattern 2 — older ou= parameter still present in some regions
    ou_urls = re.findall(r'["\s]ou\s*:\s*"(https?://[^"]+)"', html)
    urls += ou_urls

    # Deduplicate while preserving order
    seen = set()
    result = []
    for url in urls:
        url = url.replace("\\u003d", "=").replace("\\u0026", "&")
        if url not in seen and not any(
            bad in url.lower()
            for bad in ["gstatic", "google", "ggpht", "googleapis", "logo", "icon", "sprite"]
        ):
            seen.add(url)
            result.append(url)

    return result


# =========================================================
# SCRAPE SINGLE PERSON
# =========================================================

async def scrape_person(page, person_key: str, search_query: str):

    save_dir = os.path.join("data", "raw", person_key)
    os.makedirs(save_dir, exist_ok=True)

    # Skip if already fully downloaded
    existing = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
    if existing >= MAX_IMAGES_PER_PERSON:
        print(f"\n⏭  {person_key} already has {existing} images — skipping.")
        return

    print("\n" + "=" * 70)
    print(f"🔥 Scraping: {person_key}  ({search_query})")
    print("=" * 70)

    # ---------------------------------------------------------
    # Build URL:
    #   tbm=isch  → image search
    #   tbs=itp:face,isz:l  → face filter + LARGE size only (HD at source)
    # ---------------------------------------------------------
    query_encoded = search_query.replace(" ", "+")
    url = f"https://www.google.com/search?tbm=isch&q={query_encoded}&tbs=itp:face,isz:l"

    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)

    # Accept cookies / consent dialog if shown (EU / some regions)
    for selector in ["button[id*='accept']", "button[aria-label*='Accept']", "#L2AGLb"]:
        try:
            btn = page.locator(selector).first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await page.wait_for_timeout(1500)
                break
        except Exception:
            pass

    # Initial scroll — load first batch before entering the download loop
    print("  🔽 Initial scroll to load first batch...")
    for _ in range(5):
        await page.mouse.wheel(0, 5000)
        await page.wait_for_timeout(random.randint(1200, 2000))

    downloaded = existing
    skipped    = 0
    used_urls:   set[str]  = set()
    seen_hashes: list[str] = []   # fresh per person — MD5 + pHash of every saved image

    progress = tqdm(
        total=MAX_IMAGES_PER_PERSON,
        initial=existing,
        desc=person_key,
        unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} saved [{elapsed}]"
    )

    scroll_round      = 0
    max_scroll_rounds = 10          # scroll up to 10 extra rounds if needed
    no_new_url_streak = 0           # how many rounds gave zero new URLs

    while downloaded < MAX_IMAGES_PER_PERSON:

        # ── Fetch all URLs visible on page right now ──────
        html           = await page.content()
        all_urls       = extract_image_urls(html)
        new_urls       = [u for u in all_urls if u not in used_urls]

        if not new_urls and scroll_round == 0:
            # First round found nothing — try fallback click method once
            print("  ⚠️  No URLs from HTML — trying thumbnail click fallback")
            fallback = await _click_method_fallback(page)
            new_urls = [u for u in fallback if u not in used_urls]

        print(f"  [round {scroll_round}] {len(new_urls)} new candidate URLs")

        if not new_urls:
            no_new_url_streak += 1
            if no_new_url_streak >= 3 or scroll_round >= max_scroll_rounds:
                print("  ℹ️  No more new URLs found — stopping early")
                break
        else:
            no_new_url_streak = 0

        # ── Try downloading every new URL ─────────────────
        for img_url in new_urls:

            if downloaded >= MAX_IMAGES_PER_PERSON:
                break

            used_urls.add(img_url)          # mark as seen regardless of outcome

            filename  = f"{person_key}_{downloaded + 1:04d}.jpg"
            save_path = os.path.join(save_dir, filename)

            success = download_image(img_url, save_path, seen_hashes)

            if success:
                downloaded += 1
                progress.update(1)
                progress.set_postfix(saved=downloaded, skipped=skipped, refresh=True)
            else:
                skipped += 1
                if os.path.exists(save_path):
                    os.remove(save_path)
                progress.set_postfix(saved=downloaded, skipped=skipped, refresh=True)

        # ── If still need more, scroll down to load new images
        if downloaded < MAX_IMAGES_PER_PERSON:

            if scroll_round >= max_scroll_rounds:
                print("  ℹ️  Max scroll rounds reached — stopping")
                break

            print(f"  🔽 Need {MAX_IMAGES_PER_PERSON - downloaded} more — scrolling for new images...")

            for _ in range(5):
                await page.mouse.wheel(0, 5000)
                await page.wait_for_timeout(random.randint(1000, 1800))

            # Click "Show more results" if present
            for label in ["Show more results", "More results"]:
                try:
                    btn = page.get_by_text(label, exact=False).first
                    if await btn.is_visible(timeout=500):
                        await btn.click()
                        await page.wait_for_timeout(1500)
                except Exception:
                    pass

            scroll_round += 1

    progress.close()

    if downloaded < MAX_IMAGES_PER_PERSON:
        print(
            f"\n⚠️  Only {downloaded} images saved for {person_key} "
            f"({skipped} skipped — Google may not have more large face images)"
        )
    else:
        print(f"\n✅ Completed {person_key} — {downloaded} saved, {skipped} skipped")
    else:
        print(f"\n✅ Completed {person_key} — {downloaded} saved, {skipped} skipped")


# =========================================================
# FALLBACK: click each thumbnail and grab the large image
# (used only when the JSON extraction finds nothing)
# =========================================================

async def _click_method_fallback(page) -> list[str]:
    urls: list[str] = []

    # Modern selector — Google Images uses 'img[data-src]' or plain 'img' in results grid
    selectors = [
        "div[data-ri] img",          # 2024 grid layout
        "img.Q4LuWd",                # older class
        "img.rg_i",                  # very old (may still work in some regions)
    ]

    thumbnails = []
    for sel in selectors:
        thumbnails = await page.locator(sel).all()
        if thumbnails:
            break

    print(f"  [fallback] thumbnails found: {len(thumbnails)}")

    for thumb in thumbnails[:150]:
        try:
            await thumb.click(timeout=3000)
            await page.wait_for_timeout(2000)

            # The expanded panel shows the large image — grab its src
            for large_sel in [
                "img[jsname='kn3ccd']",   # 2024 large preview
                "img.sFlh5c",
                "img.n3VNCb",
            ]:
                try:
                    large = page.locator(large_sel).first
                    src = await large.get_attribute("src", timeout=2000)
                    if src and src.startswith("http") and not src.startswith("data:"):
                        urls.append(src)
                        break
                except Exception:
                    pass
        except Exception:
            continue

    return urls


# =========================================================
# MAIN
# =========================================================

async def main():

    print("\n🚀 Starting Politician Dataset Scraper (HD Mode)...\n")
    os.makedirs("data/raw", exist_ok=True)

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False,
            slow_mo=100,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1400, "height": 900},
            locale="en-US",
        )

        # Stealth: hide navigator.webdriver
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        page = await context.new_page()

        for person_key, search_query in POLITICIANS.items():
            try:
                await scrape_person(page, person_key, search_query)

                sleep_time = random.randint(8, 15)
                print(f"\n😴 Sleeping {sleep_time}s before next person...\n")
                await asyncio.sleep(sleep_time)

            except Exception as e:
                print(f"\n❌ Error scraping {person_key}: {e}")

        await browser.close()

    print("\n🎉 ALL SCRAPING COMPLETED!")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    asyncio.run(main())