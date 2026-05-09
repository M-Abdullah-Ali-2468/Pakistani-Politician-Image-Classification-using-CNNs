import os
import re
import time
import json
import random
import asyncio
import requests

from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
import pytesseract
import numpy as np
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
# TEXT DETECTION SETTINGS
# Images with too many detected characters are rejected.
# OCR runs on a small grayscale version for speed.
# Raise MAX_TEXT_CHARS to be more lenient (allow some text),
# lower it to be stricter (only accept near-zero text).
# =========================================================
MAX_TEXT_CHARS = 15   # max OCR characters allowed in the whole image

# =========================================================
# TEXT DETECTION — rejects images with news tickers,
# watermarks, captions, or heavy Urdu/English overlays.
# Uses Tesseract OCR on a small greyscale version for speed.
# =========================================================

def has_too_much_text(image_bytes: bytes) -> bool:
    """
    Returns True if the image contains more than MAX_TEXT_CHARS
    characters of detectable text (English or Urdu/Arabic script).
    Runs OCR on a shrunken greyscale copy — fast enough per image.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("L")  # greyscale

        # Shrink to max 800px wide — OCR doesn't need full resolution
        max_side = 800
        if img.width > max_side:
            ratio = max_side / img.width
            img = img.resize(
                (max_side, int(img.height * ratio)),
                Image.LANCZOS
            )

        # Sharpen edges slightly — helps OCR on soft images
        img = img.filter(ImageFilter.SHARPEN)

        # Run OCR for both Latin and Arabic/Urdu scripts
        # osd = no orientation detection (faster)
        custom_config = r"--oem 3 --psm 11 -l eng+urd"
        text = pytesseract.image_to_string(img, config=custom_config)

        # Count only non-whitespace characters
        char_count = len(text.replace(" ", "").replace("\n", "").strip())

        if char_count > MAX_TEXT_CHARS:
            print(f"  [text-skip] {char_count} chars detected — image rejected")
            return True

        return False

    except Exception as e:
        # If OCR fails, don't block the image
        print(f"  [text-check error] {e} — allowing image")
        return False


# =========================================================
# DOWNLOAD IMAGE — with content-type + HD dimension check
# =========================================================

def download_image(img_url: str, save_path: str) -> bool:
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

    # ---------------------------------------------------------
    # Scroll to load more images (20 scrolls)
    # ---------------------------------------------------------
    for _ in range(20):
        await page.mouse.wheel(0, 5000)
        await page.wait_for_timeout(random.randint(1200, 2200))

        # Click "Show more results" button if present
        for label in ["Show more results", "More results"]:
            try:
                btn = page.get_by_text(label, exact=False).first
                if await btn.is_visible(timeout=500):
                    await btn.click()
                    await page.wait_for_timeout(1500)
            except Exception:
                pass

    # ---------------------------------------------------------
    # Extract URLs from the full page source (fast & reliable)
    # ---------------------------------------------------------
    html = await page.content()
    candidate_urls = extract_image_urls(html)
    print(f"  Candidate image URLs found: {len(candidate_urls)}")

    if not candidate_urls:
        print("  ⚠️  No URLs extracted — falling back to thumbnail click method")
        candidate_urls = await _click_method_fallback(page)

    downloaded = existing
    used_urls:  set[str] = set()
    progress    = tqdm(total=MAX_IMAGES_PER_PERSON, initial=existing, desc=person_key)

    for img_url in candidate_urls:

        if downloaded >= MAX_IMAGES_PER_PERSON:
            break

        if img_url in used_urls:
            continue

        filename  = f"{person_key}_{downloaded + 1:04d}.jpg"
        save_path = os.path.join(save_dir, filename)

        success = download_image(img_url, save_path)

        if success:
            used_urls.add(img_url)
            downloaded += 1
            progress.update(1)
        else:
            # Clean up empty/bad file
            if os.path.exists(save_path):
                os.remove(save_path)

    progress.close()
    print(f"\n✅ Completed {person_key} — total: {downloaded} images")


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