
import csv
import io
import logging
import random
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from tqdm import tqdm


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

PRODUCT = "LRO_WAC_Mosaic_Global_303ppd_v02"

BASE_URL = (
    "https://trek.nasa.gov/tiles/"
    f"Moon/EQ/{PRODUCT}/"
    "1.0.0/default/default028mm"
)

# DOWNLOAD SETTINGS
ZOOM_LEVEL = 4
TARGET_IMAGES = 30

# Process candidate URLs in batches
BATCH_SIZE = 100

MAX_WORKERS = 4
MAX_RETRIES = 3
TIMEOUT = 45

JPEG_QUALITY = 95
BATCH_PAUSE_SECONDS = 1
RANDOM_SEED = 42

HEADERS = {
    "User-Agent": "MoonAIResearchProject/1.0"
}

# Separate output folder for exactly this experiment
OUTPUT_DIR = BASE_DIR / "moon_photos"

ZOOM_DIR = OUTPUT_DIR / f"zoom_{ZOOM_LEVEL}_100"

MANIFEST_FILE = (
    OUTPUT_DIR / f"manifest_zoom_{ZOOM_LEVEL}_100.csv"
)

# Assumed zoom 8 tile grid
TOTAL_ROWS = 2 ** ZOOM_LEVEL
TOTAL_COLS = 2 ** (ZOOM_LEVEL + 1)

TOTAL_CANDIDATES = TOTAL_ROWS * TOTAL_COLS

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ZOOM_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =====================================================
# VALIDATE JPG
# =====================================================

def is_valid_jpg(path: Path) -> bool:

    if not path.is_file():
        return False

    try:

        with Image.open(path) as image:

            if image.format != "JPEG":
                return False

            image.verify()

        return True

    except Exception:

        return False


# =====================================================
# FIND EXISTING IMAGES
# =====================================================

def get_existing_images():

    existing = []

    for path in ZOOM_DIR.glob("*.jpg"):

        if is_valid_jpg(path):

            existing.append(path)

        else:

            logger.warning(
                "Removing invalid JPG: %s",
                path.name
            )

            path.unlink(missing_ok=True)

    return existing


# =====================================================
# CHECK CONNECTION
# =====================================================

def check_connection():

    host = urlparse(BASE_URL).hostname

    logger.info(
        "Checking NASA server: %s",
        host
    )

    try:

        socket.getaddrinfo(host, 443)

        logger.info(
            "DNS resolution successful."
        )

    except socket.gaierror as error:

        raise RuntimeError(
            f"Cannot resolve {host}. "
            "Check your internet connection or DNS."
        ) from error


# =====================================================
# MANIFEST
# =====================================================

MANIFEST_COLUMNS = [
    "zoom",
    "row",
    "col",
    "filename",
    "url",
    "status",
    "size_bytes",
    "error"
]


def initialize_manifest(existing_paths):

    with open(
        MANIFEST_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_COLUMNS
        )

        writer.writeheader()

        pattern = re.compile(
            rf"moon_z{ZOOM_LEVEL}_r(\d+)_c(\d+)\.jpg$"
        )

        existing_rows = []

        for path in existing_paths:

            match = pattern.fullmatch(path.name)

            row = int(match.group(1)) if match else ""
            col = int(match.group(2)) if match else ""

            existing_rows.append({
                "zoom": ZOOM_LEVEL,
                "row": row,
                "col": col,
                "filename": path.name,
                "url": "",
                "status": "already_exists",
                "size_bytes": path.stat().st_size,
                "error": ""
            })

        writer.writerows(existing_rows)

    logger.info(
        "Manifest initialized: %s",
        MANIFEST_FILE
    )


def save_batch_manifest(results):

    with open(
        MANIFEST_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_COLUMNS,
            extrasaction="ignore"
        )

        writer.writerows(results)

        file.flush()


# =====================================================
# GENERATE RANDOMLY DISTRIBUTED ZOOM 8 TILES
# =====================================================

def generate_tiles():

    coordinates = [
        (row, col)
        for row in range(TOTAL_ROWS)
        for col in range(TOTAL_COLS)
    ]

    # Deterministic random order:
    # spread candidate tiles across the map.

    random.Random(RANDOM_SEED).shuffle(coordinates)

    logger.info(
        "Generated %s candidate coordinates",
        len(coordinates)
    )

    for row, col in coordinates:

        filename = (
            f"moon_z{ZOOM_LEVEL}_"
            f"r{row}_c{col}.jpg"
        )

        output_path = ZOOM_DIR / filename

        # Skip valid images from previous runs

        if is_valid_jpg(output_path):
            continue

        url = (
            f"{BASE_URL}/"
            f"{ZOOM_LEVEL}/{row}/{col}.jpg"
        )

        yield {
            "zoom": ZOOM_LEVEL,
            "row": row,
            "col": col,
            "filename": filename,
            "url": url,
            "output_path": output_path
        }


# =====================================================
# DOWNLOAD ONE JPG
# =====================================================

def download_tile(tile):

    output_path = tile["output_path"]

    temp_path = output_path.with_suffix(".jpg.part")

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                tile["url"],
                headers=HEADERS,
                timeout=TIMEOUT
            )

            # Tile not available

            if response.status_code == 404:

                return {
                    **tile,
                    "status": "not_available",
                    "size_bytes": 0,
                    "error": "HTTP 404"
                }

            # Retry temporary server errors

            if response.status_code in (
                429, 500, 502, 503, 504
            ):

                if attempt < MAX_RETRIES:

                    time.sleep(attempt * 3)
                    continue

            response.raise_for_status()

            image_data = response.content

            # Validate image

            with Image.open(
                io.BytesIO(image_data)
            ) as image:

                image_format = image.format

                image.verify()

            # Save as JPG

            if image_format == "JPEG":

                temp_path.write_bytes(image_data)

            else:

                with Image.open(
                    io.BytesIO(image_data)
                ) as image:

                    image.convert("RGB").save(
                        temp_path,
                        format="JPEG",
                        quality=JPEG_QUALITY,
                        optimize=True
                    )

            # Atomic file replacement

            temp_path.replace(output_path)

            return {
                **tile,
                "status": "downloaded",
                "size_bytes": output_path.stat().st_size,
                "error": ""
            }

        except requests.exceptions.RequestException as error:

            if attempt == MAX_RETRIES:

                temp_path.unlink(missing_ok=True)

                return {
                    **tile,
                    "status": "failed",
                    "size_bytes": 0,
                    "error": str(error)
                }

            time.sleep(attempt * 2)

        except Exception as error:

            temp_path.unlink(missing_ok=True)

            return {
                **tile,
                "status": "failed",
                "size_bytes": 0,
                "error": str(error)
            }

    return {
        **tile,
        "status": "failed",
        "size_bytes": 0,
        "error": "Maximum retries exceeded"
    }


# =====================================================
# PROCESS BATCH
# =====================================================

def process_batch(batch, executor):

    results = []

    futures = [
        executor.submit(download_tile, tile)
        for tile in batch
    ]

    for future in as_completed(futures):

        results.append(future.result())

    results.sort(
        key=lambda item: (
            item["row"],
            item["col"]
        )
    )

    save_batch_manifest(results)

    return results


# =====================================================
# MAIN
# =====================================================

def main():

    logger.info("=" * 65)

    logger.info("NASA MOON JPG DOWNLOADER")

    logger.info("Product: %s", PRODUCT)

    logger.info("Zoom level: %s ONLY", ZOOM_LEVEL)

    logger.info("Target images: %s", TARGET_IMAGES)

    logger.info("Tile rows: %s", TOTAL_ROWS)

    logger.info("Tile columns: %s", TOTAL_COLS)

    logger.info("Candidate coordinates: %s", TOTAL_CANDIDATES)

    logger.info("Batch size: %s", BATCH_SIZE)

    logger.info("Concurrent downloads: %s", MAX_WORKERS)

    logger.info("Output: %s", ZOOM_DIR)

    logger.info("=" * 65)

    check_connection()

    # Find images already downloaded

    existing_paths = get_existing_images()

    existing_count = len(existing_paths)

    # Avoid adding images if the folder already exceeds target

    if existing_count > TARGET_IMAGES:

        logger.warning(
            "Existing valid JPG count (%s) exceeds target (%s). "
            "No new images will be downloaded.",
            existing_count,
            TARGET_IMAGES
        )

        return

    initialize_manifest(existing_paths)

    saved_count = existing_count

    total_new_downloads = 0
    total_unavailable = 0
    total_failed = 0
    processed_candidates = 0

    batch_number = 0

    if saved_count >= TARGET_IMAGES:

        logger.info(
            "Target already reached: %s valid JPGs",
            saved_count
        )

        return

    tile_iterator = generate_tiles()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        with tqdm(
            total=TOTAL_CANDIDATES,
            desc=f"Searching zoom {ZOOM_LEVEL} tiles",
            unit="tile"
        ) as progress:

            while saved_count < TARGET_IMAGES:

                remaining_images = (
                    TARGET_IMAGES - saved_count
                )

                # Never process more candidates than
                # the number of images still required.

                current_batch_size = min(
                    BATCH_SIZE,
                    remaining_images
                )

                batch = list(
                    islice(
                        tile_iterator,
                        current_batch_size
                    )
                )

                if not batch:

                    logger.warning(
                        "No more candidate tiles available."
                    )

                    break

                batch_number += 1

                logger.info(
                    "Starting batch %s | "
                    "Candidates=%s | "
                    "Images needed=%s",
                    batch_number,
                    len(batch),
                    remaining_images
                )

                results = process_batch(
                    batch,
                    executor
                )

                progress.update(len(batch))

                processed_candidates += len(batch)

                # Batch statistics

                downloaded = sum(
                    r["status"] == "downloaded"
                    for r in results
                )

                unavailable = sum(
                    r["status"] == "not_available"
                    for r in results
                )

                failed = sum(
                    r["status"] == "failed"
                    for r in results
                )

                batch_size = sum(
                    r.get("size_bytes", 0)
                    for r in results
                )

                total_new_downloads += downloaded

                total_unavailable += unavailable

                total_failed += failed

                saved_count += downloaded

                logger.info("-" * 60)

                logger.info(
                    "BATCH %s COMPLETED",
                    batch_number
                )

                logger.info(
                    "Candidates checked: %s",
                    processed_candidates
                )

                logger.info(
                    "New JPGs: %s",
                    downloaded
                )

                logger.info(
                    "Unavailable: %s",
                    unavailable
                )

                logger.info(
                    "Failed: %s",
                    failed
                )

                logger.info(
                    "Valid JPG total: %s/%s",
                    saved_count,
                    TARGET_IMAGES
                )

                logger.info(
                    "Batch size: %.2f MB",
                    batch_size / (1024 ** 2)
                )

                logger.info(
                    "Manifest updated: %s",
                    MANIFEST_FILE
                )

                logger.info("-" * 60)

                if (
                    saved_count < TARGET_IMAGES
                    and BATCH_PAUSE_SECONDS > 0
                ):

                    time.sleep(BATCH_PAUSE_SECONDS)

    # =================================================
    # FINAL SUMMARY
    # =================================================

    final_paths = [
        path
        for path in ZOOM_DIR.glob("*.jpg")
        if is_valid_jpg(path)
    ]

    final_count = len(final_paths)

    final_size = sum(
        path.stat().st_size
        for path in final_paths
    )

    logger.info("=" * 65)

    logger.info("DOWNLOAD FINISHED")

    logger.info("Zoom level: %s", ZOOM_LEVEL)

    logger.info("Target JPGs: %s", TARGET_IMAGES)

    logger.info("Valid JPG files: %s", final_count)

    logger.info("New downloads: %s", total_new_downloads)

    logger.info(
        "Candidate tiles checked: %s",
        processed_candidates
    )

    logger.info("Unavailable tiles: %s", total_unavailable)

    logger.info("Failed requests: %s", total_failed)

    logger.info(
        "Total image size: %.2f MB",
        final_size / (1024 ** 2)
    )

    logger.info("Output: %s", ZOOM_DIR)

    logger.info("Manifest: %s", MANIFEST_FILE)

    if final_count < TARGET_IMAGES:

        logger.warning(
            "Only %s valid JPGs were found. "
            "Verify zoom-8 tile availability and "
            "WMTS matrix dimensions.",
            final_count
        )

    else:

        logger.info(
            "Successfully downloaded %s JPG images.",
            TARGET_IMAGES
        )

    logger.info("=" * 65)


# =====================================================
# EXECUTION
# =====================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        logger.warning(
            "Download interrupted. "
            "Completed JPG files are preserved. "
            "Run again to continue."
        )

    except Exception:

        logger.exception(
            "Downloader stopped because of an unexpected error."
        )
