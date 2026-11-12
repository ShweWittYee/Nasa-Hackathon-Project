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

# NASA Mars Trek: Viking VIS global color mosaic.
# This is a global color image mosaic assembled from Viking imagery,
# not a collection of individual raw spacecraft photographs.
PLANET = "Mars"
PRODUCT = "Mars_Viking_MDIM21_ClrMosaic_global_232m"

BASE_URL = (
    "https://trek.nasa.gov/tiles/"
    f"{PLANET}/EQ/{PRODUCT}/"
    "1.0.0/default/default028mm"
)

# Keep the zoom and target from the uploaded script.
ZOOM_LEVEL = 4
TARGET_IMAGES = 30

# Number of candidate URLs processed in one group.
BATCH_SIZE = 100

# Maximum simultaneous requests.
MAX_WORKERS = 4
MAX_RETRIES = 3
TIMEOUT = 45
JPEG_QUALITY = 95
BATCH_PAUSE_SECONDS = 1
RANDOM_SEED = 42

HEADERS = {
    "User-Agent": "MarsAIResearchProject/1.0"
}

# Use separate Mars folders so Mars and Moon images are never mixed.
OUTPUT_DIR = BASE_DIR / "mars_photos"
ZOOM_DIR = OUTPUT_DIR / f"zoom_{ZOOM_LEVEL}_{TARGET_IMAGES}"
MANIFEST_FILE = (
    OUTPUT_DIR / f"manifest_mars_zoom_{ZOOM_LEVEL}_{TARGET_IMAGES}.csv"
)

# Global equirectangular tile-grid convention:
# zoom 0 = 1 row x 2 columns; each level doubles each dimension.
# Confirmed for global equirectangular products by NASA Trek documentation.
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
# VALIDATE EXISTING JPG
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
# FIND EXISTING MARS IMAGES
# =====================================================

def get_existing_images():
    existing = []

    for path in ZOOM_DIR.glob("*.jpg"):
        if is_valid_jpg(path):
            existing.append(path)
        else:
            logger.warning("Removing invalid JPG: %s", path.name)
            path.unlink(missing_ok=True)

    return existing


# =====================================================
# CHECK DNS AND TEST A REAL TILE
# =====================================================

def check_connection():
    host = urlparse(BASE_URL).hostname
    logger.info("Checking NASA server: %s", host)

    try:
        socket.getaddrinfo(host, 443)
        logger.info("DNS resolution successful.")
    except socket.gaierror as error:
        raise RuntimeError(
            f"Cannot resolve {host}. Check your internet connection or DNS."
        ) from error

    # Test several positions. A single 404 can simply mean that
    # the mosaic has no data in that particular tile.
    sample_positions = [
        (TOTAL_ROWS // 2, TOTAL_COLS // 2),
        (TOTAL_ROWS // 4, TOTAL_COLS // 2),
        (3 * TOTAL_ROWS // 4, TOTAL_COLS // 2),
        (TOTAL_ROWS // 2, TOTAL_COLS // 4),
    ]

    last_error = None

    for row, col in sample_positions:
        test_url = f"{BASE_URL}/{ZOOM_LEVEL}/{row}/{col}.jpg"
        logger.info("Testing Mars tile: row=%s col=%s", row, col)

        try:
            response = requests.get(
                test_url,
                headers=HEADERS,
                timeout=TIMEOUT
            )

            if response.status_code == 404:
                logger.warning(
                    "Sample tile has no coverage (HTTP 404): row=%s col=%s",
                    row, col
                )
                continue

            response.raise_for_status()

            with Image.open(io.BytesIO(response.content)) as image:
                if image.format != "JPEG":
                    raise RuntimeError(
                        f"Expected JPEG but received {image.format}"
                    )
                image.verify()

            logger.info(
                "Mars JPG endpoint test successful at row=%s col=%s",
                row, col
            )
            return

        except Exception as error:
            last_error = error
            logger.warning("Sample tile test failed: %s", error)

    raise RuntimeError(
        "None of the sample Mars tiles returned a valid JPG. "
        "Check the product endpoint, zoom matrix, and service availability. "
        f"Last error: {last_error}"
    )


# =====================================================
# MANIFEST
# =====================================================

MANIFEST_COLUMNS = [
    "planet",
    "product",
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
    pattern = re.compile(
        rf"mars_z{ZOOM_LEVEL}_r(\d+)_c(\d+)\.jpg$"
    )

    existing_rows = []

    for path in existing_paths:
        match = pattern.fullmatch(path.name)
        row = int(match.group(1)) if match else ""
        col = int(match.group(2)) if match else ""

        existing_rows.append({
            "planet": PLANET,
            "product": PRODUCT,
            "zoom": ZOOM_LEVEL,
            "row": row,
            "col": col,
            "filename": path.name,
            "url": "",
            "status": "already_exists",
            "size_bytes": path.stat().st_size,
            "error": ""
        })

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
        writer.writerows(existing_rows)

    logger.info("Manifest initialized: %s", MANIFEST_FILE)


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
# GENERATE RANDOMLY DISTRIBUTED MARS TILES
# =====================================================

def generate_tiles():
    coordinates = [
        (row, col)
        for row in range(TOTAL_ROWS)
        for col in range(TOTAL_COLS)
    ]

    # Deterministically spread requests across the whole global mosaic.
    random.Random(RANDOM_SEED).shuffle(coordinates)

    logger.info(
        "Generated %s candidate coordinates for Mars zoom %s",
        len(coordinates),
        ZOOM_LEVEL
    )

    for row, col in coordinates:
        filename = f"mars_z{ZOOM_LEVEL}_r{row}_c{col}.jpg"
        output_path = ZOOM_DIR / filename

        if is_valid_jpg(output_path):
            continue

        url = f"{BASE_URL}/{ZOOM_LEVEL}/{row}/{col}.jpg"

        yield {
            "planet": PLANET,
            "product": PRODUCT,
            "zoom": ZOOM_LEVEL,
            "row": row,
            "col": col,
            "filename": filename,
            "url": url,
            "output_path": output_path
        }


# =====================================================
# DOWNLOAD ONE MARS TILE AND SAVE AS JPG
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

            if response.status_code == 404:
                return {
                    **tile,
                    "status": "not_available",
                    "size_bytes": 0,
                    "error": "HTTP 404"
                }

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < MAX_RETRIES:
                    time.sleep(attempt * 3)
                    continue

            response.raise_for_status()
            image_data = response.content

            # Validate the server response as an image.
            with Image.open(io.BytesIO(image_data)) as image:
                image_format = image.format
                image.verify()

            # Preserve JPEG bytes; convert other image formats to JPEG.
            if image_format == "JPEG":
                temp_path.write_bytes(image_data)
            else:
                with Image.open(io.BytesIO(image_data)) as image:
                    image.convert("RGB").save(
                        temp_path,
                        format="JPEG",
                        quality=JPEG_QUALITY,
                        optimize=True
                    )

            # Publish the file only after a complete write.
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
# PROCESS ONE BATCH
# =====================================================

def process_batch(batch, executor):
    results = []

    futures = [
        executor.submit(download_tile, tile)
        for tile in batch
    ]

    for future in as_completed(futures):
        results.append(future.result())

    results.sort(key=lambda item: (item["row"], item["col"]))
    save_batch_manifest(results)

    return results


# =====================================================
# MAIN
# =====================================================

def main():
    logger.info("=" * 65)
    logger.info("NASA MARS JPG DOWNLOADER")
    logger.info("Planet: %s", PLANET)
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

    existing_paths = get_existing_images()
    saved_count = len(existing_paths)

    if saved_count > TARGET_IMAGES:
        logger.warning(
            "The output folder already contains %s valid JPGs, "
            "more than the target of %s. No files were deleted.",
            saved_count,
            TARGET_IMAGES
        )
        return

    initialize_manifest(existing_paths)

    if saved_count >= TARGET_IMAGES:
        logger.info("Target already reached: %s valid JPGs", saved_count)
        return

    total_new_downloads = 0
    total_unavailable = 0
    total_failed = 0
    processed_candidates = 0
    batch_number = 0

    tile_iterator = generate_tiles()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(
            total=TOTAL_CANDIDATES,
            desc=f"Searching Mars zoom {ZOOM_LEVEL}",
            unit="tile"
        ) as progress:

            while saved_count < TARGET_IMAGES:
                remaining_images = TARGET_IMAGES - saved_count

                # Do not schedule more successful candidates than needed.
                current_batch_size = min(
                    BATCH_SIZE,
                    remaining_images
                )

                batch = list(islice(tile_iterator, current_batch_size))

                if not batch:
                    logger.warning("No more candidate tiles available.")
                    break

                batch_number += 1

                logger.info(
                    "Starting batch %s | candidates=%s | images needed=%s",
                    batch_number,
                    len(batch),
                    remaining_images
                )

                results = process_batch(batch, executor)
                progress.update(len(batch))
                processed_candidates += len(batch)

                downloaded = sum(
                    result["status"] == "downloaded"
                    for result in results
                )
                unavailable = sum(
                    result["status"] == "not_available"
                    for result in results
                )
                failed = sum(
                    result["status"] == "failed"
                    for result in results
                )
                batch_size = sum(
                    result.get("size_bytes", 0)
                    for result in results
                )

                total_new_downloads += downloaded
                total_unavailable += unavailable
                total_failed += failed
                saved_count += downloaded

                logger.info("-" * 60)
                logger.info("Batch %s completed", batch_number)
                logger.info("Candidates checked: %s", processed_candidates)
                logger.info("New JPGs: %s", downloaded)
                logger.info("Unavailable: %s", unavailable)
                logger.info("Failed: %s", failed)
                logger.info(
                    "Valid JPG total: %s/%s",
                    saved_count,
                    TARGET_IMAGES
                )
                logger.info(
                    "Batch size: %.2f MB",
                    batch_size / (1024 ** 2)
                )
                logger.info("Manifest updated: %s", MANIFEST_FILE)
                logger.info("-" * 60)

                if saved_count < TARGET_IMAGES:
                    time.sleep(BATCH_PAUSE_SECONDS)

    final_paths = [
        path for path in ZOOM_DIR.glob("*.jpg")
        if is_valid_jpg(path)
    ]
    final_count = len(final_paths)
    final_size = sum(path.stat().st_size for path in final_paths)

    logger.info("=" * 65)
    logger.info("MARS DOWNLOAD FINISHED")
    logger.info("Planet: %s", PLANET)
    logger.info("Product: %s", PRODUCT)
    logger.info("Zoom level: %s", ZOOM_LEVEL)
    logger.info("Target JPGs: %s", TARGET_IMAGES)
    logger.info("Valid JPG files: %s", final_count)
    logger.info("New downloads: %s", total_new_downloads)
    logger.info("Candidate tiles checked: %s", processed_candidates)
    logger.info("Unavailable tiles: %s", total_unavailable)
    logger.info("Failed requests: %s", total_failed)
    logger.info("Total image size: %.2f MB", final_size / (1024 ** 2))
    logger.info("Output: %s", ZOOM_DIR)
    logger.info("Manifest: %s", MANIFEST_FILE)

    if final_count < TARGET_IMAGES:
        logger.warning(
            "Only %s valid JPGs were found. Check the product tile "
            "matrix, coverage, and service response.",
            final_count
        )
    else:
        logger.info("Target reached: %s Mars JPG images.", TARGET_IMAGES)

    logger.info("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning(
            "Download interrupted. Completed JPGs are preserved; "
            "run the script again to resume."
        )
    except Exception:
        logger.exception("Downloader stopped due to an unexpected error.")
