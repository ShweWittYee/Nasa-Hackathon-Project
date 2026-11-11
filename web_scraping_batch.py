
import csv
import io
import logging
import math
import socket
import time

from pathlib import Path
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from PIL import Image
from tqdm import tqdm


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "moon_photos"

PRODUCT = "LRO_WAC_Mosaic_Global_303ppd_v02"

BASE_URL = (
    "https://trek.nasa.gov/tiles/"
    f"Moon/EQ/{PRODUCT}/"
    "1.0.0/default/default028mm"
)

# Download ONLY zoom level 8
ZOOM_LEVEL = 8

# Process 1,000 tiles per batch
BATCH_SIZE = 1000

# Maximum simultaneous HTTP requests
MAX_WORKERS = 4

# Retry configuration
MAX_RETRIES = 3

TIMEOUT = 45

JPEG_QUALITY = 95

# Pause between batches
BATCH_PAUSE_SECONDS = 1

HEADERS = {
    "User-Agent": "MoonAIResearchProject/1.0"
}

ZOOM_DIR = OUTPUT_DIR / f"zoom_{ZOOM_LEVEL}"

MANIFEST_FILE = (
    OUTPUT_DIR / f"manifest_zoom_{ZOOM_LEVEL}.csv"
)

# Tile grid for zoom level 8
TOTAL_ROWS = 2 ** ZOOM_LEVEL
TOTAL_COLS = 2 ** (ZOOM_LEVEL + 1)

TOTAL_TILES = TOTAL_ROWS * TOTAL_COLS

TOTAL_BATCHES = math.ceil(
    TOTAL_TILES / BATCH_SIZE
)


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =====================================================
# CREATE DIRECTORIES
# =====================================================

ZOOM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# CHECK DNS CONNECTION
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
            f"Cannot resolve NASA server: {host}. "
            "Check internet connection or DNS."
        ) from error


# =====================================================
# GENERATE ZOOM 8 TILES
# =====================================================

def generate_tiles():

    logger.info(
        "Generating tiles for zoom level %s",
        ZOOM_LEVEL
    )

    for row in range(TOTAL_ROWS):

        for col in range(TOTAL_COLS):

            filename = (
                f"moon_z{ZOOM_LEVEL}_"
                f"r{row}_c{col}.jpg"
            )

            output_path = ZOOM_DIR / filename

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
# VALIDATE EXISTING JPG
# =====================================================

def is_valid_jpg(path):

    if not path.exists():
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
# DOWNLOAD ONE TILE
# =====================================================

def download_tile(tile):

    output_path = tile["output_path"]

    temp_path = output_path.with_suffix(
        ".jpg.part"
    )

    # Skip existing valid images

    if is_valid_jpg(output_path):

        return {
            **tile,
            "status": "already_exists",
            "size_bytes": output_path.stat().st_size
        }

    # Remove incomplete or invalid files

    output_path.unlink(missing_ok=True)
    temp_path.unlink(missing_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                tile["url"],
                headers=HEADERS,
                timeout=TIMEOUT
            )

            # Missing tiles mean no data coverage

            if response.status_code == 404:

                return {
                    **tile,
                    "status": "not_available",
                    "size_bytes": 0
                }

            # Retry temporary server errors

            if response.status_code in (
                429, 500, 502, 503, 504
            ):

                if attempt < MAX_RETRIES:

                    time.sleep(attempt * 3)
                    continue

                response.raise_for_status()

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

                # Preserve original JPEG quality

                temp_path.write_bytes(image_data)

            else:

                # Convert other image formats to JPG

                with Image.open(
                    io.BytesIO(image_data)
                ) as image:

                    image.convert("RGB").save(
                        temp_path,
                        format="JPEG",
                        quality=JPEG_QUALITY,
                        optimize=True
                    )

            # Move completed image into final location

            temp_path.replace(output_path)

            return {
                **tile,
                "status": "downloaded",
                "size_bytes": output_path.stat().st_size
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
        "size_bytes": 0
    }


# =====================================================
# INITIALIZE MANIFEST
# =====================================================

def initialize_manifest():

    columns = [
        "zoom",
        "row",
        "col",
        "filename",
        "url",
        "status",
        "size_bytes",
        "error"
    ]

    with open(
        MANIFEST_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

    logger.info(
        "Manifest initialized: %s",
        MANIFEST_FILE
    )


# =====================================================
# SAVE EACH BATCH TO MANIFEST
# =====================================================

def save_batch_manifest(results):

    columns = [
        "zoom",
        "row",
        "col",
        "filename",
        "url",
        "status",
        "size_bytes",
        "error"
    ]

    with open(
        MANIFEST_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore"
        )

        writer.writerows(results)

        file.flush()


# =====================================================
# PROCESS ONE BATCH OF 1,000 TILES
# =====================================================

def process_batch(batch, executor, progress):

    results = []

    futures = [
        executor.submit(download_tile, tile)
        for tile in batch
    ]

    for future in as_completed(futures):

        result = future.result()

        results.append(result)

        progress.update(1)

    # Sort results by geographic tile position

    results.sort(
        key=lambda item: (
            item["row"],
            item["col"]
        )
    )

    # Save completed batch immediately

    save_batch_manifest(results)

    return results


# =====================================================
# MAIN DOWNLOADER
# =====================================================

def main():

    logger.info("=" * 60)

    logger.info("NASA MOON JPG DOWNLOADER")

    logger.info("Product: %s", PRODUCT)

    logger.info("Zoom level: %s", ZOOM_LEVEL)

    logger.info("Tile rows: %s", TOTAL_ROWS)

    logger.info("Tile columns: %s", TOTAL_COLS)

    logger.info("Total candidate tiles: %s", TOTAL_TILES)

    logger.info("Batch size: %s", BATCH_SIZE)

    logger.info("Total batches: %s", TOTAL_BATCHES)

    logger.info("Concurrent downloads: %s", MAX_WORKERS)

    logger.info("Output: %s", ZOOM_DIR)

    logger.info("=" * 60)

    # Check DNS before downloading

    check_connection()

    # Reset manifest.
    # Existing JPG images will still be preserved.

    initialize_manifest()

    tile_iterator = generate_tiles()

    total_downloaded = 0

    total_existing = 0

    total_unavailable = 0

    total_failed = 0

    total_size = 0

    processed = 0

    # Create one reusable thread pool

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        # Overall progress bar

        with tqdm(
            total=TOTAL_TILES,
            desc=f"Downloading Zoom {ZOOM_LEVEL}",
            unit="tile"
        ) as progress:

            for batch_number in range(
                1,
                TOTAL_BATCHES + 1
            ):

                # Get next 1,000 tiles only

                batch = list(
                    islice(
                        tile_iterator,
                        BATCH_SIZE
                    )
                )

                if not batch:
                    break

                logger.info(
                    "Starting batch %s/%s (%s tiles)",
                    batch_number,
                    TOTAL_BATCHES,
                    len(batch)
                )

                # Download and save current batch

                results = process_batch(
                    batch,
                    executor,
                    progress
                )

                # Update statistics

                downloaded = sum(
                    r["status"] == "downloaded"
                    for r in results
                )

                existing = sum(
                    r["status"] == "already_exists"
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
                    r["size_bytes"]
                    for r in results
                )

                total_downloaded += downloaded

                total_existing += existing

                total_unavailable += unavailable

                total_failed += failed

                total_size += batch_size

                processed += len(batch)

                # Batch completion report

                logger.info("-" * 60)

                logger.info(
                    "BATCH %s/%s COMPLETED",
                    batch_number,
                    TOTAL_BATCHES
                )

                logger.info(
                    "Processed: %s/%s",
                    processed,
                    TOTAL_TILES
                )

                logger.info(
                    "New JPGs: %s",
                    downloaded
                )

                logger.info(
                    "Already existed: %s",
                    existing
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
                    "Batch size: %.2f MB",
                    batch_size / (1024 * 1024)
                )

                logger.info(
                    "Manifest updated successfully."
                )

                logger.info("-" * 60)

                # Pause between batches

                if batch_number < TOTAL_BATCHES:

                    time.sleep(BATCH_PAUSE_SECONDS)

    # =================================================
    # FINAL SUMMARY
    # =================================================

    actual_files = sum(
        1 for path in ZOOM_DIR.glob("*.jpg")
        if is_valid_jpg(path)
    )

    logger.info("=" * 60)

    logger.info(
        "ZOOM %s DOWNLOAD COMPLETED",
        ZOOM_LEVEL
    )

    logger.info("Total candidates: %s", TOTAL_TILES)

    logger.info("Processed: %s", processed)

    logger.info("New downloads: %s", total_downloaded)

    logger.info("Already existed: %s", total_existing)

    logger.info("Unavailable: %s", total_unavailable)

    logger.info("Failed: %s", total_failed)

    logger.info("Valid JPG files: %s", actual_files)

    logger.info(
        "Total processed image size: %.2f GB",
        total_size / (1024 ** 3)
    )

    logger.info("Output directory: %s", ZOOM_DIR)

    logger.info("Manifest: %s", MANIFEST_FILE)

    logger.info("=" * 60)


# =====================================================
# EXECUTION
# =====================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        logger.warning(
            "Download interrupted by user. "
            "Completed JPG files are preserved. "
            "Run the script again to resume."
        )

    except Exception as error:

        logger.exception(
            "Downloader stopped: %s",
            error
        )
