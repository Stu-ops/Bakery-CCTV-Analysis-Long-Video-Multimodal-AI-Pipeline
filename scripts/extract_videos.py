"""
Extract CCTV Videos — ZIP Extraction Utility

Extracts the CCTV footage ZIP file to data/videos/ folder.
Renames files to consistent format: CAM_1.mp4 → CAM_5.mp4.

Usage:
    python extract_videos.py
    python extract_videos.py --output data/videos
"""

import argparse
import os
import re
import shutil
import zipfile
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "videos"

CCTV_ZIP_FILENAME = "CCTV Footage-20260529T160731Z-3-00144614ea (3).zip"


def extract_videos(
    zip_path: Path = None,
    output_dir: Path = None,
):
    """
    Extract CCTV footage from ZIP and rename consistently.

    Args:
        zip_path: Path to the CCTV ZIP file.
        output_dir: Where to extract videos.
    """
    if zip_path is None:
        zip_path = DATASETS_DIR / CCTV_ZIP_FILENAME
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT

    if not zip_path.exists():
        logger.error("ZIP file not found: %s", zip_path)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting: %s", zip_path)
    logger.info("Output:     %s", output_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        file_list = z.namelist()
        video_files = [f for f in file_list if f.lower().endswith(".mp4")]

        logger.info("Found %d video files in ZIP", len(video_files))

        for video_file in video_files:
            # Extract the camera number from filename
            basename = os.path.basename(video_file)
            match = re.search(r"CAM\s*(\d+)", basename, re.IGNORECASE)

            if match:
                cam_num = int(match.group(1))
                new_name = f"CAM_{cam_num}.mp4"
            else:
                new_name = basename.replace(" ", "_")

            # Extract file
            logger.info("  Extracting: %s -> %s", video_file, new_name)

            # Extract to temp, then move to final location
            z.extract(video_file, output_dir)
            src = output_dir / video_file
            dst = output_dir / new_name

            if src != dst:
                import time
                for _ in range(5):
                    try:
                        shutil.move(str(src), str(dst))
                        break
                    except PermissionError:
                        time.sleep(1)

            size_mb = dst.stat().st_size / (1024 * 1024)
            logger.info("    Size: %.1f MB", size_mb)

    # Clean up extracted folder structure
    cctv_folder = output_dir / "CCTV Footage"
    if cctv_folder.exists() and cctv_folder.is_dir():
        # Move any remaining files
        for f in cctv_folder.iterdir():
            dest = output_dir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
        # Remove empty directory
        try:
            cctv_folder.rmdir()
        except OSError:
            pass

    # Verify
    extracted = list(output_dir.glob("CAM_*.mp4"))
    logger.info("\nExtraction complete: %d videos", len(extracted))
    for v in sorted(extracted):
        size_mb = v.stat().st_size / (1024 * 1024)
        logger.info("  %s: %.1f MB", v.name, size_mb)

    return [str(v) for v in sorted(extracted)]


def check_videos(video_dir: Path = None):
    """Check if videos are already extracted."""
    if video_dir is None:
        video_dir = DEFAULT_OUTPUT

    if not video_dir.exists():
        return False

    videos = list(video_dir.glob("CAM_*.mp4"))
    return len(videos) >= 5


def main():
    parser = argparse.ArgumentParser(description="Extract CCTV footage from ZIP")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--zip",
        type=str,
        default=str(DATASETS_DIR / CCTV_ZIP_FILENAME),
        help="Path to CCTV ZIP file",
    )
    args = parser.parse_args()

    if check_videos(Path(args.output)):
        logger.info("Videos already extracted to %s", args.output)
        return

    extract_videos(
        zip_path=Path(args.zip),
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
