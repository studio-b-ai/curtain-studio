"""
Download window images and annotations from Open Images V7.

Fetches images annotated with the 'Window' class (/m/0d4v4) from the
Open Images V7 dataset. Downloads bounding box annotations, filters for
window entries, and downloads images in parallel.

Usage:
    python download_openimages.py --output data/openimages/ --max-images 5000
"""

import argparse
import csv
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError, HTTPError

from tqdm import tqdm

# Open Images V7 URLs
OI_BASE = "https://storage.googleapis.com/openimages/v7"
BBOX_CSV_URL = f"{OI_BASE}/oidv7-train-annotations-bbox.csv"
CLASS_DESCRIPTIONS_URL = f"{OI_BASE}/oidv7-class-descriptions.csv"
IMAGE_IDS_URL = f"{OI_BASE}/oidv7-train-images-with-labels-with-rotation.csv"

# Window class in Open Images (FREEBASE MID)
WINDOW_CLASS_ID = "/m/0d4v4"

# Image download URL template (Open Images stores images on CVDF S3)
IMAGE_URL_TEMPLATE = "https://s3.amazonaws.com/open-images-dataset/train/{image_id}.jpg"

# Parallel download workers
DEFAULT_WORKERS = 8


def download_file(url: str, dest: str, description: str = "") -> str:
    """Download a file with progress indication."""
    if os.path.isfile(dest):
        print(f"  Already downloaded: {dest}")
        return dest

    print(f"  Downloading {description or url}...")
    try:
        urlretrieve(url, dest)
    except (URLError, HTTPError) as e:
        print(f"  ERROR downloading {url}: {e}", file=sys.stderr)
        raise
    return dest


def load_bbox_annotations(csv_path: str, class_id: str) -> dict[str, list[dict[str, Any]]]:
    """
    Load bounding box annotations for a specific class.

    Returns dict mapping ImageID -> list of bbox dicts.
    """
    annotations: dict[str, list[dict[str, Any]]] = {}
    total_rows = 0
    matched_rows = 0

    print(f"  Parsing bounding box CSV for class {class_id}...")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if row["LabelName"] == class_id:
                matched_rows += 1
                image_id = row["ImageID"]
                bbox = {
                    "x_min": float(row["XMin"]),
                    "x_max": float(row["XMax"]),
                    "y_min": float(row["YMin"]),
                    "y_max": float(row["YMax"]),
                    "confidence": float(row.get("Confidence", 1.0)),
                    "is_occluded": row.get("IsOccluded", "0") == "1",
                    "is_truncated": row.get("IsTruncated", "0") == "1",
                    "is_group_of": row.get("IsGroupOf", "0") == "1",
                    "is_depiction": row.get("IsDepiction", "0") == "1",
                }
                annotations.setdefault(image_id, []).append(bbox)

    print(f"  Found {matched_rows} window annotations across {len(annotations)} images "
          f"(scanned {total_rows} total rows)")
    return annotations


def download_image(
    image_id: str, output_dir: str
) -> tuple[str, bool, str]:
    """
    Download a single image. Returns (image_id, success, error_message).
    """
    filename = f"{image_id}.jpg"
    dest = os.path.join(output_dir, filename)

    if os.path.isfile(dest):
        return image_id, True, ""

    url = IMAGE_URL_TEMPLATE.format(image_id=image_id)
    try:
        urlretrieve(url, dest)
        return image_id, True, ""
    except (URLError, HTTPError) as e:
        return image_id, False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Open Images V7 window images and annotations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/openimages/",
        help="Output directory (default: data/openimages/)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5000,
        help="Maximum number of images to download (default: 5000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Download thread count (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only download annotations, skip image download",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    images_dir = os.path.join(output_dir, "images")
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Step 1: Download bounding box annotations CSV
    print("Step 1: Download bounding box annotations")
    bbox_csv_path = os.path.join(annotations_dir, "oidv7-train-annotations-bbox.csv")
    download_file(BBOX_CSV_URL, bbox_csv_path, "bounding box annotations (~2.5 GB)")

    # Step 2: Parse annotations for window class
    print("\nStep 2: Parse window annotations")
    annotations = load_bbox_annotations(bbox_csv_path, WINDOW_CLASS_ID)

    if not annotations:
        print("ERROR: No window annotations found. Check class ID.", file=sys.stderr)
        sys.exit(1)

    # Limit to max-images (prefer images with more window annotations)
    image_ids = sorted(
        annotations.keys(),
        key=lambda img_id: len(annotations[img_id]),
        reverse=True,
    )
    image_ids = image_ids[: args.max_images]
    print(f"\n  Selected {len(image_ids)} images (of {len(annotations)} available)")

    # Step 3: Download images
    if not args.skip_download:
        print(f"\nStep 3: Download images ({args.workers} workers)")
        succeeded = 0
        failed = 0
        failed_ids: list[str] = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_image, img_id, images_dir): img_id
                for img_id in image_ids
            }

            with tqdm(total=len(image_ids), desc="Downloading", unit="img") as pbar:
                for future in as_completed(futures):
                    img_id, success, error = future.result()
                    if success:
                        succeeded += 1
                    else:
                        failed += 1
                        failed_ids.append(img_id)
                    pbar.update(1)

        print(f"\n  Downloaded: {succeeded}, Failed: {failed}")

        # Remove failed images from the working set
        for img_id in failed_ids:
            image_ids.remove(img_id)
    else:
        print("\nStep 3: Skipping image download (--skip-download)")

    # Step 4: Create manifest
    print("\nStep 4: Create manifest.json")
    manifest: list[dict[str, Any]] = []
    for img_id in image_ids:
        bboxes = annotations.get(img_id, [])
        # Filter out depictions and group-of annotations
        bboxes_clean = [
            b for b in bboxes if not b["is_depiction"] and not b["is_group_of"]
        ]
        if not bboxes_clean:
            continue

        manifest.append({
            "image_id": img_id,
            "filename": f"{img_id}.jpg",
            "source": "openimages_v7",
            "class": "Window",
            "class_id": WINDOW_CLASS_ID,
            "num_windows": len(bboxes_clean),
            "bboxes": bboxes_clean,
        })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Manifest written: {manifest_path}")
    print(f"  Total images in manifest: {len(manifest)}")
    total_bboxes = sum(entry["num_windows"] for entry in manifest)
    print(f"  Total window annotations: {total_bboxes}")

    # Summary stats
    multi_window = sum(1 for entry in manifest if entry["num_windows"] > 1)
    print(f"  Images with multiple windows: {multi_window}")

    print("\nDone.")


if __name__ == "__main__":
    main()
