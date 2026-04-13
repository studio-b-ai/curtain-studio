"""
Prepare Prolific submissions for model training.

Reads raw submission directories, validates each with validate_submission,
copies valid photos to a flat output directory, and creates a standardized
labels.json for the training pipeline.

Usage:
    python prepare_prolific.py --input data/prolific/raw/ --output data/prolific/prepared/

Expected input structure:
    data/prolific/raw/
        submission_001/
            photo1.jpg
            photo2.jpg
            metadata.json     # {width_inches, height_inches, window_type}
        submission_002/
            ...

Output structure:
    data/prolific/prepared/
        images/
            sub001_photo1.jpg
            sub001_photo2.jpg
            sub002_photo1.jpg
            ...
        labels.json
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from validate_submission import validate_submission, VALID_WINDOW_TYPES


def load_submission(submission_dir: str) -> dict[str, Any] | None:
    """
    Load a submission from a directory containing photo1.jpg, photo2.jpg,
    and metadata.json. Returns None if metadata.json is missing.
    """
    meta_path = os.path.join(submission_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    photo1 = os.path.join(submission_dir, "photo1.jpg")
    photo2 = os.path.join(submission_dir, "photo2.jpg")

    # Also check common alternate names
    if not os.path.isfile(photo1):
        for alt in ["photo_1.jpg", "Photo1.jpg", "image1.jpg", "IMG_1.jpg"]:
            alt_path = os.path.join(submission_dir, alt)
            if os.path.isfile(alt_path):
                photo1 = alt_path
                break

    if not os.path.isfile(photo2):
        for alt in ["photo_2.jpg", "Photo2.jpg", "image2.jpg", "IMG_2.jpg"]:
            alt_path = os.path.join(submission_dir, alt)
            if os.path.isfile(alt_path):
                photo2 = alt_path
                break

    return {
        "id": os.path.basename(submission_dir),
        "dir": submission_dir,
        "photo1": photo1,
        "photo2": photo2,
        "width_inches": meta.get("width_inches"),
        "height_inches": meta.get("height_inches"),
        "window_type": meta.get("window_type"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Prolific submissions for training"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing raw submission folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print details for each submission"
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover submission directories
    submission_dirs = sorted(
        [
            os.path.join(input_dir, d)
            for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        ]
    )

    if not submission_dirs:
        print(f"Error: No submission directories found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(submission_dirs)} submission directories")

    labels: list[dict[str, Any]] = []
    total = 0
    valid_count = 0
    rejected_count = 0
    warning_count = 0
    rejection_reasons: dict[str, int] = {}

    for sub_dir in submission_dirs:
        total += 1
        sub = load_submission(sub_dir)

        if sub is None:
            rejected_count += 1
            reason = "missing metadata.json"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            if args.verbose:
                print(f"  SKIP {os.path.basename(sub_dir)}: {reason}")
            continue

        result = validate_submission(sub)

        if result["warnings"]:
            warning_count += len(result["warnings"])

        if not result["valid"]:
            rejected_count += 1
            for err in result["errors"]:
                rejection_reasons[err] = rejection_reasons.get(err, 0) + 1
            if args.verbose:
                print(f"  FAIL {sub['id']}: {'; '.join(result['errors'])}")
            continue

        valid_count += 1

        # Create short submission ID for filenames
        short_id = f"sub{valid_count:04d}"

        # Copy photos to output
        photo1_dest = os.path.join(images_dir, f"{short_id}_photo1.jpg")
        photo2_dest = os.path.join(images_dir, f"{short_id}_photo2.jpg")
        shutil.copy2(sub["photo1"], photo1_dest)
        shutil.copy2(sub["photo2"], photo2_dest)

        # Add label entry
        labels.append({
            "id": short_id,
            "original_id": sub["id"],
            "source": "prolific",
            "photo1": f"{short_id}_photo1.jpg",
            "photo2": f"{short_id}_photo2.jpg",
            "width_inches": float(sub["width_inches"]),
            "height_inches": float(sub["height_inches"]),
            "window_type": sub["window_type"],
            "warnings": result["warnings"] if result["warnings"] else None,
        })

        if args.verbose:
            print(
                f"  PASS {sub['id']} -> {short_id}: "
                f"{sub['width_inches']}x{sub['height_inches']}\" "
                f"({sub['window_type']})"
            )

    # Write labels
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Prolific Data Preparation Summary")
    print(f"{'='*50}")
    print(f"Total submissions:    {total}")
    print(f"Valid (accepted):     {valid_count} ({100*valid_count/total:.1f}%)" if total > 0 else "Valid: 0")
    print(f"Rejected:             {rejected_count} ({100*rejected_count/total:.1f}%)" if total > 0 else "Rejected: 0")
    print(f"Total warnings:       {warning_count}")
    print(f"\nOutput:")
    print(f"  Images: {images_dir}/ ({valid_count * 2} files)")
    print(f"  Labels: {labels_path}")

    if rejection_reasons:
        print(f"\nRejection reasons:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:4d}x  {reason}")

    # Window type distribution
    if labels:
        type_counts: dict[str, int] = {}
        for label in labels:
            wt = label["window_type"]
            type_counts[wt] = type_counts.get(wt, 0) + 1

        print(f"\nWindow type distribution:")
        for wt, count in sorted(type_counts.items()):
            print(f"  {wt}: {count} ({100*count/len(labels):.1f}%)")

    print(f"{'='*50}")


if __name__ == "__main__":
    main()
