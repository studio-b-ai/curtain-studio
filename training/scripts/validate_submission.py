"""
Validate a Prolific window measurement submission.

Checks image quality, EXIF data, measurement plausibility, and window type.
Designed to be imported by prepare_prolific.py or run standalone against
a submissions.json file.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import exifread
except ImportError:
    exifread = None  # type: ignore[assignment]

VALID_WINDOW_TYPES = [
    "single_hung",
    "double_hung",
    "casement",
    "picture",
    "sliding",
    "bay_bow",
    "arched",
    "divided_light",
    "other",
]

MIN_WIDTH_INCHES = 12
MAX_WIDTH_INCHES = 120
MIN_HEIGHT_INCHES = 12
MAX_HEIGHT_INCHES = 96
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 4.0
MIN_IMAGE_WIDTH = 640
MIN_IMAGE_HEIGHT = 480


def validate_image(image_path: str) -> tuple[list[str], list[str]]:
    """Validate a single image file. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not os.path.isfile(image_path):
        errors.append(f"Image file not found: {image_path}")
        return errors, warnings

    # Check file size (not empty, not suspiciously small)
    file_size = os.path.getsize(image_path)
    if file_size < 10_000:
        errors.append(f"Image file too small ({file_size} bytes): {image_path}")
        return errors, warnings

    # Validate with Pillow
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        errors.append(f"Invalid image file: {image_path} ({e})")
        return errors, warnings

    # Re-open after verify (verify closes the file)
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        errors.append(f"Cannot read image dimensions: {image_path} ({e})")
        return errors, warnings

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        errors.append(
            f"Image too small ({width}x{height}), "
            f"minimum {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}: {image_path}"
        )

    # Check EXIF
    if exifread is not None:
        try:
            with open(image_path, "rb") as f:
                tags = exifread.process_file(f, details=False)

            if not tags:
                warnings.append(f"No EXIF data found: {image_path}")
            elif "EXIF FocalLength" not in tags:
                warnings.append(f"No FocalLength in EXIF: {image_path}")
        except Exception:
            warnings.append(f"Could not read EXIF data: {image_path}")
    else:
        warnings.append("exifread not installed, skipping EXIF validation")

    return errors, warnings


def validate_measurements(
    width_inches: float, height_inches: float
) -> tuple[list[str], list[str]]:
    """Validate window measurements. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    # Width range
    if width_inches < MIN_WIDTH_INCHES or width_inches > MAX_WIDTH_INCHES:
        errors.append(
            f"Width {width_inches}\" outside plausible range "
            f"({MIN_WIDTH_INCHES}-{MAX_WIDTH_INCHES}\")"
        )

    # Height range
    if height_inches < MIN_HEIGHT_INCHES or height_inches > MAX_HEIGHT_INCHES:
        errors.append(
            f"Height {height_inches}\" outside plausible range "
            f"({MIN_HEIGHT_INCHES}-{MAX_HEIGHT_INCHES}\")"
        )

    # Aspect ratio
    if width_inches > 0 and height_inches > 0:
        aspect_ratio = width_inches / height_inches
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            errors.append(
                f"Aspect ratio {aspect_ratio:.2f} outside plausible range "
                f"({MIN_ASPECT_RATIO}-{MAX_ASPECT_RATIO})"
            )

    # Warn on suspiciously round numbers (possible estimation)
    if width_inches == int(width_inches) and height_inches == int(height_inches):
        warnings.append(
            "Both measurements are whole numbers - may be estimated rather than measured"
        )

    return errors, warnings


def validate_submission(submission: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a complete Prolific submission.

    Expected submission dict keys:
        - photo1: str (path to first photo)
        - photo2: str (path to second photo)
        - width_inches: float
        - height_inches: float
        - window_type: str

    Returns dict with:
        - valid: bool
        - errors: list[str]
        - warnings: list[str]
    """
    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Validate photo1
    photo1 = submission.get("photo1")
    if not photo1:
        all_errors.append("Missing photo1 path")
    else:
        errs, warns = validate_image(str(photo1))
        all_errors.extend(errs)
        all_warnings.extend(warns)

    # Validate photo2
    photo2 = submission.get("photo2")
    if not photo2:
        all_errors.append("Missing photo2 path")
    else:
        errs, warns = validate_image(str(photo2))
        all_errors.extend(errs)
        all_warnings.extend(warns)

    # Validate measurements
    try:
        width = float(submission.get("width_inches", 0))
        height = float(submission.get("height_inches", 0))
    except (TypeError, ValueError):
        all_errors.append("Invalid measurement values (must be numeric)")
        width, height = 0, 0

    if width > 0 and height > 0:
        errs, warns = validate_measurements(width, height)
        all_errors.extend(errs)
        all_warnings.extend(warns)
    elif width <= 0 or height <= 0:
        all_errors.append("Measurements must be positive numbers")

    # Validate window type
    window_type = submission.get("window_type", "")
    if not window_type:
        all_errors.append("Missing window_type")
    elif window_type not in VALID_WINDOW_TYPES:
        all_errors.append(
            f"Invalid window_type '{window_type}'. "
            f"Must be one of: {', '.join(VALID_WINDOW_TYPES)}"
        )

    return {
        "valid": len(all_errors) == 0,
        "errors": all_errors,
        "warnings": all_warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Prolific window measurement submissions"
    )
    parser.add_argument(
        "submissions_file",
        nargs="?",
        default="submissions.json",
        help="Path to submissions JSON file (default: submissions.json)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print details for each submission"
    )
    args = parser.parse_args()

    submissions_path = Path(args.submissions_file)
    if not submissions_path.is_file():
        print(f"Error: File not found: {submissions_path}", file=sys.stderr)
        sys.exit(1)

    with open(submissions_path) as f:
        submissions = json.load(f)

    if not isinstance(submissions, list):
        print("Error: submissions.json must contain a JSON array", file=sys.stderr)
        sys.exit(1)

    total = len(submissions)
    passed = 0
    failed = 0
    warning_count = 0

    for i, sub in enumerate(submissions):
        result = validate_submission(sub)
        sub_id = sub.get("id", f"submission_{i}")

        if result["valid"]:
            passed += 1
            if args.verbose:
                print(f"  PASS: {sub_id}")
        else:
            failed += 1
            if args.verbose:
                print(f"  FAIL: {sub_id}")
                for err in result["errors"]:
                    print(f"    ERROR: {err}")

        if result["warnings"]:
            warning_count += len(result["warnings"])
            if args.verbose:
                for warn in result["warnings"]:
                    print(f"    WARN: {warn}")

    print(f"\n{'='*40}")
    print(f"Total submissions: {total}")
    print(f"  Passed: {passed} ({100*passed/total:.1f}%)" if total > 0 else "  Passed: 0")
    print(f"  Failed: {failed} ({100*failed/total:.1f}%)" if total > 0 else "  Failed: 0")
    print(f"  Warnings: {warning_count}")
    print(f"{'='*40}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
