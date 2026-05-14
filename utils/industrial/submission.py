# Copyright (C) 2025 MVTec Software GmbH
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# Modified by Intel Corporation, 2026
# SPDX-License-Identifier: Apache-2.0

"""Submission validation and packaging utilities for the industrial track."""

import re
import tarfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from industrial.types import Category, Split

EXPECTED_COUNTS: dict[Category, int] = {
    Category.CAN: 321,
    Category.FABRIC: 314,
    Category.FRUIT_JELLY: 255,
    Category.RICE: 277,
    Category.SHEET_METAL: 142,
    Category.VIAL: 276,
    Category.WALLPLUGS: 232,
    Category.WALNUTS: 228,
}

SUBMISSION_SPLITS: tuple[Split, Split] = (Split.TEST_PRIVATE, Split.TEST_PRIVATE_MIXED)


def _validate_names(paths: list[Path], split: Split, category: Category) -> None:
    """Validate file naming pattern and index coverage for a split."""

    pattern = re.compile(r"^(\d{3})_(regular|mixed)$")
    seen: set[int] = set()
    for path in paths:
        name = path.stem
        match = pattern.match(name)
        if match is None:
            raise ValueError(
                f"Invalid file name '{path.name}' in {category}/{split}; expected ###_regular or ###_mixed"
            )
        idx = int(match.group(1))
        suffix = match.group(2)
        expected_suffix = "regular" if split == Split.TEST_PRIVATE else "mixed"
        if suffix != expected_suffix:
            raise ValueError(
                f"Invalid suffix in '{path.name}' for {category}/{split}; expected _{expected_suffix}"
            )
        seen.add(idx)
    expected_count = EXPECTED_COUNTS[category]
    expected_indices = set(range(0, expected_count))
    if seen != expected_indices:
        missing = sorted(expected_indices - seen)
        extra = sorted(seen - expected_indices)
        raise ValueError(
            f"Unexpected index set for {category}/{split}; missing={missing[:5]} extra={extra[:5]}"
        )


def _validate_float16_tiffs(
    paths: list[Path], category: Category, split: Split
) -> None:
    """Validate TIFF files are single-channel 2D float16 images."""

    for path in paths:
        arr = np.asarray(tifffile.imread(path))
        if arr.ndim != 2:
            raise ValueError(
                f"TIFF must be single-channel 2D in {category}/{split}: {path.name}"
            )
        if arr.dtype != np.float16:
            raise ValueError(
                f"TIFF must be float16 in {category}/{split}: {path.name}, got {arr.dtype}"
            )


def _validate_binary_pngs(paths: list[Path], category: Category, split: Split) -> None:
    """Validate PNG files are single-channel images with values {0, 255}."""

    for path in paths:
        arr = np.asarray(Image.open(path))
        if arr.ndim != 2:
            raise ValueError(
                f"PNG must be single-channel 2D in {category}/{split}: {path.name}"
            )
        unique = np.unique(arr)
        allowed = {0, 255}
        if not set(int(v) for v in unique).issubset(allowed):
            raise ValueError(
                f"PNG must only contain values 0 or 255 in {category}/{split}: {path.name}"
            )


def validate_submission(submission_dir: str | Path) -> bool:
    """Validate prediction directory structure and file formats.

    Args:
        submission_dir (str | Path): Root directory containing prediction outputs.

    Returns:
        bool: True when the submission directory passes all checks.

    Raises:
        FileNotFoundError: If required directories are missing.
        ValueError: If file counts, names, or formats are invalid.
    """

    root = Path(submission_dir)
    if not root.exists():
        raise FileNotFoundError(f"Submission directory not found: {root}")

    anomaly_root = root / "anomaly_images"
    threshold_root = root / "anomaly_images_thresholded"
    if not anomaly_root.exists():
        raise FileNotFoundError(f"Missing directory: {anomaly_root}")
    if not threshold_root.exists():
        raise FileNotFoundError(f"Missing directory: {threshold_root}")

    for category in Category:
        expected_count = EXPECTED_COUNTS[category]
        for split in SUBMISSION_SPLITS:
            anomaly_dir = anomaly_root / category / split
            threshold_dir = threshold_root / category / split
            if not anomaly_dir.exists():
                raise FileNotFoundError(f"Missing directory: {anomaly_dir}")
            if not threshold_dir.exists():
                raise FileNotFoundError(f"Missing directory: {threshold_dir}")

            tiff_paths = sorted(
                path for path in anomaly_dir.glob("*.tiff") if path.is_file()
            )
            png_paths = sorted(
                path for path in threshold_dir.glob("*.png") if path.is_file()
            )

            if len(tiff_paths) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} TIFF files in {anomaly_dir}, found {len(tiff_paths)}"
                )
            if len(png_paths) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} PNG files in {threshold_dir}, found {len(png_paths)}"
                )

            _validate_names(tiff_paths, split, category)
            _validate_names(png_paths, split, category)
            _validate_float16_tiffs(tiff_paths, category, split)
            _validate_binary_pngs(png_paths, category, split)

    return True


def prepare_submission(
    submission_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Validate and package a submission directory as a tar.gz archive.

    Args:
        submission_dir (str | Path): Root directory containing prediction outputs.
        output_path (str | Path | None): Optional target archive path.

    Returns:
        Path: Path to the generated archive.

    Raises:
        FileNotFoundError: If required directories are missing.
        ValueError: If file counts, names, or formats are invalid.
    """

    root = Path(submission_dir)
    validate_submission(root)
    archive_path = (
        Path(output_path) if output_path is not None else root.with_suffix(".tar.gz")
    )
    if archive_path.suffixes[-2:] != [".tar", ".gz"]:
        archive_path = archive_path.with_suffix(".tar.gz")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode="w:gz") as tar:
        tar.add(root, arcname=root.name)

    return archive_path
