# Copyright (C) 2025 MVTec Software GmbH
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# Modified by Intel Corporation, 2026
# SPDX-License-Identifier: Apache-2.0

"""Local evaluation helpers for industrial track segmentation results.

Note:
    The metric utilities in this module are provided for convenience and local
    testing only. Official scores are computed by the respective challenge
    servers, whose implementations may differ. No claims or entitlements can be
    derived from the local evaluation results.
"""

from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image
from sklearn.metrics import precision_recall_curve

from industrial.types import Category, EvaluationResult


def compute_seg_f1(anomaly_map: Any, gt_mask: Any) -> float:
    """Compute best F1 score from pixelwise precision-recall curve.

    Args:
        anomaly_map (Any): Predicted anomaly score map.
        gt_mask (Any): Ground-truth binary mask.

    Returns:
        float: Maximum F1 value over all thresholds.
    """

    anomaly = anomaly_map.astype(np.float32).reshape(-1)
    gt = (gt_mask > 0).astype(np.uint8).reshape(-1)
    if gt.sum() == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(gt, anomaly)
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)
    return float(np.max(f1)) if f1.size > 0 else 0.0


def _resolve_prediction_path(predictions_dir: Path, category: str, stem: str) -> Path:
    """Resolve prediction TIFF path across allowed directory layouts."""

    candidates = [
        predictions_dir
        / "anomaly_images"
        / category
        / "test_public"
        / "bad"
        / f"{stem}.tiff",
        predictions_dir / "anomaly_images" / category / "test_public" / f"{stem}.tiff",
        predictions_dir / category / "test_public" / "bad" / f"{stem}.tiff",
        predictions_dir / category / "test_public" / f"{stem}.tiff",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing anomaly map for '{category}/{stem}'. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _print_summary(category_scores: dict[str, float], mean_seg_f1: float) -> None:
    """Print an ASCII summary table of category and mean SegF1."""

    category_width = max(
        len("Category"), *(len(category) for category in category_scores)
    )
    score_width = len("SegF1")
    border = f"+-{'-' * category_width}-+-{'-' * score_width}-+"
    print(border)
    print(f"| {'Category'.ljust(category_width)} | {'SegF1'.rjust(score_width)} |")
    print(border)
    for category in sorted(category_scores.keys()):
        score_text = f"{category_scores[category]:.4f}"
        print(f"| {category.ljust(category_width)} | {score_text.rjust(score_width)} |")
    print(border)
    mean_text = f"{mean_seg_f1:.4f}"
    print(f"| {'MEAN'.ljust(category_width)} | {mean_text.rjust(score_width)} |")
    print(border)


def evaluate_local(
    predictions_dir: str | Path,
    dataset_root: str | Path,
    categories: list[Category | str] | None = None,
) -> EvaluationResult:
    """Evaluate public test masks using predicted anomaly maps.

    Args:
        predictions_dir (str | Path): Directory containing prediction TIFF files.
        dataset_root (str | Path): MVTec AD 2 dataset root directory.
        categories (list[Category | str] | None): Optional category subset. Defaults to all.

    Returns:
        EvaluationResult: Per-category, per-image, and mean segmentation F1 scores.

    Raises:
        ValueError: If an unknown category is requested or no valid samples are found.
        FileNotFoundError: If required prediction or ground-truth files are missing.
    """

    pred_root = Path(predictions_dir)
    data_root = Path(dataset_root)
    selected = (
        [Category(c) for c in categories] if categories is not None else list(Category)
    )

    per_category: dict[str, float] = {}
    per_image: dict[str, dict[str, float]] = {}

    for category in selected:
        gt_dir = data_root / category / "test_public" / "ground_truth" / "bad"
        if not gt_dir.exists():
            raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
        gt_paths = sorted(gt_dir.glob("*_mask.png"))
        if len(gt_paths) == 0:
            raise FileNotFoundError(f"No mask files found in {gt_dir}")

        image_scores: dict[str, float] = {}
        for mask_path in gt_paths:
            stem = mask_path.stem
            if not stem.endswith("_mask"):
                continue
            image_stem = stem[: -len("_mask")]
            pred_path = _resolve_prediction_path(pred_root, category, image_stem)
            anomaly_map = np.asarray(tifffile.imread(pred_path), dtype=np.float32)
            gt_mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            if anomaly_map.shape != gt_mask.shape:
                raise ValueError(
                    f"Shape mismatch for '{category}/{image_stem}': "
                    f"prediction {anomaly_map.shape} vs GT {gt_mask.shape}"
                )
            f1 = compute_seg_f1(anomaly_map=anomaly_map, gt_mask=gt_mask)
            image_scores[image_stem] = f1

        if len(image_scores) == 0:
            raise ValueError(
                f"No evaluable public bad samples found for category '{category}'"
            )

        score = float(np.mean(np.fromiter(image_scores.values(), dtype=np.float32)))
        per_category[category] = score
        per_image[category] = image_scores

    mean_seg_f1 = float(np.mean(np.fromiter(per_category.values(), dtype=np.float32)))
    _print_summary(per_category, mean_seg_f1)
    return EvaluationResult(
        per_category=per_category,
        mean_seg_f1=mean_seg_f1,
        per_image=per_image,
    )
