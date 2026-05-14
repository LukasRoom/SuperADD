# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Enums and data containers for the industrial track."""

from dataclasses import dataclass
from enum import StrEnum

import torch


class Category(StrEnum):
    """MVTec AD 2 product categories."""

    CAN = "can"
    FABRIC = "fabric"
    FRUIT_JELLY = "fruit_jelly"
    RICE = "rice"
    SHEET_METAL = "sheet_metal"
    VIAL = "vial"
    WALLPLUGS = "wallplugs"
    WALNUTS = "walnuts"


class Split(StrEnum):
    """Dataset split identifiers."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST_PUBLIC = "test_public"
    TEST_PRIVATE = "test_private"
    TEST_PRIVATE_MIXED = "test_private_mixed"


@dataclass
class IndustrialSample:
    """Single sample returned by :meth:`MVTecAD2Dataset.__getitem__`."""

    image: torch.Tensor
    image_path: str
    label: int
    mask: torch.Tensor | None
    category: str
    split: str


@dataclass
class IndustrialBatch:
    """Collated mini-batch of industrial samples."""

    image: torch.Tensor
    image_path: list[str]
    label: torch.Tensor
    mask: torch.Tensor | list[torch.Tensor | None]
    category: list[str]
    split: list[str]


@dataclass
class EvaluationResult:
    """Result of local industrial track evaluation."""

    per_category: dict[str, float]
    mean_seg_f1: float
    per_image: dict[str, dict[str, float]]
