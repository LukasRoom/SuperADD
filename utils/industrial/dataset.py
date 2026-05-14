# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset and dataloader utilities for the industrial track."""

# pyright: reportMissingImports=false

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from industrial.types import Category, IndustrialBatch, IndustrialSample, Split


@dataclass(frozen=True)
class _Sample:
    """Single dataset sample metadata."""

    image_path: Path
    label: int
    mask_path: Path | None


class MVTecAD2Dataset(Dataset):
    """PyTorch dataset for MVTec AD 2 industrial track data.

    Args:
        root (str | Path): Root directory containing category subdirectories.
        category (Category | str): Category to read.
        split (Split | str): Dataset split to load.
        transform (Any | None): Optional image transform pipeline.

    Raises:
        ValueError: If category or split is invalid.
        FileNotFoundError: If expected directories or PNG files are missing.
    """

    def __init__(
        self,
        root: str | Path,
        category: Category | str,
        split: Split | str,
        transform: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.category = Category(category)
        self.split = Split(split)
        if transform is None:
            self.transform = transforms.Compose(
                [transforms.Resize((256, 256)), transforms.ToTensor()]
            )
        else:
            self.transform = transform

        self.category_dir = self.root / self.category
        if not self.category_dir.exists():
            raise FileNotFoundError(f"Missing category directory: {self.category_dir}")

        self.samples = self._build_samples()
        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No PNG files found for category='{self.category}' split='{self.split}'"
            )

    def _glob_pngs(self, directory: Path) -> list[Path]:
        """Return sorted PNG file paths in a directory."""

        if not directory.exists():
            return []
        return sorted(p for p in directory.glob("*.png") if p.is_file())

    def _build_samples(self) -> list[_Sample]:
        """Build sample metadata list for the selected split."""

        split_dir = self.category_dir / self.split

        if self.split in {Split.TRAIN, Split.VALIDATION}:
            return [
                _Sample(image_path=path, label=0, mask_path=None)
                for path in self._glob_pngs(split_dir / "good")
            ]

        if self.split == Split.TEST_PUBLIC:
            good_paths = self._glob_pngs(split_dir / "good")
            bad_paths = self._glob_pngs(split_dir / "bad")
            gt_bad_dir = split_dir / "ground_truth" / "bad"
            samples: list[_Sample] = []
            for path in good_paths:
                samples.append(_Sample(image_path=path, label=0, mask_path=None))
            for path in bad_paths:
                maybe_mask = gt_bad_dir / f"{path.stem}_mask.png"
                samples.append(
                    _Sample(
                        image_path=path,
                        label=1,
                        mask_path=maybe_mask if maybe_mask.exists() else None,
                    )
                )
            return samples

        return [
            _Sample(image_path=path, label=-1, mask_path=None)
            for path in self._glob_pngs(split_dir)
        ]

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""

        return len(self.samples)

    def _load_image(self, path: Path) -> Any:
        """Load and transform an RGB image."""

        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image)
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("Transform must return torch.Tensor")
        return image_tensor

    def _load_mask(self, path: Path | None, h: int, w: int) -> Any | None:
        """Load and resize a binary mask if available."""

        if path is None or not path.exists():
            return None
        mask = Image.open(path).convert("L")
        resize = transforms.Resize(
            (h, w), interpolation=transforms.InterpolationMode.NEAREST
        )
        mask = resize(mask)
        mask_tensor = transforms.ToTensor()(mask)
        return (mask_tensor > 0.5).to(torch.float32)

    def __getitem__(self, index: int) -> IndustrialSample:
        """Get one sample.

        Args:
            index (int): Sample index.

        Returns:
            IndustrialSample: Sample containing image tensor, metadata, and optional mask.
        """

        sample = self.samples[index]
        image = self._load_image(sample.image_path)
        _, h, w = image.shape
        mask = self._load_mask(sample.mask_path, h, w)
        return IndustrialSample(
            image=image,
            image_path=str(sample.image_path),
            label=sample.label,
            mask=mask,
            category=self.category,
            split=self.split,
        )


def _collate_batch(items: list[IndustrialSample]) -> IndustrialBatch:
    """Collate dataset items into a mini-batch."""

    images = torch.stack([item.image for item in items], dim=0)
    labels = torch.tensor([item.label for item in items], dtype=torch.int64)
    masks_raw = [item.mask for item in items]
    if all(m is not None for m in masks_raw):
        mask: torch.Tensor | list[torch.Tensor | None] = torch.stack(
            [m for m in masks_raw if m is not None], dim=0
        )
    else:
        mask = masks_raw
    return IndustrialBatch(
        image=images,
        image_path=[item.image_path for item in items],
        label=labels,
        mask=mask,
        category=[item.category for item in items],
        split=[item.split for item in items],
    )


def get_dataloader(
    root: str | Path,
    category: Category | str,
    split: Split | str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Any | None = None,
) -> DataLoader:
    """Create a dataloader for one category and split.

    Args:
        root (str | Path): Root directory containing category subdirectories.
        category (Category | str): Category to load.
        split (Split | str): Dataset split to load.
        batch_size (int): Number of samples per batch. Defaults to 32.
        num_workers (int): Worker process count. Defaults to 4.
        transform (Any | None): Optional image transform pipeline.

    Returns:
        DataLoader: Configured dataloader with custom collation.
    """

    split = Split(split)
    dataset = MVTecAD2Dataset(
        root=root, category=category, split=split, transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == Split.TRAIN,
        num_workers=num_workers,
        pin_memory=bool(torch.cuda.is_available()),
        collate_fn=_collate_batch,
    )
