# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Auto batch size decorator"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("vand")

F = TypeVar("F", bound=Callable[..., Any])


def auto_batch_size(
    max_batch_size: int = 128,
    min_batch_size: int = 1,
) -> Callable[[F], F]:
    """Decorator that automatically finds a batch size that fits in GPU memory.

    Tries ``max_batch_size`` first, then halves on CUDA OOM until
    ``min_batch_size``. The decorated function receives ``batch_size``
    as a keyword argument.

    Returns:
        Tuple of (function_result, used_batch_size).
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, batch_size: int | None = None, **kwargs: Any) -> Any:
            if batch_size is not None:
                return fn(*args, batch_size=batch_size, **kwargs), batch_size

            bs = max_batch_size
            while bs >= min_batch_size:
                try:
                    result = fn(*args, batch_size=bs, **kwargs)
                    return result, bs
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "OOM with batch_size=%d, retrying with %d", bs, bs // 2
                    )
                    torch.cuda.empty_cache()
                    bs //= 2

            raise RuntimeError(
                f"Could not fit even batch_size={min_batch_size} in GPU memory."
            )

        return wrapper  # type: ignore[return-value]

    return decorator
