"""
Small compatibility helpers for Hugging Face / Transformers APIs.
"""
from __future__ import annotations

import torch
import transformers


def fp32_dtype_kwargs() -> dict[str, torch.dtype]:
    """
    Return the correct fp32 dtype keyword for the installed Transformers version.

    Transformers 5 uses `dtype`; Transformers 4 uses `torch_dtype`.
    """
    try:
        major = int(transformers.__version__.split(".", 1)[0])
    except (AttributeError, ValueError):
        major = 4
    key = "dtype" if major >= 5 else "torch_dtype"
    return {key: torch.float32}
