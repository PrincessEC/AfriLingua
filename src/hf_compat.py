"""
Small compatibility helpers for Hugging Face / Transformers APIs.
"""
from __future__ import annotations

import torch
import transformers


def dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """
    Return the correct dtype keyword for the installed Transformers version.

    Transformers 5 uses `dtype`; Transformers 4 uses `torch_dtype`.
    """
    try:
        major = int(transformers.__version__.split(".", 1)[0])
    except (AttributeError, ValueError):
        major = 4
    key = "dtype" if major >= 5 else "torch_dtype"
    return {key: dtype}


def dtype_name_kwargs(dtype_name: str) -> dict[str, torch.dtype]:
    """Return dtype kwargs from the app's config string."""
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    return dtype_kwargs(dtype)


def fp32_dtype_kwargs() -> dict[str, torch.dtype]:
    """Return the correct fp32 dtype keyword for Transformers."""
    return dtype_kwargs(torch.float32)
