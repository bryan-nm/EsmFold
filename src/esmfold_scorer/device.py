"""Device detection with automatic XPU / CUDA / CPU fallback.

Priority order:
  1. Intel XPU  (Aurora, Ponte Vecchio) — requires ``intel_extension_for_pytorch``
  2. NVIDIA CUDA
  3. CPU

Importing ``intel_extension_for_pytorch`` registers the ``xpu`` backend with
PyTorch, so we import it eagerly when available.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

_ipex_available: bool = False
try:
    import intel_extension_for_pytorch  # noqa: F401

    _ipex_available = True
except ImportError:
    pass


def resolve_device(requested: str | None = None) -> torch.device:
    """Return a ``torch.device`` matching *requested*, or auto-detect one.

    Parameters
    ----------
    requested:
        ``"xpu"``, ``"cuda"``, ``"cpu"``, or ``None`` for auto-detection.

    Returns
    -------
    torch.device

    Raises
    ------
    RuntimeError
        If the requested backend is unavailable.
    """
    if requested is not None:
        requested = requested.lower()
        if requested == "xpu":
            if not _ipex_available or not torch.xpu.is_available():
                raise RuntimeError(
                    "XPU requested but intel_extension_for_pytorch is not "
                    "installed or no XPU device is visible."
                )
            log.info("Using XPU device")
            return torch.device("xpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but no GPU is visible.")
            log.info("Using CUDA device")
            return torch.device("cuda")
        if requested == "cpu":
            log.info("Using CPU device")
            return torch.device("cpu")
        raise ValueError(f"Unknown device: {requested!r}")

    if _ipex_available and torch.xpu.is_available():
        log.info("Auto-detected XPU device")
        return torch.device("xpu")
    if torch.cuda.is_available():
        log.info("Auto-detected CUDA device")
        return torch.device("cuda")

    log.info("No accelerator found; falling back to CPU")
    return torch.device("cpu")


def optimal_dtype(device: torch.device) -> torch.dtype:
    """Return the fastest safe dtype for *device*.

    ``bfloat16`` is preferred on XPU (native to Intel PVC / Aurora) and on
    CUDA (Ampere+).  Falls back to ``float32`` on CPU.
    """
    if device.type in ("xpu", "cuda"):
        return torch.bfloat16
    return torch.float32
