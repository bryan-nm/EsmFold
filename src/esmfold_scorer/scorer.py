"""Core scoring logic: load ESMFold2-Fast once, score sequences repeatedly.

The public API is :class:`StructureScorer`.  Typical integration into a
generative-model evaluation loop::

    scorer = StructureScorer(
        model_path="/models/ESMFold2-Fast",
        device="xpu",              # or "cuda", "cpu", None for auto
        num_sampling_steps=10,     # fewer steps = faster, good for ranking
        num_loops=1,               # fewer recycling loops = faster
    )

    # inside your eval callback:
    results = scorer.score(generated_sequences)
    log_metric("mean_plddt", results.mean_plddt)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field

import torch

from esmfold_scorer.device import optimal_dtype, resolve_device

log = logging.getLogger(__name__)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class ScoringResult:
    """Container for pLDDT scoring results.

    Attributes
    ----------
    mean_plddt:
        Scalar mean pLDDT across all sequences (0–100 scale).
    per_sequence_plddt:
        Mean pLDDT for each input sequence, in order.
    per_sequence_ptm:
        pTM score for each input sequence, in order.
    elapsed_seconds:
        Wall-clock time for the scoring call.
    num_sequences:
        Number of sequences scored.
    """

    mean_plddt: float
    per_sequence_plddt: list[float]
    per_sequence_ptm: list[float]
    elapsed_seconds: float
    num_sequences: int
    per_sequence_lengths: list[int] = field(default_factory=list)


class StructureScorer:
    """Score protein sequences by predicted structural quality (mean pLDDT).

    Wraps ESMFold2-Fast for fast, batched pLDDT evaluation.  Intended to be
    instantiated once and reused across many scoring calls.

    Parameters
    ----------
    model_path:
        Path to local ESMFold2-Fast weights directory (containing
        ``config.json`` and ``model.safetensors``), *or* a Hugging Face
        Hub identifier like ``"biohub/ESMFold2-Fast"``.
    device:
        ``"xpu"``, ``"cuda"``, ``"cpu"``, or ``None`` to auto-detect.
        Auto-detection prefers XPU > CUDA > CPU.
    dtype:
        Inference dtype.  ``None`` selects ``bfloat16`` on accelerators,
        ``float32`` on CPU.  On XPU, the model weights are converted to
        this dtype and ``F.linear`` / ``F.layer_norm`` are patched to
        handle float32 inputs from the esm package's feature construction.
    esmc_model_id:
        HuggingFace Hub repo ID for the ESM-C backbone (e.g.
        ``"biohub/ESMC-6B"``).  If ``None``, uses the default from the
        ESMFold2-Fast config.  The backbone is resolved from the local
        HuggingFace cache (``~/.cache/huggingface/hub/``) when
        ``HF_HUB_OFFLINE=1`` is set — no network access needed.
    num_sampling_steps:
        Diffusion sampling steps per structure prediction.  Default **10**
        is ~5x faster than the paper default (50) and sufficient for
        *relative* pLDDT ranking during training.  Increase for higher
        absolute accuracy.
    num_loops:
        Folding-trunk recycling iterations.  Default **1** is fastest.
        Paper default is 3.
    compile_model:
        If ``True``, run ``torch.compile()`` on the model after loading.
        Adds ~30–60 s startup cost but can improve throughput for many
        short sequences.  Requires PyTorch ≥ 2.1.
    """

    def __init__(
        self,
        model_path: str = "biohub/ESMFold2-Fast",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        esmc_model_id: str | None = None,
        num_sampling_steps: int = 10,
        num_loops: int = 1,
        compile_model: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.dtype = dtype or optimal_dtype(self.device)
        self.num_sampling_steps = num_sampling_steps
        self.num_loops = num_loops

        log.info(
            "Loading ESMFold2-Fast from %s (device=%s, dtype=%s)",
            model_path,
            self.device,
            self.dtype,
        )
        t0 = time.monotonic()
        self._model = self._load_model(model_path, esmc_model_id, compile_model)
        log.info("Model loaded in %.1f s", time.monotonic() - t0)

    def _load_model(
        self,
        model_path: str,
        esmc_model_id: str | None,
        compile_model: bool,
    ) -> torch.nn.Module:
        model_cls = self._resolve_model_class()
        load_path = model_path

        if esmc_model_id is not None:
            log.info("Overriding ESM-C backbone to %s", esmc_model_id)
            load_path = self._patch_config(model_path, esmc_model_id)

        model = model_cls.from_pretrained(load_path)
        model = model.to(device=self.device, dtype=self.dtype).eval()

        if self.device.type == "xpu":
            _patch_linalg_cpu_roundtrip()
            if self.dtype != torch.float32:
                _patch_dtype_casting()

        if compile_model:
            log.info("Compiling model with torch.compile (this takes a while)...")
            model = torch.compile(model)

        return model

    @staticmethod
    def _patch_config(model_path: str, esmc_model_id: str) -> str:
        """Create a temp directory with a modified config.json overriding esmc_id."""
        config_file = os.path.join(model_path, "config.json")
        with open(config_file) as f:
            config_dict = json.load(f)
        config_dict["esmc_id"] = esmc_model_id

        tmp_dir = tempfile.mkdtemp(prefix="esmfold_scorer_")
        for item in os.listdir(model_path):
            src = os.path.join(model_path, item)
            if os.path.isfile(src) and item != "config.json":
                os.symlink(src, os.path.join(tmp_dir, item))
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(config_dict, f)
        return tmp_dir

    @staticmethod
    def _resolve_model_class() -> type:
        # 1. Try the esm package (provides EsmFold2Model directly)
        try:
            from esm.models.esmfold2 import EsmFold2Model
            log.info("Using EsmFold2Model from esm package")
            return EsmFold2Model
        except (ImportError, ModuleNotFoundError):
            pass

        # 2. Try transformers (may have it built-in in future versions)
        try:
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
            log.info("Using ESMFold2Model from transformers")
            return ESMFold2Model
        except (ImportError, ModuleNotFoundError):
            pass

        raise ImportError(
            "Cannot find ESMFold2 model class. Install:\n"
            "  pip install 'esm @ git+https://github.com/Biohub/esm.git@main'"
        )

    @torch.inference_mode()
    def score(
        self,
        sequences: list[str] | str,
        *,
        num_sampling_steps: int | None = None,
        num_loops: int | None = None,
    ) -> ScoringResult:
        """Score one or more protein sequences and return pLDDT results.

        Parameters
        ----------
        sequences:
            A single amino-acid string or a list of them.  Non-standard
            residues are replaced with ``X``.
        num_sampling_steps:
            Override the instance default for this call only.
        num_loops:
            Override the instance default for this call only.

        Returns
        -------
        ScoringResult
            Contains ``mean_plddt`` (scalar), ``per_sequence_plddt``
            (list), ``per_sequence_ptm`` (list), and timing info.

        Raises
        ------
        ValueError
            If *sequences* is empty or contains only whitespace strings.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        if not sequences:
            raise ValueError("sequences must be a non-empty list")

        steps = num_sampling_steps if num_sampling_steps is not None else self.num_sampling_steps
        loops = num_loops if num_loops is not None else self.num_loops

        sequences = [self._sanitize(s) for s in sequences]
        if any(len(s) == 0 for s in sequences):
            raise ValueError("All sequences must be non-empty after sanitization")

        t0 = time.monotonic()
        per_seq_plddt: list[float] = []
        per_seq_ptm: list[float] = []
        lengths: list[int] = []

        for seq in sequences:
            output = self._model.infer_protein(
                seq,
                num_loops=loops,
                num_sampling_steps=steps,
            )
            per_seq_plddt.append(float(output["plddt"].mean()))
            per_seq_ptm.append(float(output["ptm"].mean()))
            lengths.append(len(seq))

        elapsed = time.monotonic() - t0
        mean_plddt = sum(per_seq_plddt) / len(per_seq_plddt)

        log.info(
            "Scored %d sequence(s) in %.2f s — mean pLDDT: %.2f",
            len(sequences),
            elapsed,
            mean_plddt,
        )
        return ScoringResult(
            mean_plddt=mean_plddt,
            per_sequence_plddt=per_seq_plddt,
            per_sequence_ptm=per_seq_ptm,
            elapsed_seconds=elapsed,
            num_sequences=len(sequences),
            per_sequence_lengths=lengths,
        )

    @staticmethod
    def _sanitize(seq: str) -> str:
        """Normalize a sequence: uppercase, strip whitespace, replace unknowns."""
        seq = "".join(seq.upper().split())
        return "".join(c if c in VALID_AA else "X" for c in seq)


def _patch_linalg_cpu_roundtrip() -> None:
    """Route SVD/det through an explicit CPU round-trip on XPU.

    ``torch.linalg.svd`` has no XPU kernel, so PyTorch's aten dispatcher
    silently falls back to CPU ("Aten Op fallback from XPU to CPU").  That
    automatic fallback corrupts GPU memory on Aurora's compute-runtime —
    inference aborts with a GPU page fault ("Segmentation fault from GPU",
    varying page-table levels) immediately after the fallback warning.

    Doing the device transfer ourselves — copy to CPU, compute in float32,
    copy the result back — keeps the dispatcher out of the fallback path
    and avoids the fault.  The tensors involved are per-residue 3x3
    rotation matrices, so the round-trip cost is negligible.
    """
    if getattr(torch.linalg.svd, "_xpu_patched", False):
        return

    _orig_svd = torch.linalg.svd

    def _svd_cpu(A, full_matrices=True, **kwargs):
        if A.is_xpu:
            U, S, Vh = _orig_svd(
                A.detach().to("cpu", torch.float32), full_matrices=full_matrices
            )
            return (
                U.to(A.device, A.dtype),
                S.to(A.device, A.dtype),
                Vh.to(A.device, A.dtype),
            )
        return _orig_svd(A, full_matrices=full_matrices, **kwargs)

    _svd_cpu._xpu_patched = True
    torch.linalg.svd = _svd_cpu

    _orig_det = torch.linalg.det

    def _det_cpu(A):
        if A.is_xpu:
            return _orig_det(A.detach().to("cpu", torch.float32)).to(A.device, A.dtype)
        return _orig_det(A)

    _det_cpu._xpu_patched = True
    torch.linalg.det = _det_cpu

    log.info("Patched torch.linalg.svd/det for XPU (explicit CPU round-trip)")


def _patch_dtype_casting() -> None:
    """Patch F.linear and F.layer_norm to cast inputs to match weight dtype.

    The esm package's infer_protein creates float32 feature tensors
    internally, but the model weights are bf16. On CUDA the esm package
    uses torch.autocast to handle this; on XPU that autocast is disabled
    (hardcoded to device_type="cuda"). These patches provide the same
    dtype-matching behaviour without autocast, which avoids promoting
    linalg ops (SVD, det) to bf16 — those ops lack bf16 XPU kernels and
    crash the GPU driver when autocast forces them into bf16.
    """
    import torch.nn.functional as F

    if getattr(F, "_xpu_dtype_patched", False):
        return

    _orig_linear = F.linear

    def _linear_match_dtype(input, weight, bias=None):
        if input.is_floating_point() and input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        if bias is not None and bias.dtype != weight.dtype:
            bias = bias.to(weight.dtype)
        return _orig_linear(input, weight, bias)

    F.linear = _linear_match_dtype

    _orig_layer_norm = F.layer_norm

    def _layer_norm_match_dtype(
        input, normalized_shape, weight=None, bias=None, eps=1e-5
    ):
        if weight is not None and input.is_floating_point() and input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return _orig_layer_norm(input, normalized_shape, weight, bias, eps)

    F.layer_norm = _layer_norm_match_dtype

    F._xpu_dtype_patched = True
    log.info(
        "Patched F.linear and F.layer_norm for XPU bf16 dtype matching"
    )
