"""Core scoring logic: load ESMFold2-Fast once, score sequences repeatedly.

The public API is :class:`StructureScorer`.  Typical integration into a
generative-model evaluation loop::

    scorer = StructureScorer("/models/ESMFold2-Fast", device="xpu")

    # inside your eval callback:
    results = scorer.score(generated_sequences)
    log_metric("mean_plddt", results.mean_plddt)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from esmfold_scorer.device import optimal_dtype, resolve_device

log = logging.getLogger(__name__)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class ScoringResult:
    """Container for pLDDT scoring results.

    pLDDT and pTM are both on a **0–1** scale, as returned by ESMFold2.
    Multiply by 100 for the 0–100 convention used by AlphaFold outputs.
    """

    mean_plddt: float
    per_sequence_plddt: list[float]
    per_sequence_ptm: list[float]
    per_sequence_lengths: list[int]
    elapsed_seconds: float
    num_sequences: int


class StructureScorer:
    """Score protein sequences by predicted structural quality (mean pLDDT).

    Wraps ESMFold2-Fast for fast pLDDT evaluation.  Intended to be
    instantiated once and reused across many scoring calls.

    Parameters
    ----------
    model_path:
        Path to local ESMFold2-Fast weights directory (containing
        ``config.json`` and ``model.safetensors``), *or* a Hugging Face
        Hub identifier like ``"biohub/ESMFold2-Fast"``.  The ESM-C 6B
        backbone named in the config is resolved from the HuggingFace
        cache; set ``HF_HUB_OFFLINE=1`` to resolve without network.
    device:
        ``"xpu"``, ``"cuda"``, ``"cpu"``, or ``None`` to auto-detect.
        Auto-detection prefers XPU > CUDA > CPU.
    dtype:
        Inference dtype.  ``None`` selects ``bfloat16`` on accelerators,
        ``float32`` on CPU.
    num_sampling_steps:
        Diffusion sampling steps per structure prediction.  Default
        **20**.  Do not lower this without re-benchmarking: output
        quality collapses below ~20 steps rather than degrading
        gracefully (see the benchmark table in the README).
    num_loops:
        Folding-trunk recycling iterations.  Default **1**.  Measured to
        have no effect on pLDDT for typical sequences while costing
        ~45% more wall-clock than the config default of 3.
    num_diffusion_samples:
        Structures sampled per sequence by the diffusion head.  Default
        **1**; the shipped config uses 32, which builds a structural
        ensemble.  pLDDT is averaged over samples anyway, so extra
        samples cost time and memory (pair tensors are ~L² each) without
        changing the ranking.  ``None`` uses the config default.
    empty_cache_every:
        Release cached accelerator memory every N sequences.  Default
        **1** (after each).  Prevents allocator fragmentation across long
        batches, which on Aurora XPU manifests as a GPU page fault rather
        than a clean OOM.  Set ``0`` to disable.
    """

    def __init__(
        self,
        model_path: str = "biohub/ESMFold2-Fast",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        num_sampling_steps: int = 20,
        num_loops: int = 1,
        num_diffusion_samples: int | None = 1,
        empty_cache_every: int = 1,
    ) -> None:
        self.device = resolve_device(device)
        self.dtype = dtype or optimal_dtype(self.device)
        self.num_sampling_steps = num_sampling_steps
        self.num_loops = num_loops
        self.num_diffusion_samples = num_diffusion_samples
        self.empty_cache_every = empty_cache_every

        # Cleared if this build of `esm` rejects num_diffusion_samples,
        # so we only pay for the failed call once.
        self._pass_diffusion_samples = num_diffusion_samples is not None

        log.info(
            "Loading ESMFold2-Fast from %s (device=%s, dtype=%s)",
            model_path,
            self.device,
            self.dtype,
        )
        t0 = time.monotonic()
        self._model = self._load_model(model_path)
        log.info("Model loaded in %.1f s", time.monotonic() - t0)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = self._resolve_model_class().from_pretrained(model_path)
        model = model.to(device=self.device, dtype=self.dtype).eval()

        if self.device.type == "xpu":
            _patch_linalg_cpu_roundtrip()
            if self.dtype != torch.float32:
                _patch_dtype_casting()

        return model

    @staticmethod
    def _resolve_model_class() -> type:
        try:
            from esm.models.esmfold2 import EsmFold2Model

            log.info("Using EsmFold2Model from esm package")
            return EsmFold2Model
        except ImportError:
            pass

        # Newer transformers may ship ESMFold2 natively.
        try:
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

            log.info("Using ESMFold2Model from transformers")
            return ESMFold2Model
        except ImportError:
            pass

        raise ImportError(
            "Cannot find the ESMFold2 model class. Install it with:\n"
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
        num_sampling_steps, num_loops:
            Override the instance defaults for this call only.

        Returns
        -------
        ScoringResult
            pLDDT and pTM on a 0–1 scale, plus timing.

        Raises
        ------
        ValueError
            If *sequences* is empty, any entry is empty after
            sanitization, or *steps* / *loops* are below 1.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        if not sequences:
            raise ValueError("sequences must be a non-empty list")

        steps = self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        loops = self.num_loops if num_loops is None else num_loops
        if steps < 1 or loops < 1:
            raise ValueError("num_sampling_steps and num_loops must be >= 1")

        sequences = [self._sanitize(s) for s in sequences]
        if any(not s for s in sequences):
            raise ValueError("All sequences must be non-empty after sanitization")

        t0 = time.monotonic()
        plddt: list[float] = []
        ptm: list[float] = []

        for i, seq in enumerate(sequences):
            output = self._infer(seq, loops=loops, steps=steps)
            plddt.append(float(output["plddt"].mean()))
            ptm.append(float(output["ptm"].mean()))

            # Structure prediction allocates large transient pair tensors
            # (~L² per diffusion sample). Releasing them keeps the XPU
            # caching allocator from fragmenting across a long batch.
            del output
            if self.empty_cache_every and (i + 1) % self.empty_cache_every == 0:
                empty_cache(self.device)

        elapsed = time.monotonic() - t0
        mean_plddt = sum(plddt) / len(plddt)

        log.info(
            "Scored %d sequence(s) in %.2f s — mean pLDDT: %.4f",
            len(sequences),
            elapsed,
            mean_plddt,
        )
        return ScoringResult(
            mean_plddt=mean_plddt,
            per_sequence_plddt=plddt,
            per_sequence_ptm=ptm,
            per_sequence_lengths=[len(s) for s in sequences],
            elapsed_seconds=elapsed,
            num_sequences=len(sequences),
        )

    def _infer(self, seq: str, *, loops: int, steps: int) -> dict:
        """Run one structure prediction, adapting to this build's kwargs.

        ``infer_protein`` forwards ``**forward_kwargs`` straight through,
        so an unsupported kwarg surfaces as a ``TypeError`` from
        ``forward()`` rather than being rejected up front.  Match on the
        kwarg name so unrelated ``TypeError``s still propagate.
        """
        kwargs = {"num_loops": loops, "num_sampling_steps": steps}

        if self._pass_diffusion_samples:
            try:
                return self._model.infer_protein(
                    seq, num_diffusion_samples=self.num_diffusion_samples, **kwargs
                )
            except TypeError as exc:
                if "num_diffusion_samples" not in str(exc):
                    raise
                self._pass_diffusion_samples = False
                log.warning(
                    "infer_protein() rejected num_diffusion_samples; using the "
                    "config default. Peak memory will be higher on long sequences."
                )

        return self._model.infer_protein(seq, **kwargs)

    @staticmethod
    def _sanitize(seq: str) -> str:
        """Normalize a sequence: uppercase, strip whitespace, replace unknowns."""
        seq = "".join(seq.upper().split())
        return "".join(c if c in VALID_AA else "X" for c in seq)


def empty_cache(device: torch.device) -> None:
    """Release cached allocator blocks on *device* (no-op on CPU)."""
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def peak_memory_gb(device: torch.device) -> float:
    """Peak allocator memory in GiB since the last reset (0.0 on CPU)."""
    if device.type == "xpu":
        return torch.xpu.max_memory_allocated() / 1024**3
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def _patch_linalg_cpu_roundtrip() -> None:
    """Route SVD/det through an explicit CPU round-trip on XPU.

    ``torch.linalg.svd`` has no XPU kernel, so PyTorch's aten dispatcher
    silently falls back to CPU ("Aten Op fallback from XPU to CPU").  That
    automatic fallback corrupts GPU memory on Aurora's compute-runtime:
    inference aborts with "Segmentation fault from GPU" immediately after
    the fallback warning.

    Doing the transfer ourselves — copy to CPU, compute in float32, copy
    back — keeps the dispatcher out of the fallback path.  The tensors are
    per-residue 3x3 rotations, so the round-trip cost is negligible.
    """
    if getattr(torch.linalg.svd, "_xpu_patched", False):
        return

    _orig_svd = torch.linalg.svd

    def svd(A, full_matrices=True, **kwargs):
        if A.is_xpu:
            cpu = A.detach().to("cpu", torch.float32)
            return tuple(
                t.to(A.device, A.dtype)
                for t in _orig_svd(cpu, full_matrices=full_matrices)
            )
        return _orig_svd(A, full_matrices=full_matrices, **kwargs)

    _orig_det = torch.linalg.det

    def det(A):
        if A.is_xpu:
            cpu = A.detach().to("cpu", torch.float32)
            return _orig_det(cpu).to(A.device, A.dtype)
        return _orig_det(A)

    svd._xpu_patched = True
    torch.linalg.svd = svd
    torch.linalg.det = det
    log.info("Patched torch.linalg.svd/det for XPU (explicit CPU round-trip)")


def _patch_dtype_casting() -> None:
    """Cast float32 inputs to the weight dtype in F.linear / F.layer_norm.

    ``esm`` wraps its forward passes in ``torch.autocast(device_type="cuda")``,
    which silently disables itself on XPU.  Model weights are bf16 but
    ``infer_protein`` builds float32 feature tensors, so the two meet at a
    dtype mismatch.  Patching just these two ops reproduces the dtype
    matching autocast would have done, without promoting linalg ops to
    bf16 — those lack bf16 XPU kernels and fault the GPU driver.
    """
    import torch.nn.functional as F

    if getattr(F, "_xpu_dtype_patched", False):
        return

    _orig_linear = F.linear

    def linear(input, weight, bias=None):
        if input.is_floating_point() and input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        if bias is not None and bias.dtype != weight.dtype:
            bias = bias.to(weight.dtype)
        return _orig_linear(input, weight, bias)

    _orig_layer_norm = F.layer_norm

    def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        if weight is not None and input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return _orig_layer_norm(input, normalized_shape, weight, bias, eps)

    F.linear = linear
    F.layer_norm = layer_norm
    F._xpu_dtype_patched = True
    log.info("Patched F.linear and F.layer_norm for XPU bf16 dtype matching")
