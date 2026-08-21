# esmfold-scorer

Fast mean-pLDDT scoring from protein sequences using **ESMFold2-Fast**.
Designed as a drop-in evaluation step for protein generative models:
load the structure predictor once, then score batches of generated
sequences on-the-fly during training.

## What it does

Given one or more amino-acid sequences, `esmfold-scorer` runs ESMFold2-Fast
(a single-sequence, diffusion-based structure prediction model) and returns:

- **mean pLDDT** — per-sequence and overall, on a **0–1** scale (multiply
  by 100 for the AlphaFold convention).  Higher is better; ≳0.7 generally
  indicates a confident fold.
- **pTM** — predicted template modeling score (0–1).  Measures global
  structural accuracy.
- **Timing** — wall-clock seconds, useful for throughput budgeting.

pLDDT is the primary metric for evaluating whether a generative model is
producing sequences that fold into well-defined 3D structures.

## Speed defaults

Defaults are **20 sampling steps, 1 recycling loop, 1 diffusion sample**,
chosen from the benchmark below.  This is ~1.9× faster than the config
defaults (50 steps, 3 loops) at statistically indistinguishable pLDDT.

> **Do not lower `num_sampling_steps` below 20 without re-benchmarking.**
> Quality does not degrade gracefully — it *collapses*.  See below.

### Benchmark

100 SwissProt sequences (63–297 aa, mean 180), single Aurora PVC tile,
bfloat16, `num_diffusion_samples=1`.  Model load ~30 s; 12.3 GiB resident,
13.4 GiB peak.

| steps | loops | mean pLDDT | s/seq | seq/s |
|------:|------:|-----------:|------:|------:|
|     5 |     1 |     0.2430 |  0.78 |  1.28 |
|    10 |     1 |     0.2616 |  0.95 |  1.05 |
| **20** | **1** | **0.8446** | **1.10** | **0.91** |
|    50 |     1 |     0.8440 |  1.63 |  0.61 |
|     5 |     3 |     0.2451 |  1.32 |  0.76 |
|    10 |     3 |     0.2614 |  1.42 |  0.70 |
|    20 |     3 |     0.8400 |  1.58 |  0.63 |
|    50 |     3 |     0.8415 |  2.11 |  0.47 |

Two things this measured:

**Recycling loops do nothing here.** At every step count, `num_loops=3`
matched `num_loops=1` to within 0.005 pLDDT while costing 40–45% more
wall-clock.  Hence the default of 1.

**Sampling steps have a cliff between 10 and 20.** The jump from 0.26 to
0.84 is not convergence, it is a phase change: at 5–10 steps *every*
sequence lands near 0.25, the no-information floor, so the scores carry no
discriminative signal at all.  A generative-model eval loop running at 10
steps would silently produce noise that looks like a plausible metric.
Above 20, scores are converged (20 vs 50 differ by 0.0006).

The cliff has not been localized more finely than 10–20 — the shipped
`speed_test.pbs` sweeps 12/14/16/18 to pin it down.  Until then, note that
the model config's own `inference_num_steps` is 14, suggesting the
intended floor sits just below 20 and that the default has only modest
headroom.  If you need to go faster, take it out of `num_diffusion_samples`
or sequence count, not out of steps.

## Installation

```bash
pip install -e .
```

This installs the `esm` package from the [Biohub GitHub repo](https://github.com/Biohub/esm)
and HuggingFace `transformers` ≥ 4.57.

### Aurora / HPC setup

On systems like ALCF Aurora where PyTorch and IPEX come from a
`module load frameworks`, create a lightweight venv that inherits the
system packages.  The `esm` package must be installed with `--no-deps`
to prevent it from pulling a PyPI CUDA `torch` build that would shadow
the system's XPU torch:

```bash
module load frameworks
python -m venv /path/to/esmfold-env --system-site-packages
source /path/to/esmfold-env/bin/activate
pip install --ignore-installed 'transformers>=4.57'
pip install --no-deps 'esm @ git+https://github.com/Biohub/esm.git@main'
pip install biopython biotite cloudpathlib zstd msgpack-numpy pygtrie tenacity brotli
```

`--system-site-packages` inherits `torch`, `intel_extension_for_pytorch`,
`oneCCL`, etc. from the frameworks module.  `--ignore-installed` on
`transformers` ensures the venv gets its own copy (the frameworks module
bundles an older version that would otherwise satisfy the requirement).

**ESM-C backbone:** The ESM-C 6B backbone must be cached in
`~/.cache/huggingface/hub/` before running on compute nodes.  Download it
once from a login node: `python -c "from huggingface_hub import snapshot_download; snapshot_download('biohub/ESMC-6B')"`.
On compute nodes, set `HF_HUB_OFFLINE=1` so the Hub library resolves
from cache without network access.

The 6B backbone is **required** — it is not swappable for ESM-C 300M.
ESMFold2-Fast's config hard-codes `lm_d_model: 2560` and
`lm_num_layers: 80`, whereas ESMC-300M is `d_model: 960` / `n_layers: 30`.
Loaded in bfloat16 it occupies 12.3 GiB, which fits comfortably on one
Aurora PVC tile (64 GiB).

### Integrating into another project's environment

If your generative model already has its own venv (including on Aurora),
add the dependencies there instead of creating a separate environment.
Use `--no-deps` on `esm` if the venv's `torch` comes from a
non-PyPI source (e.g. the Aurora frameworks module):

```bash
source /path/to/your-project-env/bin/activate

# Standard install (torch from PyPI):
pip install 'transformers>=4.57' 'esm @ git+https://github.com/Biohub/esm.git@main'

# Or XPU / custom-torch install (prevent esm from clobbering your torch):
pip install 'transformers>=4.57'
pip install --no-deps 'esm @ git+https://github.com/Biohub/esm.git@main'
pip install biopython biotite cloudpathlib
```

Then either `pip install -e /path/to/EsmFold` to make `esmfold_scorer`
importable, or add `EsmFold/src` to `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/EsmFold/src${PYTHONPATH:+:$PYTHONPATH}"
```

The scorer's runtime dependencies are `torch`, `transformers>=4.57`, and
`esm` — it should coexist cleanly with any training framework.

### Model weights

The first time you load the model, it needs:

1. **ESMFold2-Fast weights** — 720 MB.  Point `model_path` to a local
   directory (containing `config.json` + `model.safetensors`), or use
   `"biohub/ESMFold2-Fast"` to download from HuggingFace Hub.
2. **ESM-C 6B backbone** — named by `esmc_id` in the ESMFold2-Fast config.
   Downloaded automatically by the `esm` package on first load and cached
   in `~/.cache/huggingface/hub/`.

For HPC compute nodes without network access, pre-cache both weights
from a login node and set `HF_HUB_OFFLINE=1` at runtime.

## Python API

### Minimal example

```python
from esmfold_scorer import StructureScorer

scorer = StructureScorer("/path/to/ESMFold2-Fast")
results = scorer.score(["MQIFVKTLTGKTITLEVEPSDTIENVKAK"])

print(results.mean_plddt)          # e.g. 0.8446  (0–1 scale)
print(results.per_sequence_plddt)  # [0.8446]
print(results.per_sequence_ptm)    # [0.8712]
print(results.elapsed_seconds)     # 1.10
```

### Integration into a training eval loop

```python
from esmfold_scorer import StructureScorer

# --- once, at eval setup ---
# Loading takes ~30 s and holds 12.3 GiB, so build the scorer once and
# reuse it; do not construct one per eval step.
scorer = StructureScorer(
    model_path="/models/ESMFold2-Fast",
    device="xpu",          # auto-detects if None
)

# --- inside your eval callback ---
def evaluate_generated_sequences(sequences: list[str]) -> dict:
    """Score a batch of generated sequences and return metrics."""
    results = scorer.score(sequences)
    return {
        "mean_plddt": results.mean_plddt,
        "per_sequence_plddt": results.per_sequence_plddt,
        "num_sequences": results.num_sequences,
        "scoring_time_s": results.elapsed_seconds,
    }
```

### StructureScorer parameters

| Parameter             | Default                    | Description                                                           |
|-----------------------|----------------------------|-----------------------------------------------------------------------|
| `model_path`          | `"biohub/ESMFold2-Fast"`   | Local weights directory or HuggingFace Hub id.                        |
| `device`              | `None` (auto)              | `"xpu"`, `"cuda"`, `"cpu"`, or `None` for auto-detect.               |
| `dtype`               | `None` (auto)              | `bfloat16` on accelerators, `float32` on CPU.                         |
| `num_sampling_steps`  | `20`                       | Diffusion steps. **Do not lower** — quality collapses below ~20.      |
| `num_loops`           | `1`                        | Trunk recycling loops. Measured to have no effect on pLDDT.           |
| `num_diffusion_samples` | `1`                      | Structures sampled per sequence. Config default 32. Drives peak memory. |
| `empty_cache_every`   | `1`                        | Release cached device memory every N sequences. `0` disables.         |

### ScoringResult fields

| Field                   | Type          | Description                                      |
|-------------------------|---------------|--------------------------------------------------|
| `mean_plddt`            | `float`       | Mean pLDDT across all sequences (**0–1** scale).  |
| `per_sequence_plddt`    | `list[float]` | Mean pLDDT for each input sequence (0–1).        |
| `per_sequence_ptm`      | `list[float]` | pTM for each input sequence (0–1).               |
| `per_sequence_lengths`  | `list[int]`   | Residue count for each input sequence.           |
| `elapsed_seconds`       | `float`       | Wall-clock time for the scoring call.            |
| `num_sequences`         | `int`         | Number of sequences scored.                      |

## CLI

```bash
# Single sequence
esmfold-score -m /models/ESMFold2-Fast -s "MQIFVKTLTGKTITL..."

# FASTA file
esmfold-score -m /models/ESMFold2-Fast -f generated.fasta

# From stdin, one sequence per line
cat seqs.txt | esmfold-score -m /models/ESMFold2-Fast --stdin

# JSON output for programmatic consumption
esmfold-score -m /models/ESMFold2-Fast -f seqs.fasta --json

# Higher-accuracy settings (slower, matches the model config defaults)
esmfold-score -m /models/ESMFold2-Fast -f seqs.fasta --steps 50 --loops 3

# Offline mode (resolve all models from HuggingFace cache)
HF_HUB_OFFLINE=1 esmfold-score -m /models/ESMFold2-Fast -f seqs.fasta

# Verbose logging (model load times, per-batch info)
esmfold-score -m /models/ESMFold2-Fast -f seqs.fasta -v
```

## Device portability

The scorer auto-detects the best available device:

| Priority | Backend | When used                                     |
|----------|---------|-----------------------------------------------|
| 1        | XPU     | `intel_extension_for_pytorch` installed + GPU  |
| 2        | CUDA    | NVIDIA GPU visible                             |
| 3        | CPU     | Fallback                                       |

Override with `device="xpu"` / `device="cuda"` / `device="cpu"` in the
constructor or `--device` on the CLI.

On **CUDA (Ampere+)**, the esm package uses `torch.autocast` internally
for mixed-precision bf16 inference — no workarounds needed.

### Aurora / XPU workarounds

The esm package targets CUDA, so two things need patching on XPU.  Both
are applied automatically when `device` resolves to `xpu`:

1. **Dtype matching.**  esm's autocast is hardcoded to
   `device_type="cuda"` and silently disables itself on XPU (you'll see
   `UserWarning: CUDA is not available ... Disabling autocast`).  Model
   weights are bf16 but `infer_protein` builds float32 feature tensors,
   so `F.linear` and `F.layer_norm` are patched to cast inputs to the
   weight dtype.

2. **Linalg CPU round-trip.**  `torch.linalg.svd` has no XPU kernel.
   PyTorch's automatic aten fallback to CPU corrupts GPU memory on
   Aurora's compute-runtime — inference aborts with
   `Segmentation fault from GPU`.  `svd` and `det` are patched to do the
   CPU transfer explicitly, which keeps the dispatcher out of the
   fallback path.  The tensors are small (per-residue 3×3 rotations), so
   the cost is negligible.

> **Integration caveat:** these are process-global monkey-patches on
> `torch.linalg` and `torch.nn.functional`.  They are written to be
> no-ops outside the cases they target — the linalg patches only trigger
> on XPU tensors, and the dtype patches only cast when input and weight
> dtypes already differ — but if your training code depends on
> `F.linear` raising on a dtype mismatch, be aware they are in effect
> once a `StructureScorer` is constructed on XPU.

## Speed test (Aurora)

A PBS job script and 100-sequence FASTA file are included for
benchmarking on Aurora.  The script loads the model once, smoke-tests the
longest sequence, then sweeps step/loop configurations reporting pLDDT,
throughput and peak memory:

```bash
qsub speed_test.pbs
```

Edit the paths at the top of `speed_test.pbs` if your weight / env
locations differ.  Results go to the job's `.o` file.

To audit which aten ops silently fall back from XPU to CPU — worth
checking after an `esm` or PyTorch upgrade, since a *new* linalg fallback
would reintroduce the GPU page fault described above — add
`export PYTORCH_DEBUG_XPU_FALLBACK=1` to the script.

## Project structure

```
EsmFold/
├── pyproject.toml                  # Package metadata and dependencies
├── README.md
├── speed_test.pbs                  # PBS job script for Aurora benchmarking
├── test_sequences.fasta            # 100 SwissProt sequences (63–297 aa)
└── src/
    └── esmfold_scorer/
        ├── __init__.py             # Public API: StructureScorer, ScoringResult
        ├── scorer.py               # Model loading, inference, result aggregation
        ├── device.py               # XPU / CUDA / CPU detection
        └── cli.py                  # Command-line entry point
```

## Tuning for your workload

**Budgeting an eval step:** at the defaults, expect ~1.1 s/sequence on one
Aurora tile for 63–297 aa inputs, plus a one-time ~30 s model load.  A
256-sequence eval batch is therefore ~4.5 minutes.  Cost is dominated by
sequence count, not length — the 297 aa smoke test and the 100-sequence
mean both land near 1.1 s.

**Long sequences (> 500 residues):** Memory scales as roughly
`num_diffusion_samples × L²`.  Keep `num_diffusion_samples=1` (the default
here, vs. 32 in the shipped config).  On CUDA, `dtype=torch.float16` is an
option if bf16 causes OOM.  If you see a GPU page fault rather than a
clean OOM on XPU, that is the allocator fragmenting — check
`empty_cache_every` and the sample count before assuming the sequence is
too long.

**Very large batches:** Sequences are scored one at a time — ESMFold2's
`infer_protein` takes a single sequence, so there is no intra-batch
parallelism to exploit.  Throughput scales linearly with sequence count.
For better device utilization, run multiple scorer processes on separate
tiles (`ZE_AFFINITY_MASK`) rather than looking for a batch API.

## License

MIT — same as ESMFold2-Fast.  See the
[third-party notice](https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md)
for ESMFold2's upstream dependencies.
