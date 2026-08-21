# esmfold-scorer

Fast mean-pLDDT scoring from protein sequences using **ESMFold2-Fast**.
Designed as a drop-in evaluation step for protein generative models:
load the structure predictor once, then score batches of generated
sequences on-the-fly during training.

## What it does

Given one or more amino-acid sequences, `esmfold-scorer` runs ESMFold2-Fast
(a single-sequence, diffusion-based structure prediction model) and returns:

- **mean pLDDT** — per-sequence and overall, on the standard 0–100 scale.
  Higher is better; ≥70 generally indicates a confident fold.
- **pTM** — predicted template modeling score (0–1).  Measures global
  structural accuracy.
- **Timing** — wall-clock seconds, useful for throughput budgeting.

pLDDT is the primary metric for evaluating whether a generative model is
producing sequences that fold into well-defined 3D structures.

## Speed defaults

The paper-default inference settings (50 diffusion steps, 3 recycling
loops) maximize absolute accuracy.  For **relative ranking** during
training — which is what an eval loop needs — fewer steps are sufficient:

| Preset     | `num_sampling_steps` | `num_loops` | Relative speed |
|------------|---------------------:|------------:|:--------------:|
| Paper      |                   50 |           3 | 1×             |
| Default    |                   10 |           1 | ~8–12×         |
| Ultra-fast |                    5 |           1 | ~15–20×        |

The ranking correlation between 10-step and 50-step pLDDT is high for
typical generated sequences.  Start with the defaults and increase
`num_sampling_steps` only if you need publication-quality absolute scores.

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

**ESM-C backbone:** The ESM-C 6B backbone (~24 GB) must be cached in
`~/.cache/huggingface/hub/` before running on compute nodes.  Download it
once from a login node: `python -c "from huggingface_hub import snapshot_download; snapshot_download('biohub/ESMC-6B')"`.
On compute nodes, set `HF_HUB_OFFLINE=1` so the Hub library resolves
from cache without network access.

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
2. **ESM-C 6B backbone** — ~24 GB.  Downloaded automatically by the
   `esm` package on first load and cached in the HuggingFace cache
   directory (`~/.cache/huggingface/hub/`).

For HPC compute nodes without network access, pre-cache both weights
from a login node and set `HF_HUB_OFFLINE=1` at runtime:

## Python API

### Minimal example

```python
from esmfold_scorer import StructureScorer

scorer = StructureScorer("/path/to/ESMFold2-Fast")
results = scorer.score(["MQIFVKTLTGKTITLEVEPSDTIENVKAK"])

print(results.mean_plddt)          # e.g. 82.5
print(results.per_sequence_plddt)  # [82.5]
print(results.per_sequence_ptm)    # [0.87]
print(results.elapsed_seconds)     # 1.23
```

### Integration into a training eval loop

```python
import torch
from esmfold_scorer import StructureScorer

# --- once, at eval setup ---
scorer = StructureScorer(
    model_path="/models/ESMFold2-Fast",
    device="xpu",              # auto-detects if None
    num_sampling_steps=10,     # fast defaults
    num_loops=1,
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
| `esmc_model_id`       | `None`                     | Hub repo ID for ESM-C backbone. `None` uses config default.           |
| `num_sampling_steps`  | `10`                       | Diffusion steps. Lower = faster. Paper default: 50.                   |
| `num_loops`           | `1`                        | Trunk recycling loops. Lower = faster. Paper default: 3.              |
| `compile_model`       | `False`                    | Run `torch.compile()`. Slow startup, faster steady-state throughput.  |

### ScoringResult fields

| Field                   | Type          | Description                                     |
|-------------------------|---------------|-------------------------------------------------|
| `mean_plddt`            | `float`       | Scalar mean pLDDT across all sequences (0–100). |
| `per_sequence_plddt`    | `list[float]` | Mean pLDDT for each input sequence.             |
| `per_sequence_ptm`      | `list[float]` | pTM for each input sequence (0–1).              |
| `per_sequence_lengths`  | `list[int]`   | Residue count for each input sequence.          |
| `elapsed_seconds`       | `float`       | Wall-clock time for the scoring call.           |
| `num_sequences`         | `int`         | Number of sequences scored.                     |

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

# Faster with fewer steps
esmfold-score -m /models/ESMFold2-Fast -f seqs.fasta --steps 5 --loops 1

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
benchmarking on Aurora.  The script sweeps step/loop configurations
and reports throughput:

```bash
qsub speed_test.pbs
```

Edit the paths at the top of `speed_test.pbs` if your weight / env
locations differ.  Results go to the job's `.o` file.

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

**Short sequences (< 100 residues):** `torch.compile()` amortizes well.
Pass `compile_model=True` if you're scoring hundreds of short peptides.

**Long sequences (> 500 residues):** Memory dominates.  Keep
`num_sampling_steps` low and consider `dtype=torch.float16` on CUDA if
bf16 causes OOM.

**Very large batches:** Sequences are scored one at a time (ESMFold2's
`infer_protein` is single-sequence).  Throughput scales linearly with
sequence count.  For maximum GPU utilization on short sequences, consider
running multiple scorer processes.

## License

MIT — same as ESMFold2-Fast.  See the
[third-party notice](https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md)
for ESMFold2's upstream dependencies.
