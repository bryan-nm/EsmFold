"""Command-line interface for standalone pLDDT scoring.

Usage examples::

    # Score a single sequence
    esmfold-score -m /models/ESMFold2-Fast -s "MQIFVKTLTGKTITL..."

    # Score sequences from a FASTA file
    esmfold-score -m /models/ESMFold2-Fast -f generated.fasta

    # Score from stdin (one sequence per line)
    cat seqs.txt | esmfold-score -m /models/ESMFold2-Fast --stdin

    # Fast mode with fewer diffusion steps
    esmfold-score -m /models/ESMFold2-Fast -f seqs.fasta --steps 5 --loops 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def parse_fasta(path: str) -> list[tuple[str, str]]:
    """Parse a FASTA file into (header, sequence) pairs."""
    entries: list[tuple[str, str]] = []
    header = ""
    chunks: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if chunks:
                    entries.append((header, "".join(chunks)))
                header = line[1:].strip()
                chunks = []
            elif line:
                chunks.append(line)
    if chunks:
        entries.append((header, "".join(chunks)))
    return entries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score protein sequences by predicted pLDDT using ESMFold2-Fast.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model-path",
        default="biohub/ESMFold2-Fast",
        help="Path to local ESMFold2-Fast weights, or a HuggingFace Hub id "
             "(default: biohub/ESMFold2-Fast).",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-s", "--sequence",
        help="A single amino-acid sequence to score.",
    )
    input_group.add_argument(
        "-f", "--fasta",
        help="Path to a FASTA file of sequences.",
    )
    input_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read sequences from stdin, one per line.",
    )

    parser.add_argument(
        "--esmc-id",
        default=None,
        help="HuggingFace Hub repo ID for the ESM-C backbone "
             "(default: from ESMFold2-Fast config, usually biohub/ESMC-6B). "
             "Set HF_HUB_OFFLINE=1 to resolve from cache without network.",
    )
    parser.add_argument(
        "--device",
        choices=["xpu", "cuda", "cpu"],
        default=None,
        help="Device to run on (default: auto-detect).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Diffusion sampling steps (default: 10, paper default: 50).",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Folding trunk recycling loops (default: 1, paper default: 3).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Run torch.compile() on the model (slow startup, faster throughput).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    sequences: list[str] = []
    headers: list[str] = []

    if args.sequence:
        sequences = [args.sequence]
        headers = ["query"]
    elif args.fasta:
        entries = parse_fasta(args.fasta)
        headers = [h for h, _ in entries]
        sequences = [s for _, s in entries]
    elif args.stdin:
        for i, line in enumerate(sys.stdin):
            line = line.strip()
            if line and not line.startswith(">"):
                sequences.append(line)
                headers.append(f"seq_{i}")

    if not sequences:
        parser.error("No sequences provided.")

    from esmfold_scorer import StructureScorer

    scorer = StructureScorer(
        model_path=args.model_path,
        device=args.device,
        esmc_model_id=args.esmc_id,
        num_sampling_steps=args.steps,
        num_loops=args.loops,
        compile_model=args.compile,
    )
    results = scorer.score(sequences)

    if args.output_json:
        data = {
            "mean_plddt": results.mean_plddt,
            "elapsed_seconds": results.elapsed_seconds,
            "sequences": [
                {
                    "header": h,
                    "length": l,
                    "plddt": p,
                    "ptm": t,
                }
                for h, l, p, t in zip(
                    headers,
                    results.per_sequence_lengths,
                    results.per_sequence_plddt,
                    results.per_sequence_ptm,
                )
            ],
        }
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"{'Header':<30} {'Length':>6} {'pLDDT':>7} {'pTM':>7}")
        print("-" * 54)
        for h, l, p, t in zip(
            headers,
            results.per_sequence_lengths,
            results.per_sequence_plddt,
            results.per_sequence_ptm,
        ):
            print(f"{h:<30} {l:>6} {p:>7.2f} {t:>7.4f}")
        print("-" * 54)
        print(
            f"{'OVERALL':<30} {sum(results.per_sequence_lengths):>6} "
            f"{results.mean_plddt:>7.2f} "
            f"{sum(results.per_sequence_ptm) / len(results.per_sequence_ptm):>7.4f}"
        )
        print(f"\n{results.num_sequences} sequences scored in {results.elapsed_seconds:.2f} s")


if __name__ == "__main__":
    main()
