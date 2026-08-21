"""Command-line interface for standalone pLDDT scoring.

Usage examples::

    # Score a single sequence
    esmfold-score -m /models/ESMFold2-Fast -s "MQIFVKTLTGKTITL..."

    # Score sequences from a FASTA file
    esmfold-score -m /models/ESMFold2-Fast -f generated.fasta

    # Score from stdin (one sequence per line)
    cat seqs.txt | esmfold-score -m /models/ESMFold2-Fast --stdin
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
    )
    parser.add_argument(
        "-m", "--model-path",
        default="biohub/ESMFold2-Fast",
        help="Path to local ESMFold2-Fast weights, or a HuggingFace Hub id "
             "(default: biohub/ESMFold2-Fast).",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-s", "--sequence", help="A single amino-acid sequence.")
    source.add_argument("-f", "--fasta", help="Path to a FASTA file of sequences.")
    source.add_argument(
        "--stdin", action="store_true",
        help="Read sequences from stdin, one per line.",
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
        default=20,
        help="Diffusion sampling steps (default: 20). Quality collapses below "
             "~20; do not lower without re-benchmarking.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Folding trunk recycling loops (default: 1).",
    )
    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=1,
        help="Structures sampled per sequence (default: 1, config default: 32). "
             "More samples cost time and memory without changing pLDDT ranking.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.sequence:
        headers, sequences = ["query"], [args.sequence]
    elif args.fasta:
        entries = parse_fasta(args.fasta)
        headers = [h for h, _ in entries]
        sequences = [s for _, s in entries]
    else:
        sequences = [
            line.strip() for line in sys.stdin
            if line.strip() and not line.startswith(">")
        ]
        headers = [f"seq_{i}" for i in range(len(sequences))]

    if not sequences:
        parser.error("No sequences provided.")

    from esmfold_scorer import StructureScorer

    scorer = StructureScorer(
        model_path=args.model_path,
        device=args.device,
        num_sampling_steps=args.steps,
        num_loops=args.loops,
        num_diffusion_samples=args.diffusion_samples,
    )
    results = scorer.score(sequences)

    rows = list(zip(
        headers,
        results.per_sequence_lengths,
        results.per_sequence_plddt,
        results.per_sequence_ptm,
    ))

    if args.output_json:
        json.dump(
            {
                "mean_plddt": results.mean_plddt,
                "elapsed_seconds": results.elapsed_seconds,
                "sequences": [
                    {"header": h, "length": n, "plddt": p, "ptm": t}
                    for h, n, p, t in rows
                ],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return

    print(f"{'Header':<30} {'Length':>6} {'pLDDT':>7} {'pTM':>7}")
    print("-" * 54)
    for h, n, p, t in rows:
        print(f"{h:<30} {n:>6} {p:>7.4f} {t:>7.4f}")
    print("-" * 54)
    mean_ptm = sum(results.per_sequence_ptm) / len(results.per_sequence_ptm)
    print(
        f"{'OVERALL':<30} {sum(results.per_sequence_lengths):>6} "
        f"{results.mean_plddt:>7.4f} {mean_ptm:>7.4f}"
    )
    print(
        f"\n{results.num_sequences} sequences scored in "
        f"{results.elapsed_seconds:.2f} s"
    )


if __name__ == "__main__":
    main()
