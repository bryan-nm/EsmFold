"""Fast mean-pLDDT scoring from protein sequences using ESMFold2-Fast.

Designed for on-the-fly evaluation of protein generative models. Loads the
ESMFold2-Fast structure prediction model once and exposes a simple API for
scoring batches of amino-acid sequences by predicted structural quality
(mean pLDDT).

Quick start::

    from esmfold_scorer import StructureScorer

    scorer = StructureScorer("/path/to/ESMFold2-Fast")
    results = scorer.score(["MKTVRQERLK...", "MGSSHHHHH..."])
    print(results.mean_plddt)          # scalar across all sequences
    print(results.per_sequence_plddt)  # list[float], one per input
"""

from esmfold_scorer.scorer import ScoringResult, StructureScorer

__all__ = ["StructureScorer", "ScoringResult"]
