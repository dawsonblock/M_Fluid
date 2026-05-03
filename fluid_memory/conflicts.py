"""Conflict-aware scoring and reranking for memory retrieval."""

from typing import List

from fluid_memory.models import MemoryItem
from fluid_memory.retrieval import RetrievalResult


def compute_conflict_penalty(memory: MemoryItem) -> float:
    """Compute a conflict penalty for a memory.

    Higher penalty for:
    - More contradictions
    - Higher volatility
    - Lower stability

    Lower penalty for:
    - Higher confidence

    Returns:
        Penalty value clamped to 0.0–0.8
    """
    penalty = 0.0

    # Contradiction count penalty (0.15 per contradiction, max 0.45)
    penalty += min(memory.contradiction_count * 0.15, 0.45)

    # Volatility penalty (up to 0.25)
    penalty += memory.volatility * 0.25

    # Stability penalty (lower stability = higher penalty, up to 0.2)
    penalty += (1.0 - memory.stability) * 0.2

    # Confidence reduces penalty slightly (up to -0.1 reduction)
    penalty -= memory.confidence * 0.1

    # Clamp to 0.0–0.8
    return max(0.0, min(0.8, penalty))


def compute_support_strength(memory: MemoryItem) -> float:
    """Compute the support strength of a memory.

    Combines:
    - Confidence
    - Salience
    - Stability
    - Trust salience
    - Legal salience

    Penalizes:
    - Contradiction count
    - Volatility

    Returns:
        Support strength clamped to 0.0–1.0
    """
    # Base components
    strength = (
        memory.confidence * 0.25
        + memory.salience * 0.2
        + memory.stability * 0.2
        + memory.trust_salience * 0.15
        + memory.legal_salience * 0.2
    )

    # Penalties
    # Contradiction penalty (up to 0.3 reduction)
    contradiction_penalty = min(memory.contradiction_count * 0.15, 0.3)

    # Volatility penalty (up to 0.2 reduction)
    volatility_penalty = memory.volatility * 0.2

    strength -= contradiction_penalty
    strength -= volatility_penalty

    # Clamp to 0.0–1.0
    return max(0.0, min(1.0, strength))


def rerank_conflict_aware(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Rerank results by conflict-aware adjusted score.

    Adjusts each result's score by penalizing conflicted/volatile memories.
    Preserves all results, only changing their order.

    Args:
        results: List of RetrievalResult objects

    Returns:
        New list of RetrievalResult objects sorted by adjusted score
    """
    if not results:
        return []

    scored_results = []
    for result in results:
        conflict_penalty = compute_conflict_penalty(result.memory)
        adjusted_score = result.score * (1.0 - conflict_penalty)

        # Create new RetrievalResult with adjusted score
        # Store original score in metadata for reference
        new_result = RetrievalResult(
            memory=result.memory,
            score=adjusted_score,
            match_type=result.match_type,
        )
        # Preserve metadata if present
        if hasattr(result, "metadata") and result.metadata:
            new_result.metadata = dict(result.metadata)
            new_result.metadata["original_score"] = result.score
            new_result.metadata["conflict_penalty"] = conflict_penalty
        else:
            new_result.metadata = {
                "original_score": result.score,
                "conflict_penalty": conflict_penalty,
            }

        scored_results.append((adjusted_score, result.score, new_result))

    # Sort by adjusted score (descending), then by original score for determinism
    scored_results.sort(key=lambda x: (-x[0], -x[1]))

    return [r[2] for r in scored_results]
