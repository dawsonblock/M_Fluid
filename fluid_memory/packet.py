"""Retrieval packet dataclasses for evidence-grounded memory answers."""

from dataclasses import dataclass, field
from typing import List, Optional, Any

from fluid_memory.models import MemoryItem
from fluid_memory.retrieval import RetrievalResult


@dataclass
class MemoryEvidenceRef:
    """Reference to a memory as evidence for an answer."""

    memory_id: str
    content_hash: str
    source_refs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.5
    salience: float = 0.5
    stability: float = 0.5
    volatility: float = 0.3
    contradiction_count: int = 0
    invalidated: bool = False

    @classmethod
    def from_memory(cls, memory: MemoryItem) -> "MemoryEvidenceRef":
        """Create evidence ref from a MemoryItem."""
        return cls(
            memory_id=memory.memory_id,
            content_hash=memory.content_hash,
            source_refs=list(memory.source_refs) if memory.source_refs else [],
            tags=list(memory.tags) if memory.tags else [],
            confidence=memory.confidence,
            salience=memory.salience,
            stability=memory.stability,
            volatility=memory.volatility,
            contradiction_count=memory.contradiction_count,
            invalidated=memory.invalidated_at is not None,
        )


@dataclass
class RetrievalPacket:
    """A packet containing retrieval results with evidence grounding."""

    query: Optional[str]
    results: List[RetrievalResult] = field(default_factory=list)
    evidence_refs: List[MemoryEvidenceRef] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    support_level: str = "none"
    top_score: float = 0.0
    total_results: int = 0


def build_retrieval_packet(
    query: Optional[str],
    results: List[RetrievalResult],
    min_strong_results: int = 2,
    weak_score_threshold: float = 0.35,
    strong_score_threshold: float = 0.70,
) -> RetrievalPacket:
    """Build a RetrievalPacket from retrieval results with evidence grounding.

    Args:
        query: The original query
        results: List of RetrievalResult objects
        min_strong_results: Minimum results needed for "strong" support
        weak_score_threshold: Score below which support is considered "weak"
        strong_score_threshold: Score threshold for "strong" individual results

    Returns:
        RetrievalPacket with evidence refs and support level assessment
    """
    warnings: List[str] = []
    evidence_refs: List[MemoryEvidenceRef] = []

    if not results:
        return RetrievalPacket(
            query=query,
            results=[],
            evidence_refs=[],
            warnings=[],
            support_level="none",
            top_score=0.0,
            total_results=0,
        )

    top_score = max(r.score for r in results)
    total_results = len(results)

    # Build evidence refs and collect warnings
    for result in results:
        ref = MemoryEvidenceRef.from_memory(result.memory)
        evidence_refs.append(ref)

        # Check for invalidated memories
        if ref.invalidated:
            warnings.append(
                f"Memory {ref.memory_id} is invalidated but appeared in results"
            )

        # Check for high volatility
        if ref.volatility > 0.7:
            warnings.append(
                f"Memory {ref.memory_id} has high volatility ({ref.volatility:.2f})"
            )

        # Check for contradictions
        if ref.contradiction_count > 0:
            warnings.append(
                f"Memory {ref.memory_id} has {ref.contradiction_count} contradictions"
            )

    # Determine support level
    support_level = _compute_support_level(
        results=results,
        evidence_refs=evidence_refs,
        top_score=top_score,
        min_strong_results=min_strong_results,
        weak_score_threshold=weak_score_threshold,
        strong_score_threshold=strong_score_threshold,
    )

    return RetrievalPacket(
        query=query,
        results=results,
        evidence_refs=evidence_refs,
        warnings=warnings,
        support_level=support_level,
        top_score=top_score,
        total_results=total_results,
    )


def _compute_support_level(
    results: List[RetrievalResult],
    evidence_refs: List[MemoryEvidenceRef],
    top_score: float,
    min_strong_results: int,
    weak_score_threshold: float,
    strong_score_threshold: float,
) -> str:
    """Compute the support level for retrieval results."""
    if not results:
        return "none"

    # Check for any contradicted or volatile results
    has_contradictions = any(r.contradiction_count > 0 for r in evidence_refs)
    has_high_volatility = any(r.volatility > 0.7 for r in evidence_refs)

    # Count strong results
    strong_count = 0
    for result, ref in zip(results, evidence_refs):
        is_strong = (
            result.score >= strong_score_threshold
            and ref.confidence >= 0.7
            and ref.stability >= 0.5
            and ref.contradiction_count == 0
        )
        if is_strong:
            strong_count += 1

    # Determine support level
    # Contradictions or high volatility take precedence
    if has_contradictions or has_high_volatility:
        return "mixed"

    if strong_count >= min_strong_results:
        return "strong"

    if top_score < weak_score_threshold:
        return "weak"

    # Single decent result without corroboration
    if len(results) == 1 or strong_count == 0:
        return "supported"

    # Multiple results but none individually strong
    return "supported"
