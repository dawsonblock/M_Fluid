"""Memory orchestrator for evidence-grounded retrieval and answer grounding."""

from typing import Dict, Any, List, Optional

from fluid_memory.engine import FluidMemoryEngine
from fluid_memory.packet import RetrievalPacket, build_retrieval_packet
from fluid_memory.conflicts import rerank_conflict_aware


class MemoryOrchestrator:
    """Orchestrates memory retrieval with evidence grounding and conflict awareness.

    This is a testable retrieval/grounding layer, not a truth oracle.
    Evidence references are provenance aids, not proof by themselves.
    """

    def __init__(self, engine: FluidMemoryEngine):
        """Initialize orchestrator with a FluidMemoryEngine.

        Args:
            engine: The FluidMemoryEngine to use for retrieval
        """
        self.engine = engine

    def retrieve_packet(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        use_semantic: bool = False,
        semantic_threshold: float = 0.5,
        conflict_aware: bool = True,
        enable_mmr: bool = False,
    ) -> RetrievalPacket:
        """Retrieve memories as a grounded evidence packet.

        Args:
            query: Search query text
            tags: Filter by tags
            limit: Maximum results
            use_semantic: Use semantic search if True
            semantic_threshold: Minimum similarity for semantic search
            conflict_aware: Apply conflict-aware reranking if True
            enable_mmr: Use Maximal Marginal Relevance for diversity

        Returns:
            RetrievalPacket with evidence refs and support level
        """
        fallback_warning = None

        # Retrieve from engine
        results = self.engine.retrieve(
            query=query,
            tags=tags,
            limit=limit,
            use_semantic=use_semantic,
            semantic_threshold=semantic_threshold,
            enable_mmr=enable_mmr,
            enable_deduplication=True,
        )

        # Fallback to keyword search if semantic returned nothing
        if use_semantic and not results and query:
            results = self.engine.retrieve(
                query=query,
                tags=tags,
                limit=limit,
                use_semantic=False,
                enable_mmr=enable_mmr,
                enable_deduplication=True,
            )
            fallback_warning = (
                "Semantic search returned no results; fell back to keyword search"
            )

        # Apply conflict-aware reranking if enabled
        if conflict_aware:
            results = rerank_conflict_aware(results)

        # Build packet with evidence grounding
        packet = build_retrieval_packet(
            query=query,
            results=results,
        )

        # Add fallback warning if applicable
        if fallback_warning:
            packet.warnings.append(fallback_warning)

        return packet

    def ground_answer(
        self,
        query: str,
        answer: str,
        packet: RetrievalPacket,
    ) -> Dict[str, Any]:
        """Ground an answer with evidence from a retrieval packet.

        Args:
            query: The original query
            answer: The proposed answer
            packet: RetrievalPacket with evidence

        Returns:
            Dictionary with grounded answer metadata:
            - query: original query
            - answer: proposed answer
            - support_level: packet support level
            - warnings: list of warning strings
            - evidence: list of evidence reference dicts
            - should_answer: whether the answer should be given
        """
        evidence = []
        for ref in packet.evidence_refs:
            evidence.append({
                "memory_id": ref.memory_id,
                "content_hash": ref.content_hash,
                "source_refs": ref.source_refs,
                "tags": ref.tags,
                "confidence": ref.confidence,
                "salience": ref.salience,
                "stability": ref.stability,
                "volatility": ref.volatility,
                "contradiction_count": ref.contradiction_count,
                "invalidated": ref.invalidated,
            })

        # Determine if we should answer
        should_answer = packet.support_level in ("supported", "strong")

        # For mixed support, we may answer but with strong warnings
        if packet.support_level == "mixed":
            should_answer = True
            # Ensure warnings explain the mixed status
            has_explanation = any(
                "contradiction" in w.lower() or "volatil" in w.lower()
                for w in packet.warnings
            )
            if not has_explanation:
                packet.warnings.append(
                    "Support level is mixed due to contradictions or volatility"
                )

        # Never answer with no support
        if packet.support_level == "none":
            should_answer = False

        return {
            "query": query,
            "answer": answer,
            "support_level": packet.support_level,
            "warnings": packet.warnings,
            "evidence": evidence,
            "should_answer": should_answer,
        }
