"""
Fluid Memory Citation Graph

Tracks source citation, confirmation, contradiction, and amplification
relationships between memory nodes.

This is the evidence chain layer — it records *which sources contributed
to which nodes* and *how* (cite, confirm, contradict, amplify).

Cross-confirmation score: how many independent sources agree on a node.
A high cross-confirmation score raises event_confidence.
A node with one source and high activation may just be breaking news —
it has not been confirmed by independent sources.

Usage:
    cg = CitationGraph(store)
    await cg.add_link("ep:123", "src:abc", "src:xyz", "confirms")
    score = await cg.compute_cross_confirmation_score("ep:123")
    depth = await cg.get_citation_depth("ep:123")
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

from m_flow.shared.logging_utils import get_logger
from m_flow.memory.fluid.models import SourceLineageRecord

if TYPE_CHECKING:
    from m_flow.memory.fluid.state_store import FluidStateStore

logger = get_logger("fluid.citation_graph")

# Relationships that count as independent confirmation (not amplification/echo)
_CONFIRMING_RELATIONSHIPS = {"confirms", "cites"}
_CONTRADICTING_RELATIONSHIPS = {"contradicts"}
_AMPLIFYING_RELATIONSHIPS = {"amplifies"}


class CitationGraph:
    """
    Citation graph for fluid memory nodes.

    Provides methods to:
    - Record citation/confirmation/contradiction links between sources
    - Compute cross-confirmation scores (independent source agreement)
    - Measure citation depth (how many hops from primary evidence)
    - Identify amplification (echo/duplicate detection)
    """

    def __init__(self, store: "FluidStateStore") -> None:
        self._store = store

    async def add_link(
        self,
        node_id: str,
        parent_source_id: str,
        child_source_id: str,
        relationship: str = "cites",
    ) -> None:
        """
        Record a source lineage edge for a node.

        Args:
            node_id: Memory node this link belongs to
            parent_source_id: Source that was cited/confirmed/contradicted
            child_source_id: Source doing the citing/confirming/contradicting
            relationship: One of "cites", "confirms", "contradicts", "amplifies"
        """
        try:
            record = SourceLineageRecord(
                node_id=node_id,
                parent_source_id=parent_source_id,
                child_source_id=child_source_id,
                relationship=relationship,
            )
            await self._store.save_lineage(record)
        except Exception as exc:
            logger.warning("citation_graph.add_link failed for %s: %s", node_id, exc)

    async def get_citations(self, node_id: str) -> List[SourceLineageRecord]:
        """
        Get all lineage records for a node.

        Args:
            node_id: Memory node to look up

        Returns:
            List of SourceLineageRecord instances (may be empty)
        """
        try:
            return await self._store.get_lineage(node_id)
        except Exception as exc:
            logger.warning("citation_graph.get_citations failed for %s: %s", node_id, exc)
            return []

    async def get_citation_depth(self, node_id: str) -> int:
        """
        Get the number of distinct source documents that have cited or
        confirmed this node.

        Counts only confirming relationships (cites, confirms).
        Amplification links (duplicate articles) are excluded.

        Args:
            node_id: Memory node to measure

        Returns:
            Count of independent confirming sources
        """
        records = await self.get_citations(node_id)
        confirming = {
            r.child_source_id
            for r in records
            if r.relationship in _CONFIRMING_RELATIONSHIPS
        }
        return len(confirming)

    async def compute_cross_confirmation_score(self, node_id: str) -> float:
        """
        Compute a cross-confirmation score for a node.

        Measures how many *independent* sources confirm this node,
        normalised to [0, 1].

        Formula:
            score = min(1.0, confirming_sources / 5)

        Interpretation:
            0.0  = single unconfirmed source
            0.2  = 1 independent confirming source
            1.0  = 5+ independent confirming sources

        Amplification (duplicate articles) does NOT increase this score.
        Contradicting sources are counted separately and reduce trust
        through contradiction_pressure (handled by ContradictionDetector).

        Args:
            node_id: Memory node to evaluate

        Returns:
            Cross-confirmation score [0, 1]
        """
        records = await self.get_citations(node_id)

        confirming_sources: set = set()
        amplifying_sources: set = set()

        for r in records:
            if r.relationship in _CONFIRMING_RELATIONSHIPS:
                confirming_sources.add(r.child_source_id)
            elif r.relationship in _AMPLIFYING_RELATIONSHIPS:
                amplifying_sources.add(r.child_source_id)

        # Amplification does not count as independent confirmation
        # (same story retold by 10 outlets ≠ 10 independent confirmations)
        independent_count = len(confirming_sources - amplifying_sources)

        return min(1.0, independent_count / 5.0)

    async def compute_amplification_factor(self, node_id: str) -> float:
        """
        Compute how much media amplification (echo/duplicate detection) exists
        for this node.

        High amplification means the node is socially visible but not
        necessarily independently confirmed.

        Formula:
            factor = min(1.0, amplifying_sources / 10)

        Args:
            node_id: Memory node to evaluate

        Returns:
            Amplification factor [0, 1]
        """
        records = await self.get_citations(node_id)
        amplifying = {r.child_source_id for r in records if r.relationship in _AMPLIFYING_RELATIONSHIPS}
        return min(1.0, len(amplifying) / 10.0)

    async def get_contradicting_sources(self, node_id: str) -> List[str]:
        """
        Get source IDs that have contradicted this node.

        Args:
            node_id: Memory node to look up

        Returns:
            List of source IDs with contradicting relationship
        """
        records = await self.get_citations(node_id)
        return [
            r.child_source_id
            for r in records
            if r.relationship in _CONTRADICTING_RELATIONSHIPS
        ]
