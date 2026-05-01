"""
Fluid Memory Contradiction Detector

LLM-assisted detection of semantic conflicts between memory nodes.

When a new node is written, this detector queries existing nodes that
share entities with the new node and asks the LLM whether the claims
conflict.  Detected conflicts are persisted in fluid_claim_conflicts
and increase contradiction_pressure on both nodes.

Design principles:
- Uses M-flow's existing LLMService — no new LLM dependency.
- Gated by FluidMemoryConfig.enable_contradiction.
- Falls back gracefully if LLM is unavailable (logs, does not crash).
- Contradiction affects retrieval priority only — it never deletes evidence.
"""

from __future__ import annotations

from time import time
from typing import Optional, List, TYPE_CHECKING

from pydantic import BaseModel

from m_flow.memory.fluid.models import ClaimConflict

if TYPE_CHECKING:
    from m_flow.adapters.graph.graph_db_interface import GraphProvider


# ---------------------------------------------------------------------------
# LLM response model
# ---------------------------------------------------------------------------

class ConflictResult(BaseModel):
    """Structured LLM response for conflict detection."""
    conflicts: bool
    confidence: float           # [0.0 – 1.0]
    reason: str               # Short explanation


# Conflict status for ClaimConflict records
CONFLICT_STATUS_CONFIRMED = "confirmed_conflict"
CONFLICT_STATUS_POSSIBLE = "possible_conflict"
CONFLICT_STATUS_NEEDS_REVIEW = "needs_review"
CONFLICT_STATUS_RESOLVED = "resolved"
CONFLICT_STATUS_REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_CONFLICT_PROMPT = """\
You are a fact-checking assistant for a legal evidence system.

You will be given two memory claims extracted from different sources.
Determine whether they are in direct factual conflict.

A conflict exists when:
- Both claims describe the same subject and attribute
- The claims assert different, mutually exclusive values/facts
- The difference is substantive (not just wording)

Do NOT flag as conflict:
- Claims about different subjects or time periods
- Claims that are complementary or additive
- Ambiguous statements

Respond with:
  conflicts: true or false
  confidence: 0.0 to 1.0 (your certainty)
  reason: one sentence explaining your decision

Claim A (source: {source_a}):
{text_a}

Claim B (source: {source_b}):
{text_b}
"""


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class ContradictionDetector:
    """
    LLM-assisted contradiction detector.

    Usage:
        detector = ContradictionDetector(graph_engine)
        conflicts = await detector.detect(
            new_node_id="ep:123",
            new_node_text="Maria was present at the meeting on Monday.",
            new_source_id="doc:abc",
            new_source_type="police_release",
            candidate_node_ids=["ep:99", "ep:101"],
        )
    """

    def __init__(
        self,
        graph_engine: "GraphProvider",
        llm_model: Optional[str] = None,
    ) -> None:
        self.graph = graph_engine
        self._llm_model = llm_model
        self._logger = _get_logger()

    async def detect(
        self,
        new_node_id: str,
        new_node_text: str,
        new_source_id: Optional[str],
        new_source_type: Optional[str],
        candidate_node_ids: List[str],
    ) -> List[ClaimConflict]:
        """
        Compare a new node's text against candidate nodes for conflicts.

        Uses structured gate before LLM: only checks candidates that share
        a strong connection (same entity, case, judge, jurisdiction).

        Args:
            new_node_id: ID of the newly written node
            new_node_text: Summary/text content of the new node
            new_source_id: Source document ID for the new node
            new_source_type: Source type for the new node
            candidate_node_ids: Existing node IDs to compare against

        Returns:
            List of ClaimConflict records for detected conflicts
        """
        from m_flow.memory.fluid.config import get_fluid_config

        cfg = get_fluid_config()

        # Check if LLM contradiction is enabled
        if not cfg.enable_llm_contradiction:
            self._logger.debug("fluid.contradiction: LLM contradiction disabled by config")
            return []

        if not candidate_node_ids or not new_node_text.strip():
            return []

        # Get min confidence threshold from config
        min_confidence = cfg.min_llm_contradiction_confidence

        conflicts: List[ClaimConflict] = []

        for cand_id in candidate_node_ids[:5]:  # cap at 5 candidates per call
            # Structured gate: check if nodes share meaningful connection
            if cfg.structured_contradiction_required:
                passes_gate = await self._check_structured_gate(new_node_id, cand_id)
                if not passes_gate:
                    self._logger.debug(
                        "fluid.contradiction: structured gate failed for %s vs %s",
                        new_node_id, cand_id,
                    )
                    continue

            cand_text = await self._fetch_node_text(cand_id)
            if not cand_text:
                continue

            result = await self._run_conflict_check(
                text_a=new_node_text,
                source_a=new_source_type or "unknown",
                text_b=cand_text,
                source_b="existing",
            )

            if result and result.conflicts and result.confidence >= min_confidence:
                # Determine conflict status based on confidence
                if result.confidence >= 0.85:
                    status = CONFLICT_STATUS_CONFIRMED
                elif result.confidence >= 0.70:
                    status = CONFLICT_STATUS_POSSIBLE
                else:
                    status = CONFLICT_STATUS_NEEDS_REVIEW

                conflicts.append(ClaimConflict(
                    node_id_a=new_node_id,
                    node_id_b=cand_id,
                    source_id_a=new_source_id,
                    source_id_b=None,
                    conflict_reason=result.reason,
                    confidence=result.confidence,
                    detected_at=time(),
                    conflict_status=status,
                ))

        return conflicts

    async def _check_structured_gate(self, node_id_a: str, node_id_b: str) -> bool:
        """
        Check if two nodes share a meaningful connection for contradiction.

        Returns True if nodes share:
        - Same entity/person id
        - Same case id
        - Same event id
        - Same judge id
        - Same jurisdiction + material attribute
        - Same source cluster / citation group
        """
        from m_flow.memory.fluid.graph_access import get_connected_nodes, row_get

        try:
            # Get connected entities for both nodes
            conn_a = await get_connected_nodes(self.graph, node_id_a)
            conn_b = await get_connected_nodes(self.graph, node_id_b)

            # Build sets of connected entity IDs by edge type
            entities_a = set()
            entities_b = set()

            for nid, etype, _ in conn_a:
                if etype in ("involves_entity", "has_facet", "about_entity"):
                    entities_a.add(nid)

            for nid, etype, _ in conn_b:
                if etype in ("involves_entity", "has_facet", "about_entity"):
                    entities_b.add(nid)

            # If they share any entity, they pass the gate
            if entities_a & entities_b:
                return True

            # Check for case/judge/event connections
            cases_a = {nid for nid, etype, _ in conn_a if "case" in etype.lower()}
            cases_b = {nid for nid, etype, _ in conn_b if "case" in etype.lower()}
            if cases_a & cases_b:
                return True

            judges_a = {nid for nid, etype, _ in conn_a if "judge" in etype.lower()}
            judges_b = {nid for nid, etype, _ in conn_b if "judge" in etype.lower()}
            if judges_a & judges_b:
                return True

            events_a = {nid for nid, etype, _ in conn_a if "event" in etype.lower()}
            events_b = {nid for nid, etype, _ in conn_b if "event" in etype.lower()}
            if events_a & events_b:
                return True

            # No shared connection found
            return False

        except Exception as exc:
            self._logger.debug("fluid.contradiction: structured gate error: %s", exc)
            # If gate check fails, allow through (fail open for safety)
            return True

    async def _fetch_node_text(self, node_id: str) -> Optional[str]:
        """Fetch the summary/text of a node from the graph using graph_access helper."""
        from m_flow.memory.fluid.graph_access import get_node_text
        try:
            return await get_node_text(self.graph, node_id)
        except Exception as exc:
            self._logger.debug(
                "fluid.contradiction: graph fetch failed for %s: %s (%s)",
                node_id, type(exc).__name__, exc,
            )
            return None

    async def _run_conflict_check(
        self,
        text_a: str,
        source_a: str,
        text_b: str,
        source_b: str,
    ) -> Optional[ConflictResult]:
        """Call LLM to check for conflict between two texts."""
        try:
            from m_flow.llm.LLMGateway import LLMService

            prompt = _CONFLICT_PROMPT.format(
                source_a=source_a,
                text_a=text_a[:800],
                source_b=source_b,
                text_b=text_b[:800],
            )
            result: ConflictResult = await LLMService.extract_structured(
                text_input=prompt,
                system_prompt=(
                    "You are a fact-checking assistant. "
                    "Respond only with the structured JSON output requested."
                ),
                response_model=ConflictResult,
            )
            return result
        except Exception as exc:
            self._logger.debug("fluid.contradiction: LLM check failed: %s", exc)
            return None


def _get_logger():
    from m_flow.shared.logging_utils import get_logger
    return get_logger("fluid.contradiction")
