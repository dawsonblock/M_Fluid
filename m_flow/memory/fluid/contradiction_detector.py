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
    reason: str                 # Short explanation


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

        Args:
            new_node_id: ID of the newly written node
            new_node_text: Summary/text content of the new node
            new_source_id: Source document ID for the new node
            new_source_type: Source type for the new node
            candidate_node_ids: Existing node IDs to compare against

        Returns:
            List of ClaimConflict records for detected conflicts
        """
        if not candidate_node_ids or not new_node_text.strip():
            return []

        conflicts: List[ClaimConflict] = []

        for cand_id in candidate_node_ids[:5]:  # cap at 5 candidates per call
            cand_text = await self._fetch_node_text(cand_id)
            if not cand_text:
                continue

            result = await self._run_conflict_check(
                text_a=new_node_text,
                source_a=new_source_type or "unknown",
                text_b=cand_text,
                source_b="existing",
            )

            if result and result.conflicts and result.confidence >= 0.6:
                conflicts.append(ClaimConflict(
                    node_id_a=new_node_id,
                    node_id_b=cand_id,
                    source_id_a=new_source_id,
                    source_id_b=None,
                    conflict_reason=result.reason,
                    confidence=result.confidence,
                    detected_at=time(),
                ))

        return conflicts

    async def _fetch_node_text(self, node_id: str) -> Optional[str]:
        """Fetch the summary/text of a node from the graph."""
        try:
            query = (
                "MATCH (n {id: $node_id}) "
                "RETURN coalesce(n.summary, n.search_text, n.name, '') AS text LIMIT 1"
            )
            rows = await self.graph.query(query, {"node_id": node_id})
            if rows:
                return str(rows[0].get("text", "")).strip() or None
        except Exception as exc:
            self._logger.debug("fluid.contradiction: graph fetch failed for %s: %s", node_id, exc)
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
