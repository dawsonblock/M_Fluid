"""
Fluid Memory Timeline

Temporal indexing and compression for memory nodes.

Each node can have a sequence of TimelineEvents — court hearings, media mentions,
entity updates, arrests, filings, etc.  Events can be compressed when multiple
events within a merge_window_days share the same event_type.

This is especially useful for JudgeTracker / crime mapping systems where
the same judge or entity has thousands of events over years, and retrieving
a compressed timeline view is faster and more useful than raw event lists.

Usage:
    tl = TimelineCompressor(store)
    await tl.add_event("judge:123", event_date=..., event_type="court_hearing", ...)
    events = await tl.get_timeline("judge:123")
    compressed = await tl.compress("judge:123", merge_window_days=7)
"""

from __future__ import annotations

import json
from datetime import datetime
from time import time
from typing import List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field
from sqlalchemy import String, Float, Integer, DateTime, Text, select, insert
from sqlalchemy.orm import Mapped, mapped_column

from m_flow.adapters.relational.ModelBase import Base as ModelBase
from m_flow.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from m_flow.memory.fluid.state_store import FluidStateStore

logger = get_logger("fluid.timeline")

_SECONDS_PER_DAY = 86400.0


class TimelineEvent(BaseModel):
    """
    A single temporal event associated with a memory node.

    event_type examples: "court_hearing", "arrest", "media_mention",
    "entity_update", "evidence_added", "filing", "verdict"
    """
    node_id: str
    event_date: float                       # unix timestamp of the real-world event
    event_type: str = "event"
    confidence: float = 0.5                # how confident we are this event occurred
    source_ids: List[str] = Field(default_factory=list)
    description: str = ""
    compressed: bool = False               # True if merged from multiple events
    compressed_count: int = 1             # how many events were merged into this one
    recorded_at: float = Field(default_factory=time)


class TimelineEventORM(ModelBase):
    """SQLAlchemy model for the fluid_timeline_events table."""

    __tablename__ = "fluid_timeline_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String(255), index=True)
    event_date: Mapped[float] = mapped_column(Float, index=True)
    event_type: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    source_ids: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    description: Mapped[str] = mapped_column(Text, default="")
    compressed: Mapped[bool] = mapped_column(Integer, default=0)            # 0/1
    compressed_count: Mapped[int] = mapped_column(Integer, default=1)
    recorded_at: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TimelineCompressor:
    """
    Temporal indexing and compression for fluid memory nodes.

    Compression merges events of the same type within a rolling window,
    reducing database size and improving retrieval performance for entities
    with long event histories (judges, repeat offenders, case files).
    """

    def __init__(self, store: "FluidStateStore") -> None:
        self._store = store

    async def add_event(
        self,
        node_id: str,
        event_date: float,
        event_type: str = "event",
        confidence: float = 0.5,
        source_ids: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        """
        Add a timeline event for a node.

        Args:
            node_id: Memory node this event belongs to
            event_date: Unix timestamp of the real-world event
            event_type: Type of event (e.g. "court_hearing", "arrest")
            confidence: Confidence that this event occurred [0, 1]
            source_ids: Source document IDs that record this event
            description: Human-readable event description
        """
        try:
            await self._store._ensure_tables()
            async with self._store._engine.sessionmaker() as session:
                await session.execute(
                    insert(TimelineEventORM).values(
                        node_id=node_id,
                        event_date=event_date,
                        event_type=event_type,
                        confidence=confidence,
                        source_ids=json.dumps(source_ids or []),
                        description=description,
                        compressed=0,
                        compressed_count=1,
                        recorded_at=time(),
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.warning("timeline.add_event failed for %s: %s", node_id, exc)

    async def get_timeline(
        self,
        node_id: str,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 500,
    ) -> List[TimelineEvent]:
        """
        Get all timeline events for a node, ordered by event_date ascending.

        Args:
            node_id: Memory node to look up
            event_type: Filter by event type (optional)
            since: Only return events after this unix timestamp (optional)
            limit: Maximum number of events to return

        Returns:
            List of TimelineEvent ordered by event_date
        """
        try:
            await self._store._ensure_tables()
            async with self._store._engine.sessionmaker() as session:
                query = (
                    select(TimelineEventORM)
                    .where(TimelineEventORM.node_id == node_id)
                    .order_by(TimelineEventORM.event_date.asc())
                    .limit(limit)
                )
                if event_type:
                    query = query.where(TimelineEventORM.event_type == event_type)
                if since is not None:
                    query = query.where(TimelineEventORM.event_date >= since)

                result = await session.execute(query)
                rows = result.scalars().all()
                return [self._orm_to_model(r) for r in rows]
        except Exception as exc:
            logger.warning("timeline.get_timeline failed for %s: %s", node_id, exc)
            return []

    async def compress(
        self,
        node_id: str,
        merge_window_days: float = 1.0,
    ) -> List[TimelineEvent]:
        """
        Return a compressed view of the timeline by merging events of the
        same type within merge_window_days of each other.

        Compression is done in-memory and returned; the raw events are preserved
        in the database (lossless — original events are never deleted).

        Merging logic:
        - Events of the same type within merge_window_days are grouped.
        - The merged event takes the earliest event_date, the average confidence,
          and the union of source_ids.
        - compressed=True and compressed_count=N are set on the merged event.

        Args:
            node_id: Memory node to compress
            merge_window_days: Events within this window are merged (default 1 day)

        Returns:
            List of compressed TimelineEvent (may be shorter than raw timeline)
        """
        events = await self.get_timeline(node_id)
        if not events:
            return []

        merge_window_seconds = merge_window_days * _SECONDS_PER_DAY
        compressed: List[TimelineEvent] = []
        current_group: Optional[List[TimelineEvent]] = None

        for event in events:
            if current_group is None:
                current_group = [event]
                continue

            last = current_group[-1]
            same_type = event.event_type == last.event_type
            within_window = (event.event_date - last.event_date) <= merge_window_seconds

            if same_type and within_window:
                current_group.append(event)
            else:
                compressed.append(self._merge_group(current_group))
                current_group = [event]

        if current_group:
            compressed.append(self._merge_group(current_group))

        return compressed

    def _merge_group(self, group: List[TimelineEvent]) -> TimelineEvent:
        """Merge a group of events into a single compressed event."""
        if len(group) == 1:
            return group[0]

        all_sources: list = []
        for e in group:
            all_sources.extend(e.source_ids)

        avg_confidence = sum(e.confidence for e in group) / len(group)

        return TimelineEvent(
            node_id=group[0].node_id,
            event_date=group[0].event_date,
            event_type=group[0].event_type,
            confidence=round(avg_confidence, 4),
            source_ids=list(set(all_sources)),
            description=f"[compressed: {len(group)} events]",
            compressed=True,
            compressed_count=len(group),
            recorded_at=group[0].recorded_at,
        )

    def _orm_to_model(self, row: TimelineEventORM) -> TimelineEvent:
        """Convert ORM row to TimelineEvent model."""
        source_ids: List[str] = []
        if row.source_ids:
            try:
                source_ids = json.loads(row.source_ids)
            except Exception:
                source_ids = []
        return TimelineEvent(
            node_id=row.node_id,
            event_date=row.event_date,
            event_type=row.event_type,
            confidence=row.confidence,
            source_ids=source_ids,
            description=row.description or "",
            compressed=bool(row.compressed),
            compressed_count=row.compressed_count,
            recorded_at=row.recorded_at,
        )
