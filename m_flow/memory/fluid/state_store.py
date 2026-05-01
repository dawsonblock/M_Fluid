"""
Fluid State Store

SQLAlchemy-based storage for fluid memory state.
Supports both SQLite and PostgreSQL backends.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass

from sqlalchemy import (
    String,
    Float,
    Integer,
    DateTime,
    Text,
    select,
    update,
    insert,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncSession

from m_flow.adapters.relational.ModelBase import Base as ModelBase
from m_flow.adapters.relational.create_relational_engine import create_relational_engine
from m_flow.memory.fluid.models import (
    FluidMemoryState,
    SourceLineageRecord,
    MediaAmplificationEvent,
)


class FluidStateRecord(ModelBase):
    """SQLAlchemy model for fluid memory state table."""

    __tablename__ = "fluid_memory_state"

    node_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    activation: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    source_trust: Mapped[float] = mapped_column(Float, default=0.5)
    recency_score: Mapped[float] = mapped_column(Float, default=1.0)
    decay_rate: Mapped[float] = mapped_column(Float, default=0.05)  # per-day; INTEREST_DECAY
    decay_lane: Mapped[str] = mapped_column(String(20), default="interest")
    reinforcement_count: Mapped[int] = mapped_column(Integer, default=0)
    contradiction_pressure: Mapped[float] = mapped_column(Float, default=0.0)
    salience: Mapped[float] = mapped_column(Float, default=0.5)
    user_relevance: Mapped[float] = mapped_column(Float, default=0.5)
    legal_weight: Mapped[float] = mapped_column(Float, default=0.0)
    last_touched_at: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Provenance / lineage
    source_lineage: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    media_amplification: Mapped[float] = mapped_column(Float, default=0.0)

    # Legal / jurisdiction (JudgeTracker)
    jurisdiction: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    judge_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    event_confidence: Mapped[float] = mapped_column(Float, default=0.5)

    # Geographic
    geographic_scope: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Contradiction cluster
    contradiction_cluster_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)


class ClaimConflictRecord(ModelBase):
    """SQLAlchemy model for detected claim conflicts."""

    __tablename__ = "fluid_claim_conflicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id_a: Mapped[str] = mapped_column(String(255), index=True)
    node_id_b: Mapped[str] = mapped_column(String(255), index=True)
    source_id_a: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_id_b: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    conflict_reason: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    detected_at: Mapped[float] = mapped_column(Float)
    contradiction_cluster_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SourceLineageORM(ModelBase):
    """SQLAlchemy model for source lineage / citation graph edges."""

    __tablename__ = "fluid_source_lineage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String(255), index=True)
    parent_source_id: Mapped[str] = mapped_column(String(255))
    child_source_id: Mapped[str] = mapped_column(String(255))
    relationship: Mapped[str] = mapped_column(String(30), default="cites")
    recorded_at: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MediaAmplificationORM(ModelBase):
    """SQLAlchemy model for media amplification / duplicate collapse events."""

    __tablename__ = "fluid_media_amplification"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String(255), index=True)
    canonical_source_id: Mapped[str] = mapped_column(String(255))
    duplicate_source_ids: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    amplification_factor: Mapped[float] = mapped_column(Float, default=0.0)
    detected_at: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


@dataclass
class ActivationMergeUpdate:
    """Result of activation merge operation."""
    node_id: str
    old_activation: float
    new_activation: float
    merged: bool


class FluidStateStore:
    """
    Store for fluid memory state.
    
    Provides async CRUD operations for fluid state records,
    with support for batch operations and activation merging.
    """
    
    def __init__(
        self,
        db_provider: str = "sqlite",
        db_path: str = "",
        db_name: str = "fluid_memory",
        db_host: str = "",
        db_port: str = "",
        db_username: str = "",
        db_password: str = "",
    ):
        self._engine = create_relational_engine(
            db_path=db_path,
            db_name=db_name,
            db_host=db_host,
            db_port=db_port,
            db_username=db_username,
            db_password=db_password,
            db_provider=db_provider,
        )
        self._initialized = False
    
    async def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        if self._initialized:
            return
        async with self._engine.engine.begin() as conn:
            await conn.run_sync(ModelBase.metadata.create_all)
        self._initialized = True
    
    def _record_to_state(self, record: FluidStateRecord) -> FluidMemoryState:
        """Convert DB record to FluidMemoryState model."""
        import json
        lineage: List[str] = []
        if record.source_lineage:
            try:
                lineage = json.loads(record.source_lineage)
            except Exception:
                lineage = []
        return FluidMemoryState(
            node_id=record.node_id,
            activation=record.activation,
            confidence=record.confidence,
            source_trust=record.source_trust,
            recency_score=record.recency_score,
            decay_rate=record.decay_rate,
            decay_lane=record.decay_lane,
            reinforcement_count=record.reinforcement_count,
            contradiction_pressure=record.contradiction_pressure,
            salience=record.salience,
            user_relevance=record.user_relevance,
            legal_weight=record.legal_weight,
            last_touched_at=record.last_touched_at,
            source_lineage=lineage,
            media_amplification=record.media_amplification or 0.0,
            jurisdiction=record.jurisdiction,
            judge_id=record.judge_id,
            event_confidence=record.event_confidence or 0.5,
            geographic_scope=record.geographic_scope,
            contradiction_cluster_id=record.contradiction_cluster_id,
        )
    
    def _state_to_record_dict(self, state: FluidMemoryState) -> dict:
        """Convert FluidMemoryState to dict for DB insertion."""
        import json
        return {
            "node_id": state.node_id,
            "activation": state.activation,
            "confidence": state.confidence,
            "source_trust": state.source_trust,
            "recency_score": state.recency_score,
            "decay_rate": state.decay_rate,
            "decay_lane": state.decay_lane,
            "reinforcement_count": state.reinforcement_count,
            "contradiction_pressure": state.contradiction_pressure,
            "salience": state.salience,
            "user_relevance": state.user_relevance,
            "legal_weight": state.legal_weight,
            "last_touched_at": state.last_touched_at,
            "source_lineage": json.dumps(state.source_lineage) if state.source_lineage else None,
            "media_amplification": state.media_amplification,
            "jurisdiction": state.jurisdiction,
            "judge_id": state.judge_id,
            "event_confidence": state.event_confidence,
            "geographic_scope": state.geographic_scope,
            "contradiction_cluster_id": state.contradiction_cluster_id,
        }
    
    async def get(self, node_id: str) -> Optional[FluidMemoryState]:
        """Get fluid state for a single node."""
        await self._ensure_tables()
        async with self._engine.sessionmaker() as session:
            result = await session.execute(
                select(FluidStateRecord).where(FluidStateRecord.node_id == node_id)
            )
            record = result.scalar_one_or_none()
            return self._record_to_state(record) if record else None
    
    async def get_many(self, node_ids: List[str]) -> List[FluidMemoryState]:
        """Get fluid state for multiple nodes."""
        await self._ensure_tables()
        if not node_ids:
            return []
        
        async with self._engine.sessionmaker() as session:
            result = await session.execute(
                select(FluidStateRecord).where(FluidStateRecord.node_id.in_(node_ids))
            )
            records = result.scalars().all()
            existing = {r.node_id: r for r in records}
            
            # Return states in order, creating defaults for missing nodes
            states = []
            for node_id in node_ids:
                if node_id in existing:
                    states.append(self._record_to_state(existing[node_id]))
                else:
                    states.append(FluidMemoryState(node_id=node_id))
            return states
    
    async def upsert(self, state: FluidMemoryState) -> None:
        """Insert or update fluid state for a single node."""
        await self._ensure_tables()
        async with self._engine.sessionmaker() as session:
            data = self._state_to_record_dict(state)
            data["updated_at"] = datetime.utcnow()
            
            # Try update first
            result = await session.execute(
                update(FluidStateRecord)
                .where(FluidStateRecord.node_id == state.node_id)
                .values(**data)
            )
            
            # If no rows updated, insert
            if result.rowcount == 0:
                data["created_at"] = datetime.utcnow()
                await session.execute(insert(FluidStateRecord).values(**data))
            
            await session.commit()
    
    async def upsert_many(self, states: List[FluidMemoryState]) -> None:
        """Batch insert or update fluid states."""
        await self._ensure_tables()
        if not states:
            return
        
        async with self._engine.sessionmaker() as session:
            for state in states:
                data = self._state_to_record_dict(state)
                data["updated_at"] = datetime.utcnow()
                
                result = await session.execute(
                    update(FluidStateRecord)
                    .where(FluidStateRecord.node_id == state.node_id)
                    .values(**data)
                )
                
                if result.rowcount == 0:
                    data["created_at"] = datetime.utcnow()
                    await session.execute(insert(FluidStateRecord).values(**data))
            
            await session.commit()
    
    async def merge_activation(
        self, 
        updates: Dict[str, float],
        merge_mode: str = "max"
    ) -> List[ActivationMergeUpdate]:
        """
        Merge activation values into existing states.
        
        Args:
            updates: Dict mapping node_id -> activation value
            merge_mode: How to merge - "max" (default), "add", or "replace"
            
        Returns:
            List of merge results with old/new values
        """
        await self._ensure_tables()
        if not updates:
            return []
        
        results = []
        node_ids = list(updates.keys())
        
        async with self._engine.sessionmaker() as session:
            # Get existing records
            result = await session.execute(
                select(FluidStateRecord).where(FluidStateRecord.node_id.in_(node_ids))
            )
            existing = {r.node_id: r for r in result.scalars().all()}
            
            for node_id, new_activation in updates.items():
                if node_id in existing:
                    record = existing[node_id]
                    old_activation = record.activation
                    
                    if merge_mode == "max":
                        merged_activation = max(old_activation, new_activation)
                    elif merge_mode == "add":
                        merged_activation = min(1.0, old_activation + new_activation)
                    else:  # replace
                        merged_activation = new_activation
                    
                    record.activation = merged_activation
                    record.updated_at = datetime.utcnow()
                    
                    results.append(ActivationMergeUpdate(
                        node_id=node_id,
                        old_activation=old_activation,
                        new_activation=merged_activation,
                        merged=True
                    ))
                else:
                    # Create new record with activation
                    state = FluidMemoryState(
                        node_id=node_id,
                        activation=new_activation,
                    )
                    data = self._state_to_record_dict(state)
                    data["created_at"] = datetime.utcnow()
                    data["updated_at"] = datetime.utcnow()
                    await session.execute(insert(FluidStateRecord).values(**data))
                    
                    results.append(ActivationMergeUpdate(
                        node_id=node_id,
                        old_activation=0.0,
                        new_activation=new_activation,
                        merged=False
                    ))
            
            await session.commit()
        
        return results
    
    async def delete(self, node_id: str) -> bool:
        """Delete fluid state for a node. Returns True if deleted."""
        await self._ensure_tables()
        async with self._engine.sessionmaker() as session:
            result = await session.execute(
                select(FluidStateRecord).where(FluidStateRecord.node_id == node_id)
            )
            record = result.scalar_one_or_none()
            if record:
                await session.delete(record)
                await session.commit()
                return True
            return False
    
    async def get_all_with_activation_above(self, threshold: float = 0.0) -> List[FluidMemoryState]:
        """Get all nodes with activation above threshold."""
        await self._ensure_tables()
        async with self._engine.sessionmaker() as session:
            result = await session.execute(
                select(FluidStateRecord)
                .where(FluidStateRecord.activation > threshold)
                .order_by(FluidStateRecord.activation.desc())
            )
            records = result.scalars().all()
            return [self._record_to_state(r) for r in records]

    async def save_claim_conflict(self, conflict) -> None:
        """
        Persist a ClaimConflict to the fluid_claim_conflicts table.

        Args:
            conflict: ClaimConflict model instance
        """
        await self._ensure_tables()
        async with self._engine.sessionmaker() as session:
            await session.execute(
                insert(ClaimConflictRecord).values(
                    node_id_a=conflict.node_id_a,
                    node_id_b=conflict.node_id_b,
                    source_id_a=conflict.source_id_a,
                    source_id_b=conflict.source_id_b,
                    conflict_reason=conflict.conflict_reason,
                    confidence=conflict.confidence,
                    detected_at=conflict.detected_at,
                    contradiction_cluster_id=getattr(conflict, "contradiction_cluster_id", None),
                )
            )
            await session.commit()

    async def save_lineage(self, record: SourceLineageRecord) -> None:
        """
        Persist a source lineage edge to the fluid_source_lineage table.

        Args:
            record: SourceLineageRecord model instance
        """
        await self._ensure_tables()
        async with self._engine.sessionmaker() as session:
            await session.execute(
                insert(SourceLineageORM).values(
                    node_id=record.node_id,
                    parent_source_id=record.parent_source_id,
                    child_source_id=record.child_source_id,
                    relationship=record.relationship,
                    recorded_at=record.recorded_at,
                )
            )
            await session.commit()

    async def get_lineage(self, node_id: str) -> List[SourceLineageRecord]:
        """Get all lineage records for a node."""
        await self._ensure_tables()
        async with self._engine.sessionmaker() as session:
            result = await session.execute(
                select(SourceLineageORM).where(SourceLineageORM.node_id == node_id)
            )
            rows = result.scalars().all()
            return [
                SourceLineageRecord(
                    node_id=r.node_id,
                    parent_source_id=r.parent_source_id,
                    child_source_id=r.child_source_id,
                    relationship=r.relationship,
                    recorded_at=r.recorded_at,
                )
                for r in rows
            ]

    async def save_media_amplification(self, event: MediaAmplificationEvent) -> None:
        """
        Persist a MediaAmplificationEvent to the fluid_media_amplification table.

        Args:
            event: MediaAmplificationEvent model instance
        """
        import json
        await self._ensure_tables()
        async with self._engine.sessionmaker() as session:
            await session.execute(
                insert(MediaAmplificationORM).values(
                    node_id=event.node_id,
                    canonical_source_id=event.canonical_source_id,
                    duplicate_source_ids=json.dumps(event.duplicate_source_ids),
                    amplification_factor=event.amplification_factor,
                    detected_at=event.detected_at,
                )
            )
            await session.commit()
