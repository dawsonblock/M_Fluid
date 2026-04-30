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
    select,
    update,
    insert,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncSession

from m_flow.adapters.relational.ModelBase import Base as ModelBase
from m_flow.adapters.relational.create_relational_engine import create_relational_engine
from m_flow.memory.fluid.models import FluidMemoryState


class FluidStateRecord(ModelBase):
    """SQLAlchemy model for fluid memory state table."""
    
    __tablename__ = "fluid_memory_state"
    
    node_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    activation: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    source_trust: Mapped[float] = mapped_column(Float, default=0.5)
    recency_score: Mapped[float] = mapped_column(Float, default=1.0)
    decay_rate: Mapped[float] = mapped_column(Float, default=0.01)
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
        return FluidMemoryState(
            node_id=record.node_id,
            activation=record.activation,
            confidence=record.confidence,
            source_trust=record.source_trust,
            recency_score=record.recency_score,
            decay_rate=record.decay_rate,
            reinforcement_count=record.reinforcement_count,
            contradiction_pressure=record.contradiction_pressure,
            salience=record.salience,
            user_relevance=record.user_relevance,
            legal_weight=record.legal_weight,
            last_touched_at=record.last_touched_at,
        )
    
    def _state_to_record_dict(self, state: FluidMemoryState) -> dict:
        """Convert FluidMemoryState to dict for DB insertion."""
        return {
            "node_id": state.node_id,
            "activation": state.activation,
            "confidence": state.confidence,
            "source_trust": state.source_trust,
            "recency_score": state.recency_score,
            "decay_rate": state.decay_rate,
            "reinforcement_count": state.reinforcement_count,
            "contradiction_pressure": state.contradiction_pressure,
            "salience": state.salience,
            "user_relevance": state.user_relevance,
            "legal_weight": state.legal_weight,
            "last_touched_at": state.last_touched_at,
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
