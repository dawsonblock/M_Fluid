"""
Fluid Memory Audit

Audit logging for all fluid state mutations.
Provides traceability for how and why fluid state changed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    String,
    Float,
    DateTime,
    Integer,
    select,
    insert,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncSession

from m_flow.adapters.relational.ModelBase import Base as ModelBase
from m_flow.memory.fluid.state_store import FluidStateStore


class AuditEventType(str, Enum):
    """Types of fluid memory audit events."""
    TOUCH = "touch"           # Node was touched (activated)
    DECAY = "decay"           # Activation decayed over time
    PROPAGATE = "propagate"   # Activation propagated from another node
    CONTRADICT = "contradict" # Contradiction pressure applied
    MERGE = "merge"           # State merged/updated
    MANUAL = "manual"         # Manual admin intervention
    DELETE = "delete"         # State deleted


class FluidAuditRecord(ModelBase):
    """SQLAlchemy model for fluid memory audit log."""
    
    __tablename__ = "fluid_audit_log"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    event_type: Mapped[str] = mapped_column(String(50))  # AuditEventType value
    node_id: Mapped[str] = mapped_column(String(255), index=True)
    field_changed: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    old_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    new_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # Additional metadata stored as JSON string
    metadata_json: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)


class FluidAuditLogger:
    """
    Audit logger for fluid memory operations.
    
    Records all state mutations for debugging and compliance.
    """
    
    def __init__(self, store: FluidStateStore):
        self.store = store
    
    async def log_event(
        self,
        event_type: AuditEventType,
        node_id: str,
        field_changed: Optional[str] = None,
        old_value: Optional[float] = None,
        new_value: Optional[float] = None,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a generic audit event.
        
        Args:
            event_type: Type of event
            node_id: Affected node ID
            field_changed: Field that was modified
            old_value: Previous value
            new_value: New value
            source_id: Source document/node that triggered change
            source_type: Type of source
            metadata: Additional JSON-serializable metadata
        """
        import json
        
        async with self.store._engine.sessionmaker() as session:
            record = FluidAuditRecord(
                event_type=event_type.value,
                node_id=node_id,
                field_changed=field_changed,
                old_value=old_value,
                new_value=new_value,
                source_id=source_id,
                source_type=source_type,
                metadata_json=json.dumps(metadata) if metadata else None,
            )
            session.add(record)
            await session.commit()
    
    async def log_touch(
        self,
        node_id: str,
        new_activation: float,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> None:
        """Log a touch event (node activation)."""
        await self.log_event(
            event_type=AuditEventType.TOUCH,
            node_id=node_id,
            field_changed="activation",
            new_value=new_activation,
            source_id=source_id,
            source_type=source_type,
        )
    
    async def log_decay(
        self,
        node_id: str,
        new_activation: float,
        old_activation: Optional[float] = None,
    ) -> None:
        """Log a decay event."""
        await self.log_event(
            event_type=AuditEventType.DECAY,
            node_id=node_id,
            field_changed="activation",
            old_value=old_activation,
            new_value=new_activation,
        )
    
    async def log_propagation(
        self,
        node_id: str,
        old_activation: float,
        new_activation: float,
        source_nodes: List[str],
    ) -> None:
        """Log an activation propagation event."""
        await self.log_event(
            event_type=AuditEventType.PROPAGATE,
            node_id=node_id,
            field_changed="activation",
            old_value=old_activation,
            new_value=new_activation,
            metadata={"source_nodes": source_nodes},
        )
    
    async def log_contradiction(
        self,
        node_id: str,
        new_pressure: float,
        contradicting_source: str,
    ) -> None:
        """Log a contradiction pressure event."""
        await self.log_event(
            event_type=AuditEventType.CONTRADICT,
            node_id=node_id,
            field_changed="contradiction_pressure",
            new_value=new_pressure,
            source_id=contradicting_source,
        )
    
    async def get_audit_log(
        self,
        node_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query audit log.
        
        Args:
            node_id: Filter by node (optional)
            event_type: Filter by event type (optional)
            limit: Maximum records to return
            
        Returns:
            List of audit records as dicts
        """
        async with self.store._engine.sessionmaker() as session:
            query = select(FluidAuditRecord)
            
            if node_id:
                query = query.where(FluidAuditRecord.node_id == node_id)
            
            if event_type:
                query = query.where(FluidAuditRecord.event_type == event_type.value)
            
            query = query.order_by(FluidAuditRecord.timestamp.desc()).limit(limit)
            
            result = await session.execute(query)
            records = result.scalars().all()
            
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat(),
                    "event_type": r.event_type,
                    "node_id": r.node_id,
                    "field_changed": r.field_changed,
                    "old_value": r.old_value,
                    "new_value": r.new_value,
                    "source_id": r.source_id,
                    "source_type": r.source_type,
                    "metadata": r.metadata_json,
                }
                for r in records
            ]
    
    async def get_node_history(
        self,
        node_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get complete audit history for a node.
        
        Args:
            node_id: Node to query
            
        Returns:
            List of all audit events for the node
        """
        return await self.get_audit_log(node_id=node_id, limit=1000)
