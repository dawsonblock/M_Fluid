"""Test that fluid adapter creates events with correct contract.

This test verifies that when fluid memory is enabled, the adapter
produces events with the correct field names matching the real
FluidUpdateEvent contract from m_flow.memory.fluid.models.
"""
import sys
import types
from typing import List, Optional

import pytest

from judge_memory import JudgeMemoryConfig
from judge_memory.fluid_adapter import FluidMemoryAdapter


class FakeFluidUpdateEvent:
    """Fake event that matches the real FluidUpdateEvent structure."""

    def __init__(
        self,
        touched_node_ids: List[str],
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        source_trust: float = 0.5,
        salience: float = 0.5,
        legal_weight: float = 0.0,
        decay_lane: str = "interest",
        supports: Optional[List[str]] = None,
        contradicts: Optional[List[str]] = None,
        jurisdiction: Optional[str] = None,
        judge_id: Optional[str] = None,
        geographic_scope: Optional[str] = None,
        event_confidence: float = 0.5,
        parent_source_ids: Optional[List[str]] = None,
    ):
        self.touched_node_ids = touched_node_ids
        self.source_id = source_id
        self.source_type = source_type
        self.source_trust = source_trust
        self.salience = salience
        self.legal_weight = legal_weight
        self.decay_lane = decay_lane
        self.supports = supports or []
        self.contradicts = contradicts or []
        self.jurisdiction = jurisdiction
        self.judge_id = judge_id
        self.geographic_scope = geographic_scope
        self.event_confidence = event_confidence
        self.parent_source_ids = parent_source_ids or []


class FakeEngine:
    """Fake engine that captures events."""

    def __init__(self):
        self.events = []

    async def touch(self, event):
        self.events.append(event)


@pytest.fixture(autouse=True)
def fake_m_flow_fluid_modules(monkeypatch):
    """Inject fake m_flow modules so adapter can import without real m_flow."""
    m_flow = types.ModuleType("m_flow")
    memory = types.ModuleType("m_flow.memory")
    fluid = types.ModuleType("m_flow.memory.fluid")
    models = types.ModuleType("m_flow.memory.fluid.models")
    models.FluidUpdateEvent = FakeFluidUpdateEvent

    monkeypatch.setitem(sys.modules, "m_flow", m_flow)
    monkeypatch.setitem(sys.modules, "m_flow.memory", memory)
    monkeypatch.setitem(sys.modules, "m_flow.memory.fluid", fluid)
    monkeypatch.setitem(sys.modules, "m_flow.memory.fluid.models", models)


@pytest.mark.asyncio
async def test_touch_evidence_uses_real_fluid_event_contract(tmp_path):
    """Test that touch_evidence creates proper FluidUpdateEvent."""
    adapter = FluidMemoryAdapter(
        JudgeMemoryConfig(
            data_dir=str(tmp_path),
            enable_fluid_memory=False,
        )
    )
    adapter.enabled = True
    adapter._engine = FakeEngine()

    await adapter.touch_evidence(
        evidence_id="ev_test",
        source_type="court_record",
        jurisdiction="CA-SK",
    )
    assert len(adapter._engine.events) == 1
    event = adapter._engine.events[0]
    assert event.touched_node_ids == ["ev_test"]
    assert event.source_id == "ev_test"
    assert event.source_type == "court_record"
    assert event.decay_lane == "legal"
    assert event.jurisdiction == "CA-SK"
    assert event.source_trust > 0.5
    assert event.legal_weight > 0.5


@pytest.mark.asyncio
async def test_touch_claim_uses_real_fluid_event_contract(tmp_path):
    """Test that touch_claim creates proper FluidUpdateEvent."""
    adapter = FluidMemoryAdapter(
        JudgeMemoryConfig(
            data_dir=str(tmp_path),
            enable_fluid_memory=False,
        )
    )
    adapter.enabled = True
    adapter._engine = FakeEngine()

    await adapter.touch_claim(
        claim_id="claim_test",
        evidence_id="ev_test",
        source_type="court_record",
        claim_status="confirmed",
        jurisdiction="CA-SK",
    )
    assert len(adapter._engine.events) == 1
    event = adapter._engine.events[0]
    assert event.touched_node_ids == ["claim_test"]
    assert event.source_id == "ev_test"
    assert event.source_type == "court_record"
    assert event.decay_lane == "legal"
    assert event.jurisdiction == "CA-SK"
    assert event.supports == ["ev_test"]
    assert event.salience == 0.6
