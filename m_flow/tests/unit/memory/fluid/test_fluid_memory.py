"""
Unit tests for Fluid Memory module.
"""

import pytest
import asyncio
from time import time, sleep

from m_flow.memory.fluid.models import (
    FluidMemoryState,
    FluidUpdateEvent,
    get_source_weights,
    SOURCE_TRUST_WEIGHTS,
)
from m_flow.memory.fluid.decay import (
    compute_decayed_activation,
    apply_decay,
    compute_recency_score,
)
from m_flow.memory.fluid.contradiction import (
    compute_contradiction_pressure,
    apply_contradictions,
)
from m_flow.memory.fluid.scoring import (
    fluid_score,
    compute_fluid_boost,
    should_boost_retrieval,
)


class TestFluidMemoryState:
    """Test FluidMemoryState model."""
    
    def test_default_values(self):
        state = FluidMemoryState(node_id="test-123")
        assert state.node_id == "test-123"
        assert state.activation == 0.0
        assert state.confidence == 0.5
        assert state.source_trust == 0.5
        assert state.decay_rate == 0.01
        assert state.reinforcement_count == 0
        assert state.contradiction_pressure == 0.0
    
    def test_custom_values(self):
        state = FluidMemoryState(
            node_id="test-456",
            activation=0.8,
            confidence=0.9,
            source_trust=0.95,
        )
        assert state.activation == 0.8
        assert state.confidence == 0.9
        assert state.source_trust == 0.95


class TestFluidUpdateEvent:
    """Test FluidUpdateEvent model."""
    
    def test_basic_event(self):
        event = FluidUpdateEvent(
            touched_node_ids=["a", "b", "c"],
            source_id="doc-123",
            source_type="mainstream_news",
            source_trust=0.60,
            salience=0.7,
            legal_weight=0.3,
        )
        assert len(event.touched_node_ids) == 3
        assert event.source_trust == 0.60
        assert event.salience == 0.7
    
    def test_contradiction_lists(self):
        event = FluidUpdateEvent(
            touched_node_ids=["a"],
            supports=["b", "c"],
            contradicts=["d"],
            source_trust=0.8,
        )
        assert event.supports == ["b", "c"]
        assert event.contradicts == ["d"]


class TestSourceWeights:
    """Test source weight functions."""
    
    def test_court_record_weights(self):
        trust, legal = get_source_weights("court_record")
        assert trust == 0.95
        assert legal == 1.00
    
    def test_police_release_weights(self):
        trust, legal = get_source_weights("police_release")
        assert trust == 0.80
        assert legal == 0.70
    
    def test_mainstream_news_weights(self):
        trust, legal = get_source_weights("mainstream_news")
        assert trust == 0.60
        assert legal == 0.30
    
    def test_unknown_source(self):
        trust, legal = get_source_weights("unknown")
        assert trust == 0.10
        assert legal == 0.00
    
    def test_none_source(self):
        trust, legal = get_source_weights(None)
        assert trust == 0.10
        assert legal == 0.00
    
    def test_case_insensitive(self):
        trust, legal = get_source_weights("COURT_RECORD")
        assert trust == 0.95


class TestDecay:
    """Test decay functions."""
    
    def test_no_decay_if_just_touched(self):
        now = time()
        state = FluidMemoryState(
            node_id="test",
            activation=0.5,
            last_touched_at=now,
            decay_rate=0.01,
        )
        decayed = compute_decayed_activation(
            state.activation,
            state.last_touched_at,
            state.decay_rate,
            now=now,
        )
        assert decayed == 0.5  # No decay if just touched
    
    def test_decay_over_time(self):
        now = 1000.0
        last_touched = 900.0  # 100 units ago
        decayed = compute_decayed_activation(
            current_activation=0.5,
            last_touched=last_touched,
            decay_rate=0.01,
            now=now,
        )
        # Should decay: 0.5 * exp(-0.01 * 100) = 0.5 * exp(-1) ≈ 0.5 * 0.368 = 0.184
        assert decayed < 0.5
        assert decayed > 0.1
    
    def test_apply_decay_to_states(self):
        now = 1000.0
        states = [
            FluidMemoryState(
                node_id="a",
                activation=0.8,
                last_touched_at=900.0,
                decay_rate=0.01,
            ),
            FluidMemoryState(
                node_id="b",
                activation=0.4,
                last_touched_at=950.0,
                decay_rate=0.01,
            ),
        ]
        decayed_states = apply_decay(states, now=now)
        assert decayed_states[0].activation < 0.8
        assert decayed_states[1].activation < 0.4
    
    def test_recency_score_fresh(self):
        now = time()
        score = compute_recency_score(now, now=now)
        assert score == 1.0
    
    def test_recency_score_old(self):
        now = 100000.0
        last_touched = 0.0  # Very old
        score = compute_recency_score(last_touched, now=now)
        assert score < 0.5  # Should be below 0.5


class TestContradiction:
    """Test contradiction functions."""
    
    def test_contradiction_pressure(self):
        event = FluidUpdateEvent(
            touched_node_ids=["a"],
            contradicts=["b"],
            source_trust=0.8,
        )
        pressure = compute_contradiction_pressure("b", event)
        assert pressure > 0.0  # Should have pressure
        assert pressure <= 1.0
    
    def test_no_pressure_if_not_contradicted(self):
        event = FluidUpdateEvent(
            touched_node_ids=["a"],
            contradicts=["c"],
            source_trust=0.8,
        )
        pressure = compute_contradiction_pressure("b", event)
        assert pressure == 0.0
    
    def test_apply_contradictions(self):
        states = [
            FluidMemoryState(node_id="a"),
            FluidMemoryState(node_id="b", contradiction_pressure=0.1),
            FluidMemoryState(node_id="c"),
        ]
        event = FluidUpdateEvent(
            touched_node_ids=["a"],
            contradicts=["b"],
            source_trust=0.8,
        )
        updated = apply_contradictions(states, event)
        assert updated[0].contradiction_pressure == 0.0  # Not contradicted
        assert updated[1].contradiction_pressure > 0.1  # Increased
        assert updated[2].contradiction_pressure == 0.0  # Not contradicted


class TestScoring:
    """Test scoring functions."""
    
    def test_fluid_score_boost(self):
        state = FluidMemoryState(
            node_id="test",
            activation=0.8,
            confidence=0.9,
            source_trust=0.95,
            recency_score=1.0,
            salience=0.7,
            legal_weight=0.5,
            contradiction_pressure=0.0,
        )
        base_score = 1.0
        adjusted = fluid_score(base_score, state)
        # Should reduce score (boost) since all positive factors are high
        assert adjusted < base_score
    
    def test_fluid_score_penalty(self):
        state = FluidMemoryState(
            node_id="test",
            activation=0.1,
            confidence=0.2,
            source_trust=0.2,
            recency_score=0.1,  # Low recency
            salience=0.1,
            legal_weight=0.0,
            contradiction_pressure=0.9,  # Very high contradiction
        )
        base_score = 1.0
        adjusted = fluid_score(base_score, state)
        # High contradiction should increase score (penalty)
        # Boost = 0.1*0.25 + 0.2*0.25 + 0.2*0.15 + 0.1*0.10 + 0.1*0.10 + 0 - 0.9*0.20
        # = 0.025 + 0.05 + 0.03 + 0.01 + 0.01 - 0.18 = -0.055 (negative)
        # Score = 1.0 - (-0.055) = 1.055 (penalty)
        assert adjusted > base_score
    
    def test_compute_fluid_boost(self):
        state = FluidMemoryState(
            node_id="test",
            activation=0.5,
            confidence=0.5,
            source_trust=0.5,
            recency_score=0.5,
            salience=0.5,
            legal_weight=0.0,
            contradiction_pressure=0.0,
        )
        boost = compute_fluid_boost(state)
        # 0.5*0.25 + 0.5*0.25 + 0.5*0.15 + 0.5*0.10 + 0.5*0.10 + 0.0*0.10
        # = 0.125 + 0.125 + 0.075 + 0.05 + 0.05 + 0.0 = 0.425
        assert boost > 0.4
        assert boost < 0.5
    
    def test_should_boost_retrieval(self):
        # High activation - should boost
        state_high = FluidMemoryState(
            node_id="test",
            activation=0.5,
            confidence=0.5,
        )
        assert should_boost_retrieval(state_high) is True
        
        # Low activation, low confidence - should not boost
        state_low = FluidMemoryState(
            node_id="test",
            activation=0.05,
            confidence=0.2,
        )
        assert should_boost_retrieval(state_low) is False
        
        # High contradiction - should not boost
        state_contradicted = FluidMemoryState(
            node_id="test",
            activation=0.8,
            confidence=0.8,
            contradiction_pressure=0.8,
        )
        assert should_boost_retrieval(state_contradicted) is False


@pytest.mark.asyncio
class TestStateStore:
    """Test FluidStateStore (requires SQLite)."""
    
    async def test_create_and_get(self, tmp_path):
        from m_flow.memory.fluid.state_store import FluidStateStore
        
        store = FluidStateStore(
            db_provider="sqlite",
            db_path=str(tmp_path),
            db_name="test_fluid",
        )
        
        # Create state
        state = FluidMemoryState(
            node_id="episode-123",
            activation=0.5,
            confidence=0.8,
        )
        await store.upsert(state)
        
        # Retrieve
        retrieved = await store.get("episode-123")
        assert retrieved is not None
        assert retrieved.node_id == "episode-123"
        assert retrieved.activation == 0.5
        assert retrieved.confidence == 0.8
    
    async def test_get_many_creates_defaults(self, tmp_path):
        from m_flow.memory.fluid.state_store import FluidStateStore
        
        store = FluidStateStore(
            db_provider="sqlite",
            db_path=str(tmp_path),
            db_name="test_fluid",
        )
        
        # Get multiple nodes (some don't exist)
        states = await store.get_many(["a", "b", "c"])
        assert len(states) == 3
        # All should have default values
        for state in states:
            assert state.activation == 0.0
            assert state.node_id in ["a", "b", "c"]
    
    async def test_merge_activation(self, tmp_path):
        from m_flow.memory.fluid.state_store import FluidStateStore
        
        store = FluidStateStore(
            db_provider="sqlite",
            db_path=str(tmp_path),
            db_name="test_fluid",
        )
        
        # Create initial state
        state = FluidMemoryState(node_id="test", activation=0.3)
        await store.upsert(state)
        
        # Merge higher activation
        results = await store.merge_activation({"test": 0.5}, merge_mode="max")
        assert len(results) == 1
        assert results[0].merged is True
        assert results[0].new_activation == 0.5
        
        # Verify
        retrieved = await store.get("test")
        assert retrieved.activation == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
