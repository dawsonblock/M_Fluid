"""
Unit tests for Fluid Memory module.
"""

import pytest
import asyncio
from time import time, sleep

from m_flow.memory.fluid.models import (
    FluidMemoryState,
    FluidUpdateEvent,
    ClaimConflict,
    get_source_weights,
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
    explain_fluid_score,
    should_boost_retrieval,
)
from m_flow.memory.fluid.decay import (
    SHORT_TERM_DECAY,
    NORMAL_DECAY,
    LEGAL_DECAY,
    DEFAULT_MIN_ACTIVATION,
    _LANE_RATES,
)


class TestFluidMemoryState:
    """Test FluidMemoryState model."""
    
    def test_default_values(self):
        state = FluidMemoryState(node_id="test-123")
        assert state.node_id == "test-123"
        assert state.activation == 0.0
        assert state.confidence == 0.5
        assert state.source_trust == 0.5
        assert state.decay_rate == NORMAL_DECAY          # 0.02/day, not 0.01/sec
        assert state.decay_lane == "normal"
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


class TestClaimConflict:
    """Test ClaimConflict model."""

    def test_basic_conflict(self):
        conflict = ClaimConflict(
            node_id_a="ep:1",
            node_id_b="ep:2",
            conflict_reason="Different dates for the same event",
            confidence=0.85,
        )
        assert conflict.node_id_a == "ep:1"
        assert conflict.node_id_b == "ep:2"
        assert conflict.confidence == 0.85
        assert conflict.detected_at > 0

    def test_optional_sources(self):
        conflict = ClaimConflict(
            node_id_a="ep:1",
            node_id_b="ep:2",
        )
        assert conflict.source_id_a is None
        assert conflict.source_id_b is None


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

    _SECONDS_PER_DAY = 86400.0

    def test_no_decay_if_just_touched(self):
        now = time()
        decayed = compute_decayed_activation(
            current_activation=0.5,
            last_touched=now,
            decay_rate=NORMAL_DECAY,
            now=now,
        )
        assert decayed == 0.5

    def test_decay_per_day_normal_lane(self):
        """After 1 day with NORMAL_DECAY=0.02, activation should decrease by ~2%."""
        now = time()
        one_day_ago = now - self._SECONDS_PER_DAY
        activation = 0.8
        decayed = compute_decayed_activation(
            current_activation=activation,
            last_touched=one_day_ago,
            decay_rate=NORMAL_DECAY,
            now=now,
        )
        import math
        expected = activation * math.exp(-NORMAL_DECAY * 1.0)  # 1 day
        assert abs(decayed - expected) < 0.001
        # Should retain most activation after 1 day
        assert decayed > 0.7

    def test_decay_per_day_legal_lane(self):
        """Legal lane decays 10x slower than normal."""
        now = time()
        one_week_ago = now - 7 * self._SECONDS_PER_DAY
        decayed_legal = compute_decayed_activation(
            current_activation=1.0,
            last_touched=one_week_ago,
            decay_rate=LEGAL_DECAY,
            now=now,
        )
        decayed_normal = compute_decayed_activation(
            current_activation=1.0,
            last_touched=one_week_ago,
            decay_rate=NORMAL_DECAY,
            now=now,
        )
        assert decayed_legal > decayed_normal

    def test_decay_per_day_short_term_lane(self):
        """Short-term lane decays much faster than normal."""
        now = time()
        one_day_ago = now - self._SECONDS_PER_DAY
        decayed = compute_decayed_activation(
            current_activation=1.0,
            last_touched=one_day_ago,
            decay_rate=SHORT_TERM_DECAY,
            now=now,
        )
        # 0.25/day: after 1 day should retain < 80% of activation
        assert decayed < 0.8

    def test_activation_floor_applied(self):
        """Activation should never drop below DEFAULT_MIN_ACTIVATION."""
        now = time()
        very_old = now - 365 * self._SECONDS_PER_DAY  # 1 year ago
        decayed = compute_decayed_activation(
            current_activation=1.0,
            last_touched=very_old,
            decay_rate=SHORT_TERM_DECAY,
            now=now,
            min_activation=DEFAULT_MIN_ACTIVATION,
        )
        assert decayed >= DEFAULT_MIN_ACTIVATION

    def test_lane_rates_are_ordered(self):
        """Verify: legal < normal < short_term decay rate."""
        assert LEGAL_DECAY < NORMAL_DECAY < SHORT_TERM_DECAY

    def test_apply_decay_to_states(self):
        now = time()
        one_day_ago = now - self._SECONDS_PER_DAY
        states = [
            FluidMemoryState(
                node_id="a",
                activation=0.8,
                last_touched_at=one_day_ago,
                decay_rate=NORMAL_DECAY,
                decay_lane="normal",
            ),
            FluidMemoryState(
                node_id="b",
                activation=0.4,
                last_touched_at=now - 3 * self._SECONDS_PER_DAY,
                decay_rate=NORMAL_DECAY,
                decay_lane="normal",
            ),
        ]
        decayed_states = apply_decay(states, now=now)
        # Activation should decrease but stay above floor
        assert decayed_states[0].activation < 0.8
        assert decayed_states[0].activation >= DEFAULT_MIN_ACTIVATION
        assert decayed_states[1].activation < 0.4
        assert decayed_states[1].activation >= DEFAULT_MIN_ACTIVATION

    def test_recency_score_fresh(self):
        now = time()
        score = compute_recency_score(now, now=now)
        assert score == 1.0

    def test_recency_score_old(self):
        now = time()
        three_days_ago = now - 3 * self._SECONDS_PER_DAY
        score = compute_recency_score(three_days_ago, now=now)
        assert score < 0.5  # 3 half-lives (1-day each) → score ~ 0.125


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
        assert adjusted < base_score, "High-quality node should reduce distance score"

    def test_fluid_score_penalty(self):
        state = FluidMemoryState(
            node_id="test",
            activation=0.1,
            confidence=0.2,
            source_trust=0.2,
            recency_score=0.1,
            salience=0.1,
            legal_weight=0.0,
            contradiction_pressure=0.9,
        )
        base_score = 1.0
        adjusted = fluid_score(base_score, state)
        assert adjusted > base_score, "Heavily-contradicted node should increase distance score"

    def test_boost_is_bounded_absolute(self):
        """Boost must never exceed max_impact=0.15 in absolute terms."""
        state = FluidMemoryState(
            node_id="test",
            activation=1.0,
            confidence=1.0,
            source_trust=1.0,
            recency_score=1.0,
            salience=1.0,
            legal_weight=1.0,
            contradiction_pressure=0.0,
        )
        base_score = 1.0
        adjusted = fluid_score(base_score, state)
        assert (base_score - adjusted) <= 0.15 + 1e-9, "Boost must not exceed max_impact=0.15"

    def test_boost_is_bounded_relative(self):
        """Boost must not exceed 30% of base_score."""
        state = FluidMemoryState(
            node_id="test",
            activation=1.0,
            confidence=1.0,
            source_trust=1.0,
            recency_score=1.0,
            salience=1.0,
            legal_weight=1.0,
            contradiction_pressure=0.0,
        )
        base_score = 0.20  # small base score
        adjusted = fluid_score(base_score, state)
        max_allowed_boost = base_score * 0.30
        actual_boost = base_score - adjusted
        assert actual_boost <= max_allowed_boost + 1e-9, "Boost must not exceed 30% of base score"

    def test_penalty_is_bounded(self):
        """Penalty (negative boost) must also be bounded."""
        state = FluidMemoryState(
            node_id="test",
            activation=0.0,
            confidence=0.0,
            source_trust=0.0,
            recency_score=0.0,
            salience=0.0,
            legal_weight=0.0,
            contradiction_pressure=1.0,
        )
        base_score = 1.0
        adjusted = fluid_score(base_score, state)
        penalty = adjusted - base_score
        assert penalty <= 0.15 + 1e-9, "Penalty must not exceed max_impact=0.15"

    def test_explain_fluid_score_keys(self):
        state = FluidMemoryState(
            node_id="test",
            activation=0.5,
            confidence=0.7,
            source_trust=0.8,
            recency_score=0.9,
            salience=0.6,
            legal_weight=0.4,
            contradiction_pressure=0.1,
        )
        explanation = explain_fluid_score(1.0, state)
        assert "base_score" in explanation
        assert "final_score" in explanation
        assert "raw_boost" in explanation
        assert "bounded_boost" in explanation
        assert "cap_applied" in explanation
        assert "components" in explanation
        # final_score must differ from base_score for non-trivial input
        assert explanation["final_score"] != explanation["base_score"]

    def test_explain_fluid_score_components_sum(self):
        """Sum of components should equal raw_boost."""
        state = FluidMemoryState(
            node_id="test",
            activation=0.4,
            confidence=0.6,
            source_trust=0.7,
            recency_score=0.5,
            salience=0.5,
            legal_weight=0.3,
            contradiction_pressure=0.2,
        )
        explanation = explain_fluid_score(1.0, state)
        component_sum = sum(explanation["components"].values())
        assert abs(component_sum - explanation["raw_boost"]) < 1e-9

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
        # 0.5*0.25 + 0.5*0.25 + 0.5*0.15 + 0.5*0.10 + 0.5*0.10 + 0.0*0.10 = 0.425
        assert boost > 0.4
        assert boost < 0.5

    def test_should_boost_retrieval(self):
        state_high = FluidMemoryState(node_id="test", activation=0.5, confidence=0.5)
        assert should_boost_retrieval(state_high) is True

        state_low = FluidMemoryState(node_id="test", activation=0.05, confidence=0.2)
        assert should_boost_retrieval(state_low) is False

        state_contradicted = FluidMemoryState(
            node_id="test", activation=0.8, confidence=0.8, contradiction_pressure=0.8
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
