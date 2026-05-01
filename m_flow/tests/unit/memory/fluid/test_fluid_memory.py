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
    SourceLineageRecord,
    MediaAmplificationEvent,
    get_source_weights,
)
from m_flow.memory.fluid.decay import (
    compute_decayed_activation,
    compute_decayed_contradiction_pressure,
    apply_decay,
    compute_recency_score,
    ATTENTION_DECAY,
    INTEREST_DECAY,
    TRUST_DECAY,
    LEGAL_DECAY,
    CONTRADICTION_DECAY,
    SHORT_TERM_DECAY,
    NORMAL_DECAY,
    DEFAULT_MIN_ACTIVATION,
    _LANE_RATES,
)
from m_flow.memory.fluid.contradiction import (
    compute_contradiction_pressure,
    apply_contradictions,
)
from m_flow.memory.fluid.scoring import (
    fluid_score,
    compute_fluid_boost,
    compute_effective_score,
    explain_effective_score,
    explain_fluid_score,
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
        assert state.decay_rate == INTEREST_DECAY        # 0.05/day
        assert state.decay_lane == "interest"            # v2 default lane
        assert state.reinforcement_count == 0
        assert state.contradiction_pressure == 0.0
        # JudgeTracker fields default
        assert state.source_lineage == []
        assert state.jurisdiction is None
        assert state.judge_id is None
        assert state.event_confidence == 0.5
        assert state.geographic_scope is None
        assert state.contradiction_cluster_id is None
        assert state.media_amplification == 0.0
    
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
        assert event.decay_lane == "interest"  # v2 default
    
    def test_contradiction_lists(self):
        event = FluidUpdateEvent(
            touched_node_ids=["a"],
            supports=["b", "c"],
            contradicts=["d"],
            source_trust=0.8,
        )
        assert event.supports == ["b", "c"]
        assert event.contradicts == ["d"]

    def test_judgetracker_enrichment_fields(self):
        event = FluidUpdateEvent(
            touched_node_ids=["a"],
            source_id="court-doc-001",
            jurisdiction="US-TX",
            judge_id="judge:42",
            geographic_scope="state",
            event_confidence=0.90,
            parent_source_ids=["src:original", "src:confirm"],
        )
        assert event.jurisdiction == "US-TX"
        assert event.judge_id == "judge:42"
        assert event.geographic_scope == "state"
        assert event.event_confidence == 0.90
        assert "src:original" in event.parent_source_ids


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
    """Test 5-lane decay model."""

    _SECONDS_PER_DAY = 86400.0

    def test_no_decay_if_just_touched(self):
        now = time()
        decayed = compute_decayed_activation(
            current_activation=0.5,
            last_touched=now,
            decay_rate=INTEREST_DECAY,
            now=now,
        )
        assert decayed == 0.5

    def test_attention_decay_rate(self):
        """ATTENTION_DECAY=0.20: after 1 day, activation < 82% of original."""
        now = time()
        one_day_ago = now - self._SECONDS_PER_DAY
        import math
        expected = 0.8 * math.exp(-ATTENTION_DECAY * 1.0)
        decayed = compute_decayed_activation(0.8, one_day_ago, ATTENTION_DECAY, now=now)
        assert abs(decayed - expected) < 0.001
        assert decayed < 0.8 * 0.82  # at least 18% lost per day

    def test_interest_decay_rate(self):
        """INTEREST_DECAY=0.05: after 1 day, activation should retain ~95%."""
        now = time()
        one_day_ago = now - self._SECONDS_PER_DAY
        import math
        expected = 0.8 * math.exp(-INTEREST_DECAY * 1.0)
        decayed = compute_decayed_activation(0.8, one_day_ago, INTEREST_DECAY, now=now)
        assert abs(decayed - expected) < 0.001
        assert decayed > 0.7  # retains most after 1 day

    def test_trust_decay_is_zero(self):
        """TRUST_DECAY=0.0: trust is immutable — activation never decreases."""
        assert TRUST_DECAY == 0.0
        now = time()
        one_year_ago = now - 365 * self._SECONDS_PER_DAY
        decayed = compute_decayed_activation(0.9, one_year_ago, TRUST_DECAY, now=now)
        # decay_rate=0.0 → no decay, returns clamped original
        assert abs(decayed - 0.9) < 1e-9

    def test_legal_decay_is_zero(self):
        """LEGAL_DECAY=0.0: legal evidence is immutable by policy."""
        assert LEGAL_DECAY == 0.0

    def test_contradiction_decay_eases_pressure(self):
        """CONTRADICTION_DECAY=0.01: pressure decreases slowly over 7 days."""
        now = time()
        one_week_ago = now - 7 * self._SECONDS_PER_DAY
        import math
        eased = compute_decayed_contradiction_pressure(
            current_pressure=0.8,
            last_touched=one_week_ago,
            now=now,
        )
        expected = 0.8 * math.exp(-CONTRADICTION_DECAY * 7)
        assert abs(eased - expected) < 0.001
        assert eased < 0.8  # pressure has eased
        assert eased > 0.0  # not fully resolved

    def test_trust_lane_node_never_decays(self):
        """A node in the 'trust' lane keeps its activation unchanged."""
        now = time()
        one_year_ago = now - 365 * self._SECONDS_PER_DAY
        state = FluidMemoryState(
            node_id="court-order",
            activation=0.9,
            last_touched_at=one_year_ago,
            decay_lane="trust",
            decay_rate=TRUST_DECAY,
        )
        decayed = apply_decay([state], now=now)
        assert abs(decayed[0].activation - 0.9) < 1e-9

    def test_legal_lane_node_never_decays(self):
        """A node in the 'legal' lane keeps its activation unchanged."""
        now = time()
        one_year_ago = now - 365 * self._SECONDS_PER_DAY
        state = FluidMemoryState(
            node_id="evidence:123",
            activation=0.95,
            last_touched_at=one_year_ago,
            decay_lane="legal",
            decay_rate=LEGAL_DECAY,
        )
        decayed = apply_decay([state], now=now)
        assert abs(decayed[0].activation - 0.95) < 1e-9

    def test_attention_lane_decays_fast(self):
        """Attention lane is fastest: retains < 82% after 1 day."""
        now = time()
        one_day_ago = now - self._SECONDS_PER_DAY
        state = FluidMemoryState(
            node_id="news:breaking",
            activation=1.0,
            last_touched_at=one_day_ago,
            decay_lane="attention",
        )
        decayed = apply_decay([state], now=now)
        assert decayed[0].activation < 0.82

    def test_apply_decay_also_eases_contradiction_pressure(self):
        """apply_decay() now also eases contradiction_pressure."""
        now = time()
        one_week_ago = now - 7 * self._SECONDS_PER_DAY
        state = FluidMemoryState(
            node_id="test",
            activation=0.5,
            contradiction_pressure=0.8,
            last_touched_at=one_week_ago,
            decay_lane="interest",
        )
        decayed = apply_decay([state], now=now)
        assert decayed[0].contradiction_pressure < 0.8

    def test_apply_decay_does_not_touch_source_trust(self):
        """apply_decay() must not modify source_trust or legal_weight."""
        now = time()
        one_year_ago = now - 365 * self._SECONDS_PER_DAY
        state = FluidMemoryState(
            node_id="test",
            activation=0.5,
            source_trust=0.95,
            legal_weight=1.0,
            last_touched_at=one_year_ago,
            decay_lane="interest",
        )
        decayed = apply_decay([state], now=now)
        assert decayed[0].source_trust == 0.95, "Trust must never decay"
        assert decayed[0].legal_weight == 1.0, "Legal weight must never decay"

    def test_activation_floor_applied(self):
        """Activation should never drop below DEFAULT_MIN_ACTIVATION."""
        now = time()
        very_old = now - 365 * self._SECONDS_PER_DAY
        decayed = compute_decayed_activation(
            current_activation=1.0,
            last_touched=very_old,
            decay_rate=ATTENTION_DECAY,
            now=now,
            min_activation=DEFAULT_MIN_ACTIVATION,
        )
        assert decayed >= DEFAULT_MIN_ACTIVATION

    def test_lane_rates_are_ordered(self):
        """Verify 5-lane rate ordering: legal=trust=0 < contradiction < interest < attention."""
        assert LEGAL_DECAY == 0.0
        assert TRUST_DECAY == 0.0
        assert TRUST_DECAY < CONTRADICTION_DECAY < INTEREST_DECAY < ATTENTION_DECAY

    def test_backward_compat_aliases(self):
        """Legacy SHORT_TERM_DECAY and NORMAL_DECAY aliases still exist."""
        assert SHORT_TERM_DECAY == ATTENTION_DECAY
        assert NORMAL_DECAY == INTEREST_DECAY

    def test_apply_decay_to_states(self):
        now = time()
        one_day_ago = now - self._SECONDS_PER_DAY
        states = [
            FluidMemoryState(
                node_id="a",
                activation=0.8,
                last_touched_at=one_day_ago,
                decay_rate=INTEREST_DECAY,
                decay_lane="interest",
            ),
            FluidMemoryState(
                node_id="b",
                activation=0.4,
                last_touched_at=now - 3 * self._SECONDS_PER_DAY,
                decay_rate=INTEREST_DECAY,
                decay_lane="interest",
            ),
        ]
        decayed_states = apply_decay(states, now=now)
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
        assert score < 0.5


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


class TestEffectiveScore:
    """Test compute_effective_score (v2 primary scoring path)."""

    def _state(self, **kwargs) -> FluidMemoryState:
        defaults = dict(
            node_id="test",
            activation=0.5,
            recency_score=0.8,
            source_trust=0.7,
            legal_weight=0.0,
            contradiction_pressure=0.0,
        )
        defaults.update(kwargs)
        return FluidMemoryState(**defaults)

    def test_basic_formula(self):
        """Verify the 4-component weighted formula."""
        state = self._state(activation=0.5, recency_score=0.5, source_trust=0.5, legal_weight=0.0)
        score = compute_effective_score(semantic_score=0.8, state=state, graph_score=0.6)
        # activation_score = 0.5*0.6 + 0.5*0.4 = 0.50
        # trust_score      = 0.5*0.7 + 0.0*0.3 = 0.35
        # raw = 0.8*0.55 + 0.6*0.20 + 0.50*0.15 + 0.35*0.10
        #     = 0.44 + 0.12 + 0.075 + 0.035 = 0.67
        assert 0.60 < score < 0.80

    def test_output_is_bounded_0_1(self):
        """Output must always be in [0, 1]."""
        state = self._state(activation=1.0, recency_score=1.0, source_trust=1.0, legal_weight=1.0)
        score = compute_effective_score(semantic_score=1.0, state=state, graph_score=1.0)
        assert 0.0 <= score <= 1.0

        state_min = self._state(activation=0.0, recency_score=0.0, source_trust=0.0)
        score_min = compute_effective_score(semantic_score=0.0, state=state_min, graph_score=0.0)
        assert 0.0 <= score_min <= 1.0

    def test_contradiction_penalty_is_multiplicative(self):
        """Contradiction reduces effective score multiplicatively, not additively."""
        state_clean = self._state(contradiction_pressure=0.0)
        state_conflicted = self._state(contradiction_pressure=1.0)
        semantic = 0.8
        score_clean = compute_effective_score(semantic, state_clean)
        score_conflicted = compute_effective_score(semantic, state_conflicted)
        assert score_conflicted < score_clean
        # At max contradiction, penalty capped at 30%
        assert score_conflicted >= score_clean * 0.70 - 1e-9

    def test_contradiction_penalty_capped_at_30_pct(self):
        """Even with contradiction_pressure=1.0, score reduced by at most 30%."""
        state = self._state(contradiction_pressure=1.0)
        s_clean = self._state(contradiction_pressure=0.0)
        base = compute_effective_score(0.8, s_clean)
        penalised = compute_effective_score(0.8, state)
        assert penalised >= base * 0.70 - 1e-9

    def test_semantic_is_dominant_at_55_pct(self):
        """With graph=0, activation=0, trust=0, score should be ~55% of semantic."""
        state = self._state(activation=0.0, recency_score=0.0, source_trust=0.0, legal_weight=0.0)
        score = compute_effective_score(semantic_score=1.0, state=state, graph_score=0.0)
        assert abs(score - 0.55) < 0.02

    def test_no_graph_fallback(self):
        """graph_score defaults to 0.0, formula still works."""
        state = self._state()
        score = compute_effective_score(semantic_score=0.7, state=state)
        assert 0.0 <= score <= 1.0

    def test_explain_effective_score_keys(self):
        state = self._state()
        result = explain_effective_score(0.8, state, graph_score=0.5)
        for key in ("semantic_score", "graph_score", "activation_score", "trust_score",
                     "raw_score", "contradiction_pressure", "contradiction_penalty_fraction",
                     "final_score", "components", "weights"):
            assert key in result, f"Missing key: {key}"

    def test_explain_components_match_formula(self):
        """Sum of components + penalty should equal final_score."""
        state = self._state(activation=0.6, recency_score=0.9, source_trust=0.8, legal_weight=0.3)
        result = explain_effective_score(0.75, state, graph_score=0.4)
        comp_sum = sum(result["components"].values())
        assert abs(comp_sum - result["raw_score"]) < 1e-9
        # final = raw_score * (1 - penalty_fraction), clamped to [0,1]
        expected_final = min(1.0, max(0.0, result["raw_score"])) * (1.0 - result["contradiction_penalty_fraction"])
        assert abs(result["final_score"] - expected_final) < 1e-9


class TestJudgeTrackerModels:
    """Test JudgeTracker-specific model fields."""

    def test_source_lineage_record(self):
        rec = SourceLineageRecord(
            node_id="ep:001",
            parent_source_id="src:abc",
            child_source_id="src:xyz",
            relationship="confirms",
        )
        assert rec.node_id == "ep:001"
        assert rec.relationship == "confirms"
        assert rec.recorded_at > 0

    def test_media_amplification_event(self):
        evt = MediaAmplificationEvent(
            node_id="ep:002",
            canonical_source_id="src:original",
            duplicate_source_ids=["src:copy1", "src:copy2"],
            amplification_factor=0.4,
        )
        assert evt.amplification_factor == 0.4
        assert len(evt.duplicate_source_ids) == 2

    def test_claim_conflict_has_cluster_id(self):
        conflict = ClaimConflict(
            node_id_a="ep:1",
            node_id_b="ep:2",
            contradiction_cluster_id="cluster-uuid-123",
            confidence=0.9,
        )
        assert conflict.contradiction_cluster_id == "cluster-uuid-123"

    def test_state_judgetracker_fields(self):
        state = FluidMemoryState(
            node_id="judge:99",
            jurisdiction="US-TX",
            judge_id="judge:42",
            geographic_scope="state",
            event_confidence=0.85,
            source_lineage=["src:a", "src:b"],
            media_amplification=0.3,
            contradiction_cluster_id="cluster-abc",
        )
        assert state.jurisdiction == "US-TX"
        assert state.judge_id == "judge:42"
        assert state.geographic_scope == "state"
        assert state.event_confidence == 0.85
        assert state.source_lineage == ["src:a", "src:b"]
        assert state.media_amplification == 0.3
        assert state.contradiction_cluster_id == "cluster-abc"


class TestJurisdictionWeighter:
    """Test JurisdictionWeighter."""

    def setup_method(self):
        from m_flow.memory.fluid.jurisdiction import JurisdictionWeighter
        self.weighter = JurisdictionWeighter()

    def test_federal_is_authoritative(self):
        assert self.weighter.is_authoritative("federal") is True

    def test_local_not_authoritative(self):
        assert self.weighter.is_authoritative("local") is False

    def test_unknown_returns_low_multiplier(self):
        mult = self.weighter.weight("completely_unknown_jurisdiction")
        assert mult == 0.50

    def test_court_record_bypasses_jurisdiction(self):
        """court_record source type always returns 1.0 regardless of jurisdiction."""
        mult = self.weighter.weight("local", source_type="court_record")
        assert mult == 1.0

    def test_apply_scales_trust(self):
        weighted = self.weighter.apply(0.80, "state", "government_data")
        assert weighted == 0.80  # government_data bypasses (1.0 * 0.80 = 0.80)

    def test_apply_with_local_scales_down(self):
        weighted = self.weighter.apply(0.80, "local", "blog_social")
        assert weighted < 0.80  # local multiplier = 0.60
        assert abs(weighted - 0.80 * 0.60) < 1e-9

    def test_none_jurisdiction_returns_unknown(self):
        mult = self.weighter.weight(None)
        assert mult == self.weighter.weight("unknown")

    def test_custom_overrides(self):
        from m_flow.memory.fluid.jurisdiction import JurisdictionWeighter
        custom_w = JurisdictionWeighter(custom_multipliers={"mytown": 0.999})
        assert custom_w.weight("mytown") == 0.999

    def test_list_jurisdictions_returns_dict(self):
        all_j = self.weighter.list_jurisdictions()
        assert isinstance(all_j, dict)
        assert "federal" in all_j
        assert "local" in all_j


@pytest.mark.asyncio
class TestCitationGraph:
    """Test CitationGraph with an in-memory SQLite store."""

    async def _make_store(self, tmp_path):
        from m_flow.memory.fluid.state_store import FluidStateStore
        return FluidStateStore(
            db_provider="sqlite",
            db_path=str(tmp_path),
            db_name="test_citation",
        )

    async def test_add_and_get_citations(self, tmp_path):
        from m_flow.memory.fluid.citation_graph import CitationGraph
        store = await self._make_store(tmp_path)
        cg = CitationGraph(store)

        await cg.add_link("ep:1", "src:a", "src:b", "confirms")
        await cg.add_link("ep:1", "src:c", "src:d", "cites")

        records = await cg.get_citations("ep:1")
        assert len(records) == 2

    async def test_cross_confirmation_score_empty(self, tmp_path):
        from m_flow.memory.fluid.citation_graph import CitationGraph
        store = await self._make_store(tmp_path)
        cg = CitationGraph(store)
        score = await cg.compute_cross_confirmation_score("ep:no-citations")
        assert score == 0.0

    async def test_cross_confirmation_score_grows_with_sources(self, tmp_path):
        from m_flow.memory.fluid.citation_graph import CitationGraph
        store = await self._make_store(tmp_path)
        cg = CitationGraph(store)

        for i in range(5):
            await cg.add_link("ep:1", f"src:parent_{i}", f"src:confirmer_{i}", "confirms")

        score = await cg.compute_cross_confirmation_score("ep:1")
        assert score == 1.0  # 5 independent confirming sources → score = 1.0

    async def test_amplification_does_not_count_as_confirmation(self, tmp_path):
        from m_flow.memory.fluid.citation_graph import CitationGraph
        store = await self._make_store(tmp_path)
        cg = CitationGraph(store)

        # 10 amplification links
        for i in range(10):
            await cg.add_link("ep:1", "src:orig", f"src:dup_{i}", "amplifies")

        # Cross-confirmation score should be 0 (no independent confirms)
        score = await cg.compute_cross_confirmation_score("ep:1")
        assert score == 0.0

        # But amplification factor should be 1.0
        amp = await cg.compute_amplification_factor("ep:1")
        assert amp == 1.0

    async def test_citation_depth(self, tmp_path):
        from m_flow.memory.fluid.citation_graph import CitationGraph
        store = await self._make_store(tmp_path)
        cg = CitationGraph(store)

        await cg.add_link("ep:1", "src:x", "src:y", "cites")
        await cg.add_link("ep:1", "src:x", "src:z", "confirms")

        depth = await cg.get_citation_depth("ep:1")
        assert depth == 2  # two distinct confirming sources


@pytest.mark.asyncio
class TestTimelineCompressor:
    """Test TimelineCompressor with in-memory SQLite."""

    async def _make_store(self, tmp_path):
        from m_flow.memory.fluid.state_store import FluidStateStore
        return FluidStateStore(
            db_provider="sqlite",
            db_path=str(tmp_path),
            db_name="test_timeline",
        )

    async def test_add_and_get_events(self, tmp_path):
        from m_flow.memory.fluid.timeline import TimelineCompressor
        store = await self._make_store(tmp_path)
        tl = TimelineCompressor(store)
        base = time()

        await tl.add_event("judge:1", base, "court_hearing", confidence=0.9)
        await tl.add_event("judge:1", base + 3600, "court_hearing", confidence=0.85)

        events = await tl.get_timeline("judge:1")
        assert len(events) == 2

    async def test_compress_merges_same_type_within_window(self, tmp_path):
        from m_flow.memory.fluid.timeline import TimelineCompressor
        store = await self._make_store(tmp_path)
        tl = TimelineCompressor(store)
        base = time()

        # 3 court hearings within 1 hour — should compress to 1
        for i in range(3):
            await tl.add_event("judge:1", base + i * 3600, "court_hearing", confidence=0.9)

        compressed = await tl.compress("judge:1", merge_window_days=1.0)
        assert len(compressed) == 1
        assert compressed[0].compressed is True
        assert compressed[0].compressed_count == 3

    async def test_compress_keeps_different_event_types_separate(self, tmp_path):
        from m_flow.memory.fluid.timeline import TimelineCompressor
        store = await self._make_store(tmp_path)
        tl = TimelineCompressor(store)
        base = time()

        await tl.add_event("judge:1", base, "court_hearing", confidence=0.9)
        await tl.add_event("judge:1", base + 3600, "arrest", confidence=0.8)
        await tl.add_event("judge:1", base + 7200, "court_hearing", confidence=0.7)

        # court_hearing at base, arrest, court_hearing again — 3 distinct groups
        compressed = await tl.compress("judge:1", merge_window_days=1.0)
        # court_hearing(base), arrest, court_hearing(base+7200) — 3 separate events
        assert len(compressed) == 3

    async def test_compress_averages_confidence(self, tmp_path):
        from m_flow.memory.fluid.timeline import TimelineCompressor
        store = await self._make_store(tmp_path)
        tl = TimelineCompressor(store)
        base = time()

        await tl.add_event("judge:1", base, "court_hearing", confidence=0.8)
        await tl.add_event("judge:1", base + 3600, "court_hearing", confidence=0.6)

        compressed = await tl.compress("judge:1", merge_window_days=1.0)
        assert len(compressed) == 1
        assert abs(compressed[0].confidence - 0.7) < 1e-4  # avg(0.8, 0.6)


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
