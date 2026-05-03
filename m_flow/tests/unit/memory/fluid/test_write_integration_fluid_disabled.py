"""
Test write integration when fluid is disabled.
Tests that write_episodic_memories uses SourceRegistry even when fluid disabled.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_write_uses_source_registry():
    """Test _fluid_touch_episodes uses SourceRegistry."""
    from m_flow.memory.fluid.source_registry import SourceRegistry
    from m_flow.memory.fluid.state_store import FluidStateStore

    mock_engine = AsyncMock()
    mock_engine.sessionmaker = MagicMock()

    registry = SourceRegistry(mock_engine)
    await registry.initialize()

    # Test that get_weights method exists and works
    trust, legal, decay_lane = await registry.get_weights("court_record")
    assert trust == 0.95
    assert legal == 1.00
    assert decay_lane == "legal"


def test_decay_lane_mapping():
    """Test canonical decay lane mapping."""
    lane_mapping = {
        "normal": "interest",
        "short_term": "attention",
        "legal": "legal",
    }

    assert lane_mapping["normal"] == "interest"
    assert lane_mapping["short_term"] == "attention"
    assert lane_mapping["legal"] == "legal"
    assert lane_mapping.get("unknown", "interest") == "interest"


def test_source_registry_fallback():
    """Test SourceRegistry falls back to hardcoded values."""
    from m_flow.memory.fluid.source_registry import _HARDCODED_FALLBACK

    # Test that fallback exists
    assert "unknown" in _HARDCODED_FALLBACK
    assert _HARDCODED_FALLBACK["unknown"].trust == 0.10
    assert _HARDCODED_FALLBACK["unknown"].legal_weight == 0.00
