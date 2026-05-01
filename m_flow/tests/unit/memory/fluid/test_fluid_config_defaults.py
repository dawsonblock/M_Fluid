"""
Test Fluid Memory configuration defaults.
Tests that Fluid Memory is disabled by default and other config values.
"""


def test_fluid_disabled_by_default():
    """Test that Fluid Memory is disabled by default."""
    from m_flow.memory.fluid.config import get_fluid_config

    cfg = get_fluid_config()
    assert cfg.enabled is False


def test_decay_rate_default():
    """Test default decay rate is 0.05/day."""
    from m_flow.memory.fluid.config import get_fluid_config

    cfg = get_fluid_config()
    # Default decay_rate should be 0.05/day
    assert cfg.decay_rate == 0.05


def test_fail_closed_on_scoring_error_default():
    """Test fail closed on scoring error is enabled by default."""
    from m_flow.memory.fluid.config import get_fluid_config

    cfg = get_fluid_config()
    assert cfg.fail_closed_on_scoring_error is True
