"""
Test MCP mutation guard.
Tests that mutation tools and CYPHER mode are disabled by default.
"""

import os
import pytest


def test_mutation_guard_default_disabled():
    """Test MFLOW_MCP_ALLOW_MUTATION defaults to false."""
    # Clear environment variable to test default
    if "MFLOW_MCP_ALLOW_MUTATION" in os.environ:
        del os.environ["MFLOW_MCP_ALLOW_MUTATION"]

    from m_flow_mcp.src.server import _MUTATION_ALLOWED

    assert _MUTATION_ALLOWED is False


def test_mutation_guard_enabled_with_env():
    """Test MFLOW_MCP_ALLOW_MUTATION=true enables mutation."""
    os.environ["MFLOW_MCP_ALLOW_MUTATION"] = "true"
    # Need to reload the module to pick up new env var
    import importlib
    import m_flow_mcp.src.server as server_module
    importlib.reload(server_module)

    assert server_module._MUTATION_ALLOWED is True

    # Clean up
    del os.environ["MFLOW_MCP_ALLOW_MUTATION"]
    importlib.reload(server_module)


def test_cypher_guard_default_disabled():
    """Test MFLOW_MCP_ALLOW_CYPHER defaults to false."""
    # Clear environment variable to test default
    if "MFLOW_MCP_ALLOW_CYPHER" in os.environ:
        del os.environ["MFLOW_MCP_ALLOW_CYPHER"]

    from m_flow_mcp.src.server import _CYPHER_ALLOWED

    assert _CYPHER_ALLOWED is False


def test_cypher_guard_enabled_with_env():
    """Test MFLOW_MCP_ALLOW_CYPHER=true enables Cypher."""
    os.environ["MFLOW_MCP_ALLOW_CYPHER"] = "true"
    # Need to reload the module to pick up new env var
    import importlib
    import m_flow_mcp.src.server as server_module
    importlib.reload(server_module)

    assert server_module._CYPHER_ALLOWED is True

    # Clean up
    del os.environ["MFLOW_MCP_ALLOW_CYPHER"]
    importlib.reload(server_module)


def test_check_mutation_allowed_blocks():
    """Test _check_mutation_allowed returns error when disabled."""
    from m_flow_mcp.src.server import _check_mutation_allowed

    # Ensure disabled
    if "MFLOW_MCP_ALLOW_MUTATION" in os.environ:
        del os.environ["MFLOW_ALLOW_MUTATION"]

    result = _check_mutation_allowed("test_tool")
    assert result is not None
    assert len(result) == 1
    assert "disabled by default" in result[0].text.lower()


def test_check_cypher_allowed_blocks():
    """Test _check_cypher_allowed returns error when disabled."""
    from m_flow_mcp.src.server import _check_cypher_allowed

    # Ensure disabled
    if "MFLOW_MCP_ALLOW_CYPHER" in os.environ:
        del os.environ["MFLOW_MCP_ALLOW_CYPHER"]

    result = _check_cypher_allowed("search")
    assert result is not None
    assert len(result) == 1
    assert "disabled by default" in result[0].text.lower()
