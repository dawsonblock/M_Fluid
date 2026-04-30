"""
Fluid Memory Configuration

Environment-variable-driven feature flags and tuning parameters
for the fluid memory module.  All settings can be overridden via
environment variables without code changes.
"""

from __future__ import annotations

from functools import lru_cache

from m_flow.config.env_compat import MflowSettings, SettingsConfigDict


class FluidMemoryConfig(MflowSettings):
    """
    Configuration for the Fluid Memory module.

    Environment variable mapping (all prefixed MFLOW_FLUID_):
        MFLOW_FLUID_ENABLE              -> enable_fluid_memory
        MFLOW_FLUID_ON_WRITE            -> enable_fluid_on_write
        MFLOW_FLUID_ON_RETRIEVAL        -> enable_fluid_on_retrieval
        MFLOW_FLUID_AUDIT               -> enable_fluid_audit
        MFLOW_FLUID_CONTRADICTION       -> enable_fluid_contradiction
        MFLOW_FLUID_DB_PROVIDER         -> fluid_db_provider
        MFLOW_FLUID_DB_PATH             -> fluid_db_path
        MFLOW_FLUID_DB_NAME             -> fluid_db_name

    Decay lane constants (per-day rates):
        SHORT_TERM_DECAY  = 0.25   # temporary attention
        NORMAL_DECAY      = 0.02   # standard activation (default)
        LEGAL_DECAY       = 0.002  # court/legal evidence

    Scoring:
        max_boost_impact  = 0.15   # absolute cap on fluid score adjustment
        max_boost_fraction = 0.30  # fluid boost ≤ 30% of base score
    """

    model_config = SettingsConfigDict(env_prefix="MFLOW_FLUID_")

    # Master on/off switch
    enable: bool = True

    # Per-integration-point flags
    enable_on_write: bool = True
    enable_on_retrieval: bool = True

    # Sub-feature flags
    enable_audit: bool = True
    enable_contradiction: bool = True

    # Storage
    db_provider: str = "sqlite"
    db_path: str = ""
    db_name: str = "fluid_memory"
    db_host: str = ""
    db_port: str = ""
    db_username: str = ""
    db_password: str = ""

    # Decay lane rates (per-day)
    short_term_decay: float = 0.25
    normal_decay: float = 0.02
    legal_decay: float = 0.002
    minimum_activation: float = 0.05

    # Scoring bounds
    max_boost_impact: float = 0.15
    max_boost_fraction: float = 0.30

    # Activation propagation
    propagation_start_activation: float = 0.18
    propagation_max_depth: int = 2
    activation_increment: float = 0.25
    max_activation: float = 1.0


@lru_cache(maxsize=1)
def get_fluid_config() -> FluidMemoryConfig:
    """Singleton accessor for fluid memory configuration."""
    return FluidMemoryConfig()
