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
        MFLOW_FLUID_ENABLE                      -> enable
        MFLOW_FLUID_ENABLE_ON_WRITE             -> enable_on_write
        MFLOW_FLUID_ENABLE_ON_RETRIEVAL         -> enable_on_retrieval
        MFLOW_FLUID_ENABLE_AUDIT                -> enable_audit
        MFLOW_FLUID_ENABLE_CONTRADICTION        -> enable_contradiction
        MFLOW_FLUID_ENABLE_LLM_CONTRADICTION    -> enable_llm_contradiction
        MFLOW_FLUID_DB_PROVIDER                 -> db_provider
        MFLOW_FLUID_DB_PATH                     -> db_path
        MFLOW_FLUID_DB_NAME                     -> db_name

    Decay lane constants (per-day rates):
        ATTENTION_DECAY      = 0.20   # short-term visibility (breaking news)
        INTEREST_DECAY       = 0.05   # user/session engagement (default)
        TRUST_DECAY          = 0.000  # provenance trust — immutable
        LEGAL_DECAY          = 0.000  # court/legal evidence — immutable
        CONTRADICTION_DECAY  = 0.01   # conflict pressure eases slowly

    Scoring:
        max_boost_impact     = 0.15   # absolute cap on fluid score adjustment
        max_boost_fraction   = 0.30   # fluid boost ≤ 30% of base score
    """

    model_config = SettingsConfigDict(env_prefix="MFLOW_FLUID_")

    # Master on/off switch — DISABLED BY DEFAULT (safety)
    enable: bool = False

    # Per-integration-point flags — DISABLED BY DEFAULT
    enable_on_write: bool = False
    enable_on_retrieval: bool = False

    # Sub-feature flags
    enable_audit: bool = True  # audit enabled when fluid is enabled
    enable_contradiction: bool = False  # contradiction detection disabled by default

    # LLM contradiction safety (disabled by default for legal/crime safety)
    enable_llm_contradiction: bool = False
    min_llm_contradiction_confidence: float = 0.75
    structured_contradiction_required: bool = True

    # Scoring safety: if True, return original bundles on scoring error
    fail_closed_on_scoring_error: bool = False

    # Storage
    db_provider: str = "sqlite"
    db_path: str = ""
    db_name: str = "fluid_memory"
    db_host: str = ""
    db_port: str = ""
    db_username: str = ""
    db_password: str = ""

    # ---------------------------------------------------------------------------
    # 5-Lane decay rates (per-day)
    # activation/recency use attention_decay or interest_decay
    # trust and legal_weight use trust_decay / legal_decay_rate (both 0.0 → immutable)
    # contradiction_pressure eases via contradiction_decay
    # ---------------------------------------------------------------------------
    attention_decay: float = 0.20        # short-term visibility (breaking news)
    interest_decay: float = 0.05         # user/session engagement
    trust_decay: float = 0.000           # provenance trust — never decays
    legal_decay_rate: float = 0.000      # court/gov evidence — immutable by policy
    contradiction_decay: float = 0.01    # conflict pressure eases slowly

    # Legacy aliases kept for backward compat (mapped to closest equivalent)
    short_term_decay: float = 0.20       # alias → attention_decay
    normal_decay: float = 0.05           # alias → interest_decay
    legal_decay: float = 0.000           # alias → legal_decay_rate

    minimum_activation: float = 0.05

    # ---------------------------------------------------------------------------
    # Effective score weights (must sum to 1.0)
    # effective_score = semantic*w_semantic + graph*w_graph
    #                  + activation*w_activation + trust*w_trust
    # ---------------------------------------------------------------------------
    w_semantic: float = 0.55
    w_graph: float = 0.20
    w_activation: float = 0.15
    w_trust: float = 0.10

    # Fluid boost bounds (legacy path)
    max_boost_impact: float = 0.15
    max_boost_fraction: float = 0.30

    # Activation propagation
    propagation_start_activation: float = 0.18
    propagation_max_depth: int = 2
    activation_increment: float = 0.25
    max_activation: float = 1.0

    # JudgeTracker / legal features
    enable_jurisdiction: bool = True
    enable_citation_graph: bool = True
    enable_timeline: bool = True
    enable_media_amplification: bool = True
    default_jurisdiction: str = "unknown"


@lru_cache(maxsize=1)
def get_fluid_config() -> FluidMemoryConfig:
    """Singleton accessor for fluid memory configuration."""
    return FluidMemoryConfig()
