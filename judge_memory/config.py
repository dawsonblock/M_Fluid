"""Judge Memory Configuration"""

import os
from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

PathLike = Union[str, Path]


class JudgeMemoryConfig(BaseModel):
    """Configuration for Judge Memory subsystem."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    data_dir: PathLike = Field(
        default="./judge_memory_data",
        description="Directory for database and evidence storage",
    )
    sqlite_path: Optional[PathLike] = Field(
        default=None,
        description="Optional explicit SQLite database path",
    )
    evidence_dir_override: Optional[PathLike] = Field(
        default=None,
        alias="evidence_dir",
        description="Optional explicit evidence storage directory",
    )
    enable_fluid_memory: bool = Field(
        default=False,
        description="Enable fluid memory scoring",
    )
    min_claim_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for claims",
    )
    max_claims_per_evidence: int = Field(
        default=100,
        ge=1,
        description="Maximum claims per evidence record",
    )
    enable_audit: bool = Field(
        default=True,
        description="Enable audit event logging",
    )
    # Vault configuration for external evidence storage
    vault_type: str = Field(
        default="local",
        description="Evidence vault type: local, s3, gcs",
    )
    vault_config: dict = Field(
        default_factory=dict,
        description="Vault-specific configuration (bucket, endpoint, credentials, etc.)",
    )

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def db_path(self) -> str:
        if self.sqlite_path:
            return str(Path(self.sqlite_path))
        return str(self.data_path / "judge_memory.db")

    @property
    def evidence_dir(self) -> Path:
        if self.evidence_dir_override:
            return Path(self.evidence_dir_override)
        return self.data_path / "evidence"

    @model_validator(mode="after")
    def apply_env_vault_config(self):
        """Apply JTA_EVIDENCE_STORE_ROOT environment variable if set."""
        env_root = os.environ.get("JTA_EVIDENCE_STORE_ROOT")
        if env_root:
            self.vault_type = "local"
            self.vault_config["base_path"] = env_root
            # Also set evidence_dir_override for backward compatibility
            self.evidence_dir_override = env_root
        return self
