"""Evidence Vault

Abstracts storage backends for evidence files.
Supports local filesystem, S3, and GCS backends.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO
from io import BytesIO
import hashlib

from judge_memory._logger import get_logger

logger = get_logger(__name__)


class EvidenceVault(ABC):
    """Abstract base for evidence storage backends.

    All evidence is content-addressed by SHA256 hash.
    Implementations must be thread-safe.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def store(self, content_hash: str, content: bytes) -> str:
        """Store evidence content and return URI.

        Args:
            content_hash: SHA256 hash of content
            content: Binary content to store

        Returns:
            Storage URI (e.g., "local://path", "s3://bucket/key", "gs://bucket/key")
        """
        ...

    @abstractmethod
    async def retrieve(self, uri: str) -> Optional[bytes]:
        """Retrieve content by URI.

        Args:
            uri: Storage URI from store()

        Returns:
            Content bytes or None if not found
        """
        ...

    @abstractmethod
    async def exists(self, uri: str) -> bool:
        """Check if content exists at URI.

        Args:
            uri: Storage URI

        Returns:
            True if exists, False otherwise
        """
        ...

    @abstractmethod
    async def delete(self, uri: str) -> bool:
        """Delete content at URI.

        Args:
            uri: Storage URI

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def healthcheck(self) -> Dict[str, Any]:
        """Verify vault connectivity and permissions.

        Returns:
            Dict with keys: status (ok/error), message (str), details (dict)
        """
        ...


class LocalVault(EvidenceVault):
    """Local filesystem evidence storage."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_path = Path(config.get("base_path", "./evidence"))
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _hash_to_path(self, content_hash: str) -> Path:
        """Convert hash to sharded path (2-char prefix)."""
        return self.base_path / content_hash[:2] / content_hash

    async def store(self, content_hash: str, content: bytes) -> str:
        path = self._hash_to_path(content_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        # Return absolute path for backward compatibility
        return str(path.absolute())

    async def retrieve(self, uri: str) -> Optional[bytes]:
        # Handle both legacy paths and local:// URIs
        if uri.startswith("local://"):
            path = Path(uri[8:])
        else:
            path = Path(uri)
        if path.exists():
            return path.read_bytes()
        return None

    async def exists(self, uri: str) -> bool:
        # Handle both legacy paths and local:// URIs
        if uri.startswith("local://"):
            path = Path(uri[8:])
        else:
            path = Path(uri)
        return path.exists()

    async def delete(self, uri: str) -> bool:
        # Handle both legacy paths and local:// URIs
        if uri.startswith("local://"):
            path = Path(uri[8:])
        else:
            path = Path(uri)
        if path.exists():
            path.unlink()
            return True
        return False

    async def healthcheck(self) -> Dict[str, Any]:
        try:
            test_file = self.base_path / ".healthcheck"
            test_file.write_text("ok")
            test_file.unlink()
            return {
                "status": "ok",
                "type": "local",
                "base_path": str(self.base_path),
                "message": "Local vault healthy",
            }
        except Exception as e:
            return {
                "status": "error",
                "type": "local",
                "base_path": str(self.base_path),
                "message": f"Healthcheck failed: {e}",
            }


class S3Vault(EvidenceVault):
    """S3-compatible evidence storage (AWS S3, MinIO, etc.)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bucket = config.get("bucket", "")
        self.prefix = config.get("prefix", "evidence/")
        self.endpoint = config.get("endpoint")  # For S3-compatible services
        self.region = config.get("region", "us-east-1")
        self.access_key = config.get("access_key")
        self.secret_key = config.get("secret_key")
        self._client = None

    def _get_client(self):
        """Lazy initialization of S3 client."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config

                session_kwargs = {}
                if self.access_key and self.secret_key:
                    session_kwargs["aws_access_key_id"] = self.access_key
                    session_kwargs["aws_secret_access_key"] = self.secret_key

                session = boto3.Session(**session_kwargs)

                client_kwargs = {
                    "region_name": self.region,
                    "config": Config(connect_timeout=10, read_timeout=30),
                }
                if self.endpoint:
                    client_kwargs["endpoint_url"] = self.endpoint

                self._client = session.client("s3", **client_kwargs)
            except ImportError:
                raise RuntimeError("boto3 required for S3 vault. Install: pip install boto3")
        return self._client

    def _hash_to_key(self, content_hash: str) -> str:
        """Convert hash to S3 key with sharding."""
        return f"{self.prefix}{content_hash[:2]}/{content_hash}"

    async def store(self, content_hash: str, content: bytes) -> str:
        key = self._hash_to_key(content_hash)
        self._get_client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType="application/octet-stream",
        )
        return f"s3://{self.bucket}/{key}"

    async def retrieve(self, uri: str) -> Optional[bytes]:
        if not uri.startswith("s3://"):
            return None
        # Parse s3://bucket/key
        parts = uri[5:].split("/", 1)
        if len(parts) != 2:
            return None
        bucket, key = parts
        try:
            response = self._get_client().get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except Exception as e:
            logger.warning(f"S3 retrieve failed for {uri}: {e}")
            return None

    async def exists(self, uri: str) -> bool:
        if not uri.startswith("s3://"):
            return False
        parts = uri[5:].split("/", 1)
        if len(parts) != 2:
            return False
        bucket, key = parts
        try:
            self._get_client().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    async def delete(self, uri: str) -> bool:
        if not uri.startswith("s3://"):
            return False
        parts = uri[5:].split("/", 1)
        if len(parts) != 2:
            return False
        bucket, key = parts
        try:
            self._get_client().delete_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    async def healthcheck(self) -> Dict[str, Any]:
        try:
            client = self._get_client()
            # Check bucket exists and we have write access
            client.head_bucket(Bucket=self.bucket)
            test_key = f"{self.prefix}.healthcheck"
            client.put_object(Bucket=self.bucket, Key=test_key, Body=b"ok")
            client.delete_object(Bucket=self.bucket, Key=test_key)
            return {
                "status": "ok",
                "type": "s3",
                "bucket": self.bucket,
                "endpoint": self.endpoint or "default",
                "message": "S3 vault healthy",
            }
        except Exception as e:
            return {
                "status": "error",
                "type": "s3",
                "bucket": self.bucket,
                "endpoint": self.endpoint or "default",
                "message": f"Healthcheck failed: {e}",
            }


class GCSVault(EvidenceVault):
    """Google Cloud Storage evidence vault."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bucket = config.get("bucket", "")
        self.prefix = config.get("prefix", "evidence/")
        self.project = config.get("project")
        self.credentials_path = config.get("credentials_path")
        self._client = None

    def _get_client(self):
        """Lazy initialization of GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage

                client_kwargs = {}
                if self.project:
                    client_kwargs["project"] = self.project
                if self.credentials_path:
                    from google.oauth2 import service_account
                    credentials = service_account.Credentials.from_service_account_file(
                        self.credentials_path
                    )
                    client_kwargs["credentials"] = credentials

                self._client = storage.Client(**client_kwargs)
            except ImportError:
                raise RuntimeError("google-cloud-storage required for GCS vault")
        return self._client

    def _hash_to_key(self, content_hash: str) -> str:
        """Convert hash to GCS key with sharding."""
        return f"{self.prefix}{content_hash[:2]}/{content_hash}"

    async def store(self, content_hash: str, content: bytes) -> str:
        key = self._hash_to_key(content_hash)
        bucket = self._get_client().bucket(self.bucket)
        blob = bucket.blob(key)
        blob.upload_from_string(content)
        return f"gs://{self.bucket}/{key}"

    async def retrieve(self, uri: str) -> Optional[bytes]:
        if not uri.startswith("gs://"):
            return None
        parts = uri[5:].split("/", 1)
        if len(parts) != 2:
            return None
        bucket, key = parts
        try:
            bucket_obj = self._get_client().bucket(bucket)
            blob = bucket_obj.blob(key)
            return blob.download_as_bytes()
        except Exception as e:
            logger.warning(f"GCS retrieve failed for {uri}: {e}")
            return None

    async def exists(self, uri: str) -> bool:
        if not uri.startswith("gs://"):
            return False
        parts = uri[5:].split("/", 1)
        if len(parts) != 2:
            return False
        bucket, key = parts
        try:
            bucket_obj = self._get_client().bucket(bucket)
            blob = bucket_obj.blob(key)
            return blob.exists()
        except Exception:
            return False

    async def delete(self, uri: str) -> bool:
        if not uri.startswith("gs://"):
            return False
        parts = uri[5:].split("/", 1)
        if len(parts) != 2:
            return False
        bucket, key = parts
        try:
            bucket_obj = self._get_client().bucket(bucket)
            blob = bucket_obj.blob(key)
            blob.delete()
            return True
        except Exception:
            return False

    async def healthcheck(self) -> Dict[str, Any]:
        try:
            client = self._get_client()
            bucket = client.bucket(self.bucket)
            # Check bucket exists
            if not bucket.exists():
                return {
                    "status": "error",
                    "type": "gcs",
                    "bucket": self.bucket,
                    "message": f"Bucket {self.bucket} does not exist",
                }
            test_key = f"{self.prefix}.healthcheck"
            blob = bucket.blob(test_key)
            blob.upload_from_string("ok")
            blob.delete()
            return {
                "status": "ok",
                "type": "gcs",
                "bucket": self.bucket,
                "project": self.project or "default",
                "message": "GCS vault healthy",
            }
        except Exception as e:
            return {
                "status": "error",
                "type": "gcs",
                "bucket": self.bucket,
                "project": self.project or "default",
                "message": f"Healthcheck failed: {e}",
            }


def create_vault(config: Dict[str, Any]) -> EvidenceVault:
    """Factory function to create vault from config.

    Args:
        config: Dict with at least "type" key (local, s3, gcs)

    Returns:
        EvidenceVault instance
    """
    vault_type = config.get("type", "local").lower()

    if vault_type == "local":
        return LocalVault(config)
    elif vault_type in ("s3", "s3_compatible"):
        return S3Vault(config)
    elif vault_type in ("gcs", "gs", "google"):
        return GCSVault(config)
    else:
        raise ValueError(f"Unknown vault type: {vault_type}")
