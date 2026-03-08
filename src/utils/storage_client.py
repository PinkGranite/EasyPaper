"""
Lightweight OSS upload client for EasyPaper artifact persistence.

- **Description**:
    - Provides a singleton StorageClient that uploads artifact files directly
      to OSS (S3-compatible) so the backend does not need to read them from
      the local filesystem.
    - Initialised from environment variables shared with the backend:
      STORAGE_TYPE, OSS_ENDPOINT, OSS_BUCKET, OSS_ACCESS_KEY_ID,
      OSS_ACCESS_KEY_SECRET, OSS_REGION.
    - When STORAGE_TYPE != "oss", all operations are no-ops (local dev mode).
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.config import Config

    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False


class StorageClient:
    """
    Minimal async-compatible object-storage uploader with lazy initialization.

    - **Description**:
        - Wraps boto3 S3 client for OSS uploads.
        - Falls back to no-op when OSS is not configured.
        - Uses lazy init so that env vars loaded by dotenv at startup are available.
    """

    def __init__(self) -> None:
        self._client = None
        self._bucket: Optional[str] = None
        self._initialized = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        storage_type = os.getenv("STORAGE_TYPE", "local").lower()

        if storage_type != "oss":
            logger.info("StorageClient: STORAGE_TYPE=%s, uploads disabled", storage_type)
            return

        if not _HAS_BOTO3:
            logger.warning("StorageClient: boto3 not installed, OSS uploads disabled")
            return

        endpoint = os.getenv("OSS_ENDPOINT", "")
        bucket = os.getenv("OSS_BUCKET", "")
        access_key = os.getenv("OSS_ACCESS_KEY_ID", "")
        secret_key = os.getenv("OSS_ACCESS_KEY_SECRET", "")
        region = os.getenv("OSS_REGION", "cn-hangzhou")

        if not all([endpoint, bucket, access_key, secret_key]):
            logger.warning("StorageClient: incomplete OSS config, uploads disabled")
            return

        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(
                signature_version="s3",
                s3={"addressing_style": "virtual"},
            ),
        )
        logger.info("StorageClient: OSS configured bucket=%s endpoint=%s", bucket, endpoint)

    @property
    def enabled(self) -> bool:
        self._ensure_init()
        return self._client is not None

    async def upload(self, key: str, data: bytes) -> bool:
        """
        Upload bytes to the configured OSS bucket.

        - **Args**:
            - `key` (str): Object key (e.g. "papers/uid/tid/sections/intro.tex").
            - `data` (bytes): File content.

        - **Returns**:
            - `bool`: True on success, False if uploads are disabled or failed.
        """
        self._ensure_init()
        if self._client is None:
            return False
        try:
            self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
            return True
        except Exception:
            logger.exception("StorageClient: failed to upload key=%s", key)
            return False


storage_client = StorageClient()
