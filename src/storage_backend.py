"""
Storage backend abstraction for data management.
Supports local filesystem and S3 storage with caching.
"""

import os
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, BinaryIO
from datetime import datetime
import logging

import pandas as pd
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StorageConfig(BaseModel):
    """Configuration for storage backend"""
    storage_type: str = Field(default="local", description="Storage type: 'local' or 's3'")
    base_path: str = Field(default="data/time_series", description="Base path for local storage or S3 prefix")
    
    # S3 specific configuration
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    s3_region: Optional[str] = Field(default="us-east-1", description="AWS region")
    s3_access_key_id: Optional[str] = Field(default=None, description="AWS access key ID")
    s3_secret_access_key: Optional[str] = Field(default=None, description="AWS secret access key")
    
    # Caching configuration
    enable_cache: bool = Field(default=True, description="Enable local caching")
    cache_dir: Optional[str] = Field(default=".cache", description="Local cache directory")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    
    class Config:
        arbitrary_types_allowed = True


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a file exists"""
        pass
    
    @abstractmethod
    def read(self, key: str) -> bytes:
        """Read a file and return its contents as bytes"""
        pass
    
    @abstractmethod
    def write(self, key: str, data: bytes) -> None:
        """Write data to a file"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a file"""
        pass
    
    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with the given prefix"""
        pass
    
    @abstractmethod
    def get_last_modified(self, key: str) -> Optional[datetime]:
        """Get last modified timestamp of a file"""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized LocalStorageBackend with base_path: {self.base_path}")
    
    def _get_path(self, key: str) -> Path:
        """Convert key to local file path"""
        # Ensure key doesn't start with / to avoid absolute paths
        key = key.lstrip('/')
        return self.base_path / key
    
    def exists(self, key: str) -> bool:
        path = self._get_path(key)
        return path.exists() and path.is_file()
    
    def read(self, key: str) -> bytes:
        path = self._get_path(key)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_bytes()
    
    def write(self, key: str, data: bytes) -> None:
        path = self._get_path(key)
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.debug(f"Written {len(data)} bytes to {path}")
    
    def delete(self, key: str) -> None:
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted {path}")
    
    def list_keys(self, prefix: str = "") -> list[str]:
        prefix_path = self._get_path(prefix) if prefix else self.base_path
        if not prefix_path.exists():
            return []
        
        keys = []
        for file_path in prefix_path.rglob("*"):
            if file_path.is_file():
                # Get relative path from base_path
                rel_path = file_path.relative_to(self.base_path)
                keys.append(str(rel_path).replace("\\", "/"))
        return keys
    
    def get_last_modified(self, key: str) -> Optional[datetime]:
        path = self._get_path(key)
        if not path.exists():
            return None
        return datetime.fromtimestamp(path.stat().st_mtime)


class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend"""
    
    def __init__(
        self,
        bucket: str,
        base_path: str = "",
        region: str = "us-east-1",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None
    ):
        self.bucket = bucket
        self.base_path = base_path.rstrip("/")
        self.region = region
        
        # Initialize S3 client
        s3_kwargs = {"region_name": region}
        if access_key_id and secret_access_key:
            s3_kwargs["aws_access_key_id"] = access_key_id
            s3_kwargs["aws_secret_access_key"] = secret_access_key
        
        try:
            self.s3_client = boto3.client("s3", **s3_kwargs)
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket)
            logger.info(f"Initialized S3StorageBackend with bucket: {bucket}, prefix: {self.base_path}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                raise ValueError(f"S3 bucket '{bucket}' does not exist")
            elif error_code == "403":
                raise ValueError(f"Access denied to S3 bucket '{bucket}'. Check credentials.")
            else:
                raise ValueError(f"Failed to connect to S3 bucket '{bucket}': {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to initialize S3 client: {str(e)}")
    
    def _get_key(self, key: str) -> str:
        """Convert file path to S3 key"""
        # Remove leading slash and ensure consistent path separators
        key = key.lstrip("/").replace("\\", "/")
        if self.base_path:
            return f"{self.base_path}/{key}"
        return key
    
    def exists(self, key: str) -> bool:
        s3_key = self._get_key(key)
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            raise
    
    def read(self, key: str) -> bytes:
        s3_key = self._get_key(key)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            return response["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                raise FileNotFoundError(f"S3 object not found: {s3_key}")
            raise
    
    def write(self, key: str, data: bytes) -> None:
        s3_key = self._get_key(key)
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=data
            )
            logger.debug(f"Written {len(data)} bytes to s3://{self.bucket}/{s3_key}")
        except ClientError as e:
            raise IOError(f"Failed to write to S3: {str(e)}")
    
    def delete(self, key: str) -> None:
        s3_key = self._get_key(key)
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            logger.debug(f"Deleted s3://{self.bucket}/{s3_key}")
        except ClientError as e:
            logger.warning(f"Failed to delete from S3: {str(e)}")
    
    def list_keys(self, prefix: str = "") -> list[str]:
        s3_prefix = self._get_key(prefix) if prefix else self.base_path
        if s3_prefix and not s3_prefix.endswith("/"):
            s3_prefix += "/"
        
        keys = []
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix)
            
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        # Remove base_path prefix if present
                        if self.base_path and key.startswith(f"{self.base_path}/"):
                            key = key[len(self.base_path) + 1:]
                        keys.append(key)
        except ClientError as e:
            logger.error(f"Failed to list S3 objects: {str(e)}")
        
        return keys
    
    def get_last_modified(self, key: str) -> Optional[datetime]:
        s3_key = self._get_key(key)
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return response["LastModified"]
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return None
            raise


class CachedStorageBackend(StorageBackend):
    """Storage backend wrapper with local caching"""
    
    def __init__(
        self,
        backend: StorageBackend,
        cache_dir: str = ".cache",
        ttl_seconds: int = 3600
    ):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        logger.info(f"Initialized CachedStorageBackend with cache_dir: {self.cache_dir}, TTL: {ttl_seconds}s")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key"""
        # Create a safe filename from the key
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / safe_key
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is valid (exists and not expired)"""
        if not cache_path.exists():
            return False
        
        cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        return cache_age < self.ttl_seconds
    
    def exists(self, key: str) -> bool:
        return self.backend.exists(key)
    
    def read(self, key: str) -> bytes:
        cache_path = self._get_cache_path(key)
        
        # Try to read from cache first
        if self._is_cache_valid(cache_path):
            logger.debug(f"Cache hit for {key}")
            return cache_path.read_bytes()
        
        # Read from backend
        logger.debug(f"Cache miss for {key}, reading from backend")
        data = self.backend.read(key)
        
        # Write to cache
        cache_path.write_bytes(data)
        return data
    
    def write(self, key: str, data: bytes) -> None:
        # Write to backend
        self.backend.write(key, data)
        
        # Update cache
        cache_path = self._get_cache_path(key)
        cache_path.write_bytes(data)
        logger.debug(f"Cached {key}")
    
    def delete(self, key: str) -> None:
        # Delete from backend
        self.backend.delete(key)
        
        # Delete from cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
    
    def list_keys(self, prefix: str = "") -> list[str]:
        return self.backend.list_keys(prefix)
    
    def get_last_modified(self, key: str) -> Optional[datetime]:
        return self.backend.get_last_modified(key)
    
    def clear_cache(self) -> None:
        """Clear all cached files"""
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                cache_file.unlink()
        logger.info("Cache cleared")


def create_storage_backend(config: StorageConfig) -> StorageBackend:
    """
    Factory function to create a storage backend based on configuration.
    
    Args:
        config: Storage configuration
        
    Returns:
        StorageBackend instance
    """
    # Determine storage type from config or environment
    storage_type = config.storage_type.lower()
    if storage_type == "auto":
        # Auto-detect: use S3 if credentials are available, otherwise local
        if config.s3_bucket and (
            config.s3_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        ):
            storage_type = "s3"
        else:
            storage_type = "local"
    
    # Create backend
    if storage_type == "local":
        backend = LocalStorageBackend(base_path=config.base_path)
    elif storage_type == "s3":
        # Get S3 credentials from config or environment
        bucket = config.s3_bucket or os.getenv("S3_BUCKET")
        if not bucket:
            raise ValueError("S3 bucket must be specified in config or S3_BUCKET environment variable")
        
        access_key_id = config.s3_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = config.s3_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        region = config.s3_region or os.getenv("AWS_REGION", "us-east-1")
        
        backend = S3StorageBackend(
            bucket=bucket,
            base_path=config.base_path,
            region=region,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key
        )
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")
    
    # Wrap with caching if enabled
    if config.enable_cache:
        cache_dir = config.cache_dir or ".cache"
        backend = CachedStorageBackend(
            backend=backend,
            cache_dir=cache_dir,
            ttl_seconds=config.cache_ttl_seconds
        )
    
    return backend

