"""
Example configuration for using storage backends in the Equity Factor Analysis Platform.

This file demonstrates how to configure and use different storage backends
for development and production environments.
"""

import os
from file_data_manager import FileConfig, FileDataManager


# Example 1: Local Storage (Development)
def get_local_config():
    """Configuration for local filesystem storage (development)"""
    return FileConfig(
        base_dir="data/time_series",
        file_format="parquet",
        use_storage_backend=False
    )


# Example 2: S3 Storage (Production)
def get_s3_config():
    """Configuration for S3 storage (production)"""
    return FileConfig(
        base_dir="data/time_series",  # This becomes the S3 prefix
        file_format="parquet",
        use_storage_backend=True,
        storage_config={
            "storage_type": "s3",
            "s3_bucket": os.getenv("S3_BUCKET", "your-bucket-name"),
            "s3_region": os.getenv("AWS_REGION", "us-east-1"),
            "enable_cache": True,
            "cache_dir": ".cache",
            "cache_ttl_seconds": 3600
        }
    )


# Example 3: Auto-detect Storage (Uses S3 if credentials available, else local)
def get_auto_config():
    """Configuration that auto-detects storage backend"""
    return FileConfig(
        base_dir="data/time_series",
        file_format="parquet",
        use_storage_backend=True,
        storage_config={
            "storage_type": "auto",  # Auto-detect: S3 if credentials available, else local
            "s3_bucket": os.getenv("S3_BUCKET"),
            "s3_region": os.getenv("AWS_REGION", "us-east-1"),
            "enable_cache": True,
            "cache_dir": ".cache"
        }
    )


# Example 4: Environment-based Configuration
def get_config_from_env():
    """Get configuration from environment variables"""
    use_backend = os.getenv("USE_STORAGE_BACKEND", "false").lower() == "true"
    
    if use_backend:
        return FileConfig(
            use_storage_backend=True,
            storage_config={
                "storage_type": os.getenv("STORAGE_TYPE", "auto"),
                "base_path": os.getenv("STORAGE_BASE_PATH", "data/time_series"),
                "s3_bucket": os.getenv("S3_BUCKET"),
                "s3_region": os.getenv("AWS_REGION", "us-east-1"),
                "enable_cache": os.getenv("ENABLE_CACHE", "true").lower() == "true",
                "cache_dir": os.getenv("CACHE_DIR", ".cache"),
                "cache_ttl_seconds": int(os.getenv("CACHE_TTL_SECONDS", "3600"))
            }
        )
    else:
        return FileConfig(
            use_storage_backend=False,
            base_dir=os.getenv("DATA_DIR", "data/time_series")
        )


# Example Usage
if __name__ == "__main__":
    # Choose configuration based on environment
    is_production = os.getenv("ENVIRONMENT", "development") == "production"
    
    if is_production:
        config = get_s3_config()
    else:
        config = get_local_config()
    
    # Create data manager
    mgr = FileDataManager(config)
    
    # Use the manager as usual - it works the same way regardless of storage backend
    # Example: Load returns data
    try:
        df_returns = mgr.load_returns("SPX_Index_members")
        print(f"Loaded {len(df_returns)} rows of returns data")
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        print("This is expected if data hasn't been loaded yet.")
    
    # Example: Store data
    # import pandas as pd
    # df_new = pd.DataFrame({"date": ["2024-01-01"], "sid": ["AAPL"], "return": [0.01]})
    # mgr.store_returns(df_new, "SPX_Index_members", update_history=False)

