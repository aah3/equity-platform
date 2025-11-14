# Storage Configuration Guide

This guide explains how to configure data storage for the Equity Factor Analysis Platform. The platform supports both local filesystem storage (for development) and cloud storage backends (S3) for production deployment.

## Overview

The platform uses a storage backend abstraction that supports:
- **Local Filesystem**: For development and testing
- **AWS S3**: For production deployment with cloud storage
- **Caching**: Optional local caching for improved performance

## Quick Start

### Development (Local Storage)

By default, the app uses local filesystem storage. No configuration needed:

```python
from file_data_manager import FileConfig, FileDataManager

# Uses local storage in data/time_series/
cfg = FileConfig()
mgr = FileDataManager(cfg)
```

### Production (S3 Storage)

For production, configure S3 storage using environment variables:

```bash
# Set environment variables
export USE_STORAGE_BACKEND=true
export STORAGE_TYPE=s3
export S3_BUCKET=your-bucket-name
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-east-1
```

Or configure programmatically:

```python
from file_data_manager import FileConfig, FileDataManager

cfg = FileConfig(
    use_storage_backend=True,
    storage_config={
        "storage_type": "s3",
        "s3_bucket": "your-bucket-name",
        "s3_region": "us-east-1",
        "enable_cache": True,
        "cache_dir": ".cache",
        "cache_ttl_seconds": 3600
    }
)
mgr = FileDataManager(cfg)
```

## Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `USE_STORAGE_BACKEND` | Enable storage backend (true/false) | `false` | No |
| `STORAGE_TYPE` | Storage type: `local`, `s3`, or `auto` | `auto` | No |
| `S3_BUCKET` | S3 bucket name | None | Yes (for S3) |
| `AWS_ACCESS_KEY_ID` | AWS access key ID | None | Yes (for S3) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key | None | Yes (for S3) |
| `AWS_REGION` | AWS region | `us-east-1` | No |
| `ENABLE_CACHE` | Enable local caching | `true` | No |
| `CACHE_DIR` | Cache directory | `.cache` | No |
| `CACHE_TTL_SECONDS` | Cache TTL in seconds | `3600` | No |

### FileConfig Parameters

```python
class FileConfig:
    base_dir: Union[Path, str] = "data/time_series"  # Base path/prefix
    file_format: Literal["parquet", "csv", "json"] = "parquet"
    use_storage_backend: bool = False  # Enable storage backend
    storage_config: Optional[dict] = None  # Storage backend config
```

### StorageConfig Parameters

```python
class StorageConfig:
    storage_type: str = "local"  # "local", "s3", or "auto"
    base_path: str = "data/time_series"  # Base path/prefix
    s3_bucket: Optional[str] = None  # S3 bucket name
    s3_region: str = "us-east-1"  # AWS region
    s3_access_key_id: Optional[str] = None  # AWS access key
    s3_secret_access_key: Optional[str] = None  # AWS secret key
    enable_cache: bool = True  # Enable caching
    cache_dir: str = ".cache"  # Cache directory
    cache_ttl_seconds: int = 3600  # Cache TTL
```

## Deployment Scenarios

### Scenario 1: Streamlit Cloud (Recommended)

For Streamlit Cloud deployment, use S3 storage with environment variables:

1. **Create S3 Bucket**: Create an S3 bucket for your data
2. **Set up IAM User**: Create an IAM user with S3 read/write permissions
3. **Configure Secrets**: Add secrets in Streamlit Cloud dashboard:
   - `USE_STORAGE_BACKEND=true`
   - `STORAGE_TYPE=s3`
   - `S3_BUCKET=your-bucket-name`
   - `AWS_ACCESS_KEY_ID=your-access-key`
   - `AWS_SECRET_ACCESS_KEY=your-secret-key`
   - `AWS_REGION=us-east-1`

4. **Update app_factors.py**: The app will automatically use S3 if configured

### Scenario 2: Local Development with S3

For local development that uses S3:

1. **Install dependencies**:
```bash
pip install boto3
```

2. **Configure AWS credentials** (one of):
   - Environment variables (see above)
   - AWS credentials file (`~/.aws/credentials`)
   - IAM roles (if running on EC2)

3. **Run the app**: The app will use S3 if `USE_STORAGE_BACKEND=true`

### Scenario 3: Hybrid (Local + S3 Sync)

For local development with periodic S3 sync:

1. **Use local storage for development**
2. **Sync to S3 periodically** using the existing `sync_data_with_s3()` function
3. **Use S3 in production**

## S3 Bucket Setup

### 1. Create S3 Bucket

```bash
aws s3 mb s3://your-bucket-name --region us-east-1
```

### 2. Configure Bucket Policies

Create a bucket policy that allows read/write access:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::ACCOUNT_ID:user/YOUR_USER"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name/*",
                "arn:aws:s3:::your-bucket-name"
            ]
        }
    ]
}
```

### 3. Set up IAM User

Create an IAM user with the following policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name/*",
                "arn:aws:s3:::your-bucket-name"
            ]
        }
    ]
}
```

## Caching

The storage backend includes optional local caching for improved performance:

- **Cache TTL**: Files are cached for a configurable time (default: 1 hour)
- **Cache Directory**: Cached files are stored in `.cache/` by default
- **Automatic Invalidation**: Cache is automatically invalidated when files are updated

To disable caching:

```python
cfg = FileConfig(
    use_storage_backend=True,
    storage_config={
        "enable_cache": False
    }
)
```

To clear cache:

```python
# If using CachedStorageBackend directly
storage_backend.clear_cache()
```

## Migration Guide

### Migrating from Local to S3

1. **Upload existing data to S3**:
```bash
aws s3 sync data/time_series/ s3://your-bucket-name/data/time_series/
```

2. **Update configuration** to use S3 (see above)

3. **Test the application** to ensure data is accessible

4. **Remove local data** (optional, after verification):
```bash
rm -rf data/time_series/
```

### Migrating from S3 to Local

1. **Download data from S3**:
```bash
aws s3 sync s3://your-bucket-name/data/time_series/ data/time_series/
```

2. **Update configuration** to use local storage:
```python
cfg = FileConfig(use_storage_backend=False)
```

## Troubleshooting

### Common Issues

#### 1. S3 Connection Errors

**Error**: `Failed to connect to S3 bucket`

**Solutions**:
- Check AWS credentials are set correctly
- Verify bucket name is correct
- Ensure bucket exists and is accessible
- Check IAM permissions

#### 2. Cache Issues

**Error**: Stale data in cache

**Solutions**:
- Clear cache: `rm -rf .cache/`
- Reduce cache TTL
- Disable caching during development

#### 3. File Not Found Errors

**Error**: `File not found` when using S3

**Solutions**:
- Verify file exists in S3 bucket
- Check base_path/prefix configuration
- Ensure file key format is correct

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check storage backend status:

```python
# Check if file exists
exists = mgr._load.storage_backend.exists("prices/SPX_Index.parquet")

# List all files
files = mgr._load.storage_backend.list_keys("prices/")
```

## Best Practices

1. **Use S3 for Production**: Always use S3 or cloud storage for production deployments
2. **Enable Caching**: Use caching for improved performance, especially with S3
3. **Secure Credentials**: Never commit AWS credentials to git; use environment variables or secrets management
4. **Regular Backups**: Set up S3 versioning or regular backups for critical data
5. **Monitor Costs**: Monitor S3 storage and request costs
6. **Use Parquet Format**: Parquet is more efficient than CSV for large datasets
7. **Cache Strategy**: Use appropriate cache TTL based on data update frequency

## Example Configuration Files

### `.env` (Local Development)

```bash
# Local storage (default)
USE_STORAGE_BACKEND=false
```

### `.env.production` (Production)

```bash
# S3 storage
USE_STORAGE_BACKEND=true
STORAGE_TYPE=s3
S3_BUCKET=your-bucket-name
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
ENABLE_CACHE=true
CACHE_DIR=.cache
CACHE_TTL_SECONDS=3600
```

### `config.py` (Programmatic Configuration)

```python
import os
from file_data_manager import FileConfig

# Production configuration
def get_production_config():
    return FileConfig(
        use_storage_backend=True,
        storage_config={
            "storage_type": "s3",
            "s3_bucket": os.getenv("S3_BUCKET"),
            "s3_region": os.getenv("AWS_REGION", "us-east-1"),
            "enable_cache": True,
            "cache_dir": ".cache",
            "cache_ttl_seconds": 3600
        }
    )

# Development configuration
def get_development_config():
    return FileConfig(
        use_storage_backend=False,
        base_dir="data/time_series"
    )
```

## Additional Resources

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [Streamlit Cloud Secrets](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)

