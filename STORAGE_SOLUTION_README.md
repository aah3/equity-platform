# Storage Backend Solution for Equity Factor Analysis Platform

## Overview

This solution provides a flexible, production-ready storage backend for the Equity Factor Analysis Platform. It supports both local filesystem storage (for development) and cloud storage (S3) for production deployments, with automatic caching for improved performance.

## Key Features

✅ **Multiple Storage Backends**: Local filesystem and AWS S3 support  
✅ **Automatic Caching**: Local caching with configurable TTL for improved performance  
✅ **Backward Compatible**: Existing code works without changes  
✅ **Environment-based Configuration**: Easy configuration via environment variables  
✅ **Production Ready**: Designed for Streamlit Cloud and other hosting platforms  
✅ **Git-friendly**: Data files excluded from version control  

## Quick Start

### Development (Local Storage)

No configuration needed - works out of the box:

```python
from file_data_manager import FileConfig, FileDataManager

cfg = FileConfig()
mgr = FileDataManager(cfg)
```

### Production (S3 Storage)

Set environment variables:

```bash
export USE_STORAGE_BACKEND=true
export S3_BUCKET=your-bucket-name
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

Or configure programmatically:

```python
cfg = FileConfig(
    use_storage_backend=True,
    storage_config={
        "storage_type": "s3",
        "s3_bucket": "your-bucket-name",
        "s3_region": "us-east-1"
    }
)
mgr = FileDataManager(cfg)
```

## Files Created

### Core Implementation
- `src/storage_backend.py` - Storage backend abstraction and implementations
- `src/file_data_manager.py` - Updated to support storage backends (backward compatible)

### Documentation
- `STORAGE_CONFIGURATION.md` - Detailed configuration guide
- `DEPLOYMENT_STORAGE.md` - Production deployment guide
- `example_storage_config.py` - Example configurations

### Configuration
- `.gitignore` - Updated to exclude data files and cache

## Architecture

```
FileDataManager
    ├── LocalStorageBackend (development)
    └── S3StorageBackend (production)
            └── CachedStorageBackend (performance)
```

The storage backend is abstracted, so the application code doesn't need to know whether data is stored locally or in S3. The `FileDataManager` automatically uses the appropriate backend based on configuration.

## Benefits

### For Development
- ✅ No setup required - works with local files
- ✅ Fast local file access
- ✅ Easy testing and debugging

### For Production
- ✅ Scalable cloud storage (S3)
- ✅ No data files in git repository
- ✅ Shared data across multiple instances
- ✅ Automatic backups with S3 versioning
- ✅ Cost-effective storage

### For Both
- ✅ Automatic caching for improved performance
- ✅ Same API regardless of storage backend
- ✅ Easy migration between backends
- ✅ Environment-based configuration

## Migration Path

### Step 1: Development
Continue using local storage (no changes needed)

### Step 2: Setup S3
- Create S3 bucket
- Set up IAM user
- Configure credentials

### Step 3: Deploy
- Set environment variables
- Deploy to Streamlit Cloud
- Data automatically uses S3

### Step 4: Verify
- Test data loading
- Verify S3 storage
- Monitor performance

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_STORAGE_BACKEND` | Enable storage backend | `false` |
| `STORAGE_TYPE` | Storage type (`local`, `s3`, `auto`) | `auto` |
| `S3_BUCKET` | S3 bucket name | None |
| `AWS_ACCESS_KEY_ID` | AWS access key | None |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | None |
| `AWS_REGION` | AWS region | `us-east-1` |
| `ENABLE_CACHE` | Enable caching | `true` |
| `CACHE_DIR` | Cache directory | `.cache` |

### Programmatic Configuration

```python
from file_data_manager import FileConfig

# Local storage
cfg = FileConfig(use_storage_backend=False)

# S3 storage
cfg = FileConfig(
    use_storage_backend=True,
    storage_config={
        "storage_type": "s3",
        "s3_bucket": "my-bucket",
        "enable_cache": True
    }
)
```

## Usage Examples

### Loading Data

```python
from file_data_manager import FileDataManager

# Works with both local and S3 storage
mgr = FileDataManager()
df = mgr.load_returns("SPX_Index_members")
```

### Storing Data

```python
# Store new data
mgr.store_returns(df_new, "SPX_Index_members", update_history=False)

# Overwrite existing data
mgr.store_returns(df_all, "SPX_Index_members", update_history=True)
```

### Checking Data

```python
# Check if file exists
exists = mgr._load.storage_backend.exists("returns/SPX_Index_members.parquet")

# List all files
files = mgr._load.storage_backend.list_keys("returns/")
```

## Troubleshooting

### Common Issues

1. **S3 Connection Errors**
   - Check AWS credentials
   - Verify bucket name and region
   - Ensure IAM permissions are correct

2. **Cache Issues**
   - Clear cache: `rm -rf .cache/`
   - Reduce cache TTL
   - Disable cache during development

3. **File Not Found**
   - Verify file exists in storage
   - Check base_path/prefix configuration
   - Verify file key format

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. **Use S3 for Production**: Always use S3 or cloud storage for production
2. **Enable Caching**: Use caching for improved performance
3. **Secure Credentials**: Never commit credentials to git
4. **Monitor Costs**: Monitor S3 storage and request costs
5. **Use Parquet Format**: More efficient than CSV for large datasets
6. **Regular Backups**: Set up S3 versioning or regular backups

## Next Steps

1. **Read Configuration Guide**: See `STORAGE_CONFIGURATION.md` for detailed configuration options
2. **Read Deployment Guide**: See `DEPLOYMENT_STORAGE.md` for production deployment instructions
3. **Try Examples**: See `example_storage_config.py` for configuration examples
4. **Test Locally**: Test with local storage first
5. **Deploy to Production**: Follow deployment guide for S3 setup

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the configuration guide
3. Check AWS S3 documentation
4. Review application logs

## License

Same as the main project.

