# Production Deployment with Storage Backend

This guide provides step-by-step instructions for deploying the Equity Factor Analysis Platform with S3 storage backend.

## Prerequisites

1. AWS Account with S3 access
2. S3 bucket created for data storage
3. IAM user with S3 permissions
4. Streamlit Cloud account (or other hosting platform)

## Step 1: Create S3 Bucket

1. Log in to AWS Console
2. Navigate to S3 service
3. Click "Create bucket"
4. Configure:
   - **Bucket name**: `equity-factor-data` (or your preferred name)
   - **Region**: `us-east-1` (or your preferred region)
   - **Block Public Access**: Keep enabled (recommended)
   - **Versioning**: Enable (recommended for data safety)

## Step 2: Create IAM User

1. Navigate to IAM service in AWS Console
2. Click "Users" → "Create user"
3. Set username: `equity-app-s3-user`
4. Select "Programmatic access"
5. Attach policy: Create custom policy with:

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
                "arn:aws:s3:::equity-factor-data/*",
                "arn:aws:s3:::equity-factor-data"
            ]
        }
    ]
}
```

6. Save the Access Key ID and Secret Access Key

## Step 3: Upload Existing Data to S3 (Optional)

If you have existing local data to migrate:

```bash
# Install AWS CLI if not already installed
pip install awscli

# Configure AWS credentials
aws configure

# Upload data to S3
aws s3 sync data/time_series/ s3://equity-factor-data/data/time_series/
```

## Step 4: Configure Streamlit Cloud

### Option A: Using Streamlit Cloud Secrets

1. Go to your Streamlit Cloud dashboard
2. Navigate to your app → Settings → Secrets
3. Add the following secrets:

```toml
# .streamlit/secrets.toml
USE_STORAGE_BACKEND = "true"
STORAGE_TYPE = "s3"
S3_BUCKET = "equity-factor-data"
AWS_ACCESS_KEY_ID = "your-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-secret-access-key"
AWS_REGION = "us-east-1"
ENABLE_CACHE = "true"
CACHE_DIR = ".cache"
CACHE_TTL_SECONDS = "3600"
```

### Option B: Using Environment Variables

If your hosting platform supports environment variables:

```bash
export USE_STORAGE_BACKEND=true
export STORAGE_TYPE=s3
export S3_BUCKET=equity-factor-data
export AWS_ACCESS_KEY_ID=your-access-key-id
export AWS_SECRET_ACCESS_KEY=your-secret-access-key
export AWS_REGION=us-east-1
export ENABLE_CACHE=true
export CACHE_DIR=.cache
export CACHE_TTL_SECONDS=3600
```

## Step 5: Update Application Code

The application code in `app_factors.py` already supports storage backends. However, you may want to add explicit configuration:

```python
# In app_factors.py, update the FileConfig initialization
from file_data_manager import FileConfig, FileDataManager
import os

# Get configuration from environment
def get_data_manager():
    use_backend = os.getenv("USE_STORAGE_BACKEND", "false").lower() == "true"
    
    if use_backend:
        cfg = FileConfig(
            use_storage_backend=True,
            storage_config={
                "storage_type": os.getenv("STORAGE_TYPE", "s3"),
                "base_path": "data/time_series",
                "s3_bucket": os.getenv("S3_BUCKET"),
                "s3_region": os.getenv("AWS_REGION", "us-east-1"),
                "enable_cache": True,
                "cache_dir": ".cache",
                "cache_ttl_seconds": 3600
            }
        )
    else:
        cfg = FileConfig(use_storage_backend=False)
    
    return FileDataManager(cfg)

# Use in your functions
def load_existing_data(model_input):
    mgr = get_data_manager()
    # ... rest of the code
```

## Step 6: Deploy to Streamlit Cloud

1. Push your code to GitHub (ensure `.gitignore` excludes data files)
2. Connect repository to Streamlit Cloud
3. Configure secrets (Step 4)
4. Deploy the app

## Step 7: Verify Deployment

1. Access your deployed app
2. Try loading existing data
3. Run data update process
4. Verify data is being stored in S3:
   ```bash
   aws s3 ls s3://equity-factor-data/data/time_series/ --recursive
   ```

## Troubleshooting

### Issue: "S3 bucket does not exist"
- Verify bucket name is correct
- Check bucket exists in the specified region
- Ensure IAM user has permissions to access the bucket

### Issue: "Access denied"
- Verify IAM user has correct permissions
- Check bucket policy allows access
- Verify AWS credentials are correct

### Issue: "Data not found"
- Check if data exists in S3 bucket
- Verify base_path/prefix is correct
- Check file key format matches expected format

### Issue: "Cache issues"
- Clear cache: Remove `.cache/` directory
- Reduce cache TTL if data updates frequently
- Disable cache during development if needed

## Cost Considerations

### S3 Storage Costs
- Standard storage: ~$0.023 per GB/month
- Requests: ~$0.005 per 1,000 requests
- Data transfer: Varies by region and amount

### Cost Optimization
- Use S3 Intelligent-Tiering for automatic cost optimization
- Enable compression (parquet format is already compressed)
- Use lifecycle policies to archive old data
- Monitor usage with AWS Cost Explorer

## Security Best Practices

1. **Never commit credentials**: Use environment variables or secrets management
2. **Use IAM roles**: Prefer IAM roles over access keys when possible
3. **Enable MFA**: Enable MFA for IAM users with S3 access
4. **Least privilege**: Grant minimum required permissions
5. **Encryption**: Enable S3 server-side encryption
6. **Audit logging**: Enable S3 access logging and CloudTrail

## Monitoring

### Set up CloudWatch Alarms
- Monitor S3 bucket size
- Monitor request rates
- Set up billing alerts

### Application Logging
- Log storage backend operations
- Monitor cache hit rates
- Track data access patterns

## Backup and Recovery

### S3 Versioning
- Enable versioning on S3 bucket
- Allows recovery of previous file versions
- Protects against accidental deletion

### Cross-Region Replication
- Set up replication to another region
- Provides disaster recovery capability
- Increases availability

### Regular Backups
- Schedule regular backups of critical data
- Store backups in separate S3 bucket or region
- Test backup restoration procedures

## Migration Checklist

- [ ] S3 bucket created and configured
- [ ] IAM user created with appropriate permissions
- [ ] Existing data uploaded to S3 (if applicable)
- [ ] Environment variables/secrets configured
- [ ] Application code updated (if needed)
- [ ] Application deployed and tested
- [ ] Data access verified
- [ ] Monitoring and alerts configured
- [ ] Backup strategy implemented
- [ ] Documentation updated

## Additional Resources

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Storage Configuration Guide](./STORAGE_CONFIGURATION.md)

