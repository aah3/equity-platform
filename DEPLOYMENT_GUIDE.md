# Deployment Guide - Equity Factor Analysis Platform

## Deployment Overview

This guide covers deployment strategies for the Equity Factor Analysis Platform, including local development, cloud deployment, containerization, and production environments.

## Deployment Architecture

### Production Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Server    │    │   Application   │
│   (nginx/AWS)   │────│   (nginx)       │────│   (Streamlit)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Database      │    │   File Storage  │
                       │   (PostgreSQL)  │    │   (S3/EFS)      │
                       └─────────────────┘    └─────────────────┘
```

## Local Development Setup

### 1. Prerequisites

#### System Requirements
- **Python**: 3.10 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 50GB free space
- **OS**: Windows 10+, macOS 10.15+, or Ubuntu 20.04+

#### Development Tools
```bash
# Install Python
python --version  # Should be 3.10+

# Install Git
git --version

# Install VS Code (recommended)
# Download from https://code.visualstudio.com/

# Install Docker (optional, for containerization)
docker --version
```

### 2. Environment Setup

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Verify activation
which python  # Should point to venv/bin/python
```

#### Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

#### Verify Installation
```bash
# Test Streamlit installation
streamlit hello

# Test application
streamlit run app_factors.py
```

### 3. Configuration

#### Environment Variables
Create `.env` file in project root:
```bash
# Data Configuration
DATA_SOURCE=yahoo
UNIVERSE=NDX Index
MAX_MEMORY_MB=2048

# AWS Configuration (if using cloud features)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name

# Database Configuration (if using PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost:5432/equitydb

# Application Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_THEME_BASE=light
```

#### Data Directory Setup
```bash
# Create data directories
mkdir -p data/time_series/{factors,returns,exposures,prices,benchmarks}
mkdir -p data/portfolios
mkdir -p data/static
mkdir -p logs
mkdir -p results
```

## Docker Deployment

### 1. Dockerfile

#### Main Application Dockerfile
```dockerfile
# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/time_series/{factors,returns,exposures,prices,benchmarks} \
    && mkdir -p data/portfolios \
    && mkdir -p data/static \
    && mkdir -p logs \
    && mkdir -p results

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_THEME_BASE=light

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app_factors.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Multi-stage Dockerfile (for production)
```dockerfile
# Multi-stage Dockerfile for production
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create data directories
RUN mkdir -p data/time_series/{factors,returns,exposures,prices,benchmarks} \
    && mkdir -p data/portfolios \
    && mkdir -p data/static \
    && mkdir -p logs \
    && mkdir -p results \
    && chown -R appuser:appuser data logs results

# Switch to non-root user
USER appuser

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_THEME_BASE=light

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "app_factors.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Docker Compose

#### Development Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./results:/app/results
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    env_file:
      - .env
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: equitydb
      POSTGRES_USER: equityuser
      POSTGRES_PASSWORD: equitypass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  postgres_data:
```

#### Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

  app:
    build: .
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    env_file:
      - .env.prod
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

volumes:
  postgres_data:
```

### 3. Docker Commands

#### Build and Run
```bash
# Build Docker image
docker build -t equity-app .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data equity-app

# Run with Docker Compose
docker-compose up -d

# Run production setup
docker-compose -f docker-compose.prod.yml up -d
```

#### Management Commands
```bash
# View logs
docker-compose logs -f app

# Scale application
docker-compose up -d --scale app=3

# Update application
docker-compose pull app
docker-compose up -d app

# Backup database
docker-compose exec postgres pg_dump -U equityuser equitydb > backup.sql
```

## Cloud Deployment

### 1. AWS Deployment

#### AWS ECS Deployment
```yaml
# aws-ecs-task-definition.json
{
  "family": "equity-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "equity-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/equity-app:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "STREAMLIT_SERVER_PORT",
          "value": "8501"
        },
        {
          "name": "STREAMLIT_SERVER_ADDRESS",
          "value": "0.0.0.0"
        }
      ],
      "secrets": [
        {
          "name": "AWS_ACCESS_KEY_ID",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:equity-app/aws-credentials:access-key-id::"
        },
        {
          "name": "AWS_SECRET_ACCESS_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:equity-app/aws-credentials:secret-access-key::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/equity-app",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### AWS CloudFormation Template
```yaml
# cloudformation-template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Equity Factor Analysis Platform'

Parameters:
  Environment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]
  
  VpcId:
    Type: AWS::EC2::VPC::Id
    Description: VPC ID
  
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
    Description: Subnet IDs

Resources:
  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub 'equity-app-${Environment}'
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT
      DefaultCapacityProviderStrategy:
        - CapacityProvider: FARGATE
          Weight: 1

  # ECS Task Definition
  ECSTaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: !Sub 'equity-app-${Environment}'
      NetworkMode: awsvpc
      RequiresCompatibilities:
        - FARGATE
      Cpu: 1024
      Memory: 2048
      ExecutionRoleArn: !Ref ECSExecutionRole
      TaskRoleArn: !Ref ECSTaskRole
      ContainerDefinitions:
        - Name: equity-app
          Image: !Sub '${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/equity-app:latest'
          PortMappings:
            - ContainerPort: 8501
              Protocol: tcp
          Environment:
            - Name: STREAMLIT_SERVER_PORT
              Value: '8501'
            - Name: STREAMLIT_SERVER_ADDRESS
              Value: '0.0.0.0'
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref CloudWatchLogGroup
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: ecs

  # ECS Service
  ECSService:
    Type: AWS::ECS::Service
    Properties:
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref ECSTaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups:
            - !Ref ECSSecurityGroup
          Subnets: !Ref SubnetIds
          AssignPublicIp: ENABLED
      LoadBalancers:
        - ContainerName: equity-app
          ContainerPort: 8501
          TargetGroupArn: !Ref TargetGroup

  # Application Load Balancer
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub 'equity-app-alb-${Environment}'
      Scheme: internet-facing
      Type: application
      Subnets: !Ref SubnetIds
      SecurityGroups:
        - !Ref ALBSecurityGroup

  # Target Group
  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: !Sub 'equity-app-tg-${Environment}'
      Port: 8501
      Protocol: HTTP
      VpcId: !Ref VpcId
      TargetType: ip
      HealthCheckPath: /_stcore/health
      HealthCheckProtocol: HTTP
      HealthCheckIntervalSeconds: 30
      HealthCheckTimeoutSeconds: 10
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3

  # CloudWatch Log Group
  CloudWatchLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub '/ecs/equity-app-${Environment}'
      RetentionInDays: 30

  # IAM Roles
  ECSExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ecs-tasks.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

  ECSTaskRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ecs-tasks.amazonaws.com
            Action: sts:AssumeRole
      Policies:
        - PolicyName: S3Access
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:DeleteObject
                Resource: !Sub 'arn:aws:s3:::${S3Bucket}/*'

  # Security Groups
  ALBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for ALB
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0

  ECSSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for ECS tasks
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8501
          ToPort: 8501
          SourceSecurityGroupId: !Ref ALBSecurityGroup

Outputs:
  LoadBalancerDNS:
    Description: DNS name of the load balancer
    Value: !GetAtt ApplicationLoadBalancer.DNSName
    Export:
      Name: !Sub '${AWS::StackName}-LoadBalancerDNS'
  
  ECSClusterName:
    Description: Name of the ECS cluster
    Value: !Ref ECSCluster
    Export:
      Name: !Sub '${AWS::StackName}-ECSClusterName'
```

### 2. Google Cloud Platform Deployment

#### GCP Cloud Run Deployment
```yaml
# cloud-run-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: equity-app
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/minScale: "1"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
      - image: gcr.io/PROJECT_ID/equity-app:latest
        ports:
        - containerPort: 8501
        env:
        - name: STREAMLIT_SERVER_PORT
          value: "8501"
        - name: STREAMLIT_SERVER_ADDRESS
          value: "0.0.0.0"
        resources:
          limits:
            cpu: 2000m
            memory: 4Gi
          requests:
            cpu: 1000m
            memory: 2Gi
```

#### GCP Deployment Script
```bash
#!/bin/bash
# deploy-gcp.sh

# Set variables
PROJECT_ID="your-project-id"
REGION="us-central1"
SERVICE_NAME="equity-app"

# Build and push image
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 100 \
  --max-instances 10
```

### 3. Azure Deployment

#### Azure Container Instances
```yaml
# azure-container-instance.yaml
apiVersion: 2021-09-01
location: eastus
name: equity-app
properties:
  containers:
  - name: equity-app
    properties:
      image: your-registry.azurecr.io/equity-app:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 4
      ports:
      - port: 8501
        protocol: TCP
      environmentVariables:
      - name: STREAMLIT_SERVER_PORT
        value: "8501"
      - name: STREAMLIT_SERVER_ADDRESS
        value: "0.0.0.0"
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8501
    dnsNameLabel: equity-app-unique-name
  restartPolicy: Always
type: Microsoft.ContainerInstance/containerGroups
```

## Production Configuration

### 1. Nginx Configuration

#### Nginx Reverse Proxy
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream streamlit {
        server app:8501;
    }

    server {
        listen 80;
        server_name your-domain.com;

        location / {
            proxy_pass http://streamlit;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check endpoint
        location /_stcore/health {
            proxy_pass http://streamlit;
            access_log off;
        }
    }
}
```

### 2. Environment Configuration

#### Production Environment Variables
```bash
# .env.prod
# Application
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_THEME_BASE=light
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/equitydb
POSTGRES_DB=equitydb
POSTGRES_USER=equityuser
POSTGRES_PASSWORD=secure_password

# AWS
AWS_ACCESS_KEY_ID=your_production_key
AWS_SECRET_ACCESS_KEY=your_production_secret
AWS_REGION=us-east-1
S3_BUCKET=equity-app-prod

# Security
SECRET_KEY=your_secret_key
ENCRYPTION_KEY=your_encryption_key

# Performance
MAX_MEMORY_MB=4096
CACHE_ENABLED=true
CACHE_SIZE=1000
PARALLEL_PROCESSING=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/application.log
```

### 3. Monitoring and Logging

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'equity-app'
    static_configs:
      - targets: ['app:8501']
    metrics_path: /metrics
    scrape_interval: 30s
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Equity App Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

## Security Configuration

### 1. SSL/TLS Setup

#### Let's Encrypt Certificate
```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

#### SSL Configuration in Nginx
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://streamlit;
        # ... other proxy settings
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### 2. Authentication and Authorization

#### Basic Authentication
```python
# auth.py
import streamlit as st
import hashlib
import secrets

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == st.secrets["password_hash"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False
```

#### OAuth Integration
```python
# oauth.py
import streamlit as st
from authlib.integrations.requests_client import OAuth2Session

def oauth_login():
    """OAuth login implementation."""
    client_id = st.secrets["oauth"]["client_id"]
    client_secret = st.secrets["oauth"]["client_secret"]
    redirect_uri = st.secrets["oauth"]["redirect_uri"]
    
    if "oauth_token" not in st.session_state:
        # Redirect to OAuth provider
        oauth = OAuth2Session(client_id, redirect_uri=redirect_uri)
        authorization_url, state = oauth.authorization_url(
            "https://oauth.provider.com/oauth/authorize"
        )
        st.session_state["oauth_state"] = state
        st.markdown(f"[Login with OAuth]({authorization_url})")
        return False
    else:
        return True
```

## Backup and Recovery

### 1. Database Backup

#### Automated Backup Script
```bash
#!/bin/bash
# backup-db.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="equitydb"
DB_USER="equityuser"

# Create backup
docker-compose exec postgres pg_dump -U $DB_USER $DB_NAME > $BACKUP_DIR/backup_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/backup_$DATE.sql

# Remove old backups (keep 30 days)
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete

# Upload to S3
aws s3 cp $BACKUP_DIR/backup_$DATE.sql.gz s3://your-backup-bucket/database/
```

#### Restore Script
```bash
#!/bin/bash
# restore-db.sh

BACKUP_FILE=$1
DB_NAME="equitydb"
DB_USER="equityuser"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Download from S3 if needed
if [[ $BACKUP_FILE == s3://* ]]; then
    aws s3 cp $BACKUP_FILE /tmp/backup.sql.gz
    BACKUP_FILE="/tmp/backup.sql.gz"
fi

# Decompress if needed
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | docker-compose exec -T postgres psql -U $DB_USER $DB_NAME
else
    docker-compose exec -T postgres psql -U $DB_USER $DB_NAME < $BACKUP_FILE
fi
```

### 2. Application Data Backup

#### S3 Sync Script
```bash
#!/bin/bash
# backup-data.sh

S3_BUCKET="your-backup-bucket"
LOCAL_DATA_DIR="/app/data"

# Sync data to S3
aws s3 sync $LOCAL_DATA_DIR s3://$S3_BUCKET/data/ --delete

# Create snapshot
aws s3 cp s3://$S3_BUCKET/data/ s3://$S3_BUCKET/snapshots/$(date +%Y%m%d_%H%M%S)/ --recursive
```

## Performance Optimization

### 1. Caching Configuration

#### Redis Configuration
```yaml
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

#### Application Caching
```python
# cache_config.py
import redis
from functools import wraps
import json
import pickle

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_result(expiration=3600):
    """Cache function results in Redis."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(key)
            if cached_result:
                return pickle.loads(cached_result)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(key, expiration, pickle.dumps(result))
            
            return result
        return wrapper
    return decorator
```

### 2. Load Balancing

#### HAProxy Configuration
```haproxy
# haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend http_front
    bind *:80
    redirect scheme https code 301 if !{ ssl_fc }

frontend https_front
    bind *:443 ssl crt /etc/ssl/certs/your-domain.pem
    default_backend streamlit_backend

backend streamlit_backend
    balance roundrobin
    option httpchk GET /_stcore/health
    server app1 app1:8501 check
    server app2 app2:8501 check
    server app3 app3:8501 check
```

## Deployment Scripts

### 1. Automated Deployment

#### Deployment Script
```bash
#!/bin/bash
# deploy.sh

set -e

ENVIRONMENT=${1:-dev}
VERSION=${2:-latest}

echo "Deploying to $ENVIRONMENT environment with version $VERSION"

# Build Docker image
docker build -t equity-app:$VERSION .

# Tag for registry
docker tag equity-app:$VERSION your-registry.com/equity-app:$VERSION

# Push to registry
docker push your-registry.com/equity-app:$VERSION

# Update deployment
if [ "$ENVIRONMENT" = "prod" ]; then
    # Production deployment
    kubectl set image deployment/equity-app equity-app=your-registry.com/equity-app:$VERSION
    kubectl rollout status deployment/equity-app
else
    # Development deployment
    docker-compose up -d
fi

echo "Deployment completed successfully"
```

#### Rollback Script
```bash
#!/bin/bash
# rollback.sh

ENVIRONMENT=${1:-dev}

echo "Rolling back $ENVIRONMENT environment"

if [ "$ENVIRONMENT" = "prod" ]; then
    # Production rollback
    kubectl rollout undo deployment/equity-app
    kubectl rollout status deployment/equity-app
else
    # Development rollback
    docker-compose down
    docker-compose up -d
fi

echo "Rollback completed successfully"
```

### 2. Health Checks

#### Health Check Script
```bash
#!/bin/bash
# health-check.sh

URL="http://localhost:8501/_stcore/health"
MAX_ATTEMPTS=30
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    if curl -f -s $URL > /dev/null; then
        echo "Application is healthy"
        exit 0
    fi
    
    echo "Attempt $ATTEMPT failed, waiting 10 seconds..."
    sleep 10
    ATTEMPT=$((ATTEMPT + 1))
done

echo "Application failed health check after $MAX_ATTEMPTS attempts"
exit 1
```

## Troubleshooting

### 1. Common Issues

#### Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limits
docker run --memory=4g --memory-swap=8g equity-app
```

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8501

# Use different port
docker run -p 8502:8501 equity-app
```

#### Database Connection Issues
```bash
# Check database connectivity
docker-compose exec postgres psql -U equityuser -d equitydb -c "SELECT 1;"

# Reset database
docker-compose down -v
docker-compose up -d
```

### 2. Log Analysis

#### Application Logs
```bash
# View application logs
docker-compose logs -f app

# Filter error logs
docker-compose logs app | grep ERROR

# Follow logs in real-time
tail -f logs/application.log
```

#### System Logs
```bash
# Check system resources
htop
df -h
free -h

# Check Docker logs
journalctl -u docker.service
```

This comprehensive deployment guide provides all the necessary information to deploy the Equity Factor Analysis Platform in various environments, from local development to production cloud deployments. The guide includes security considerations, monitoring, backup strategies, and troubleshooting tips to ensure a robust and reliable deployment.
