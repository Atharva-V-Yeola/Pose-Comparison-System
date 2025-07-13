# Deployment Guide for Pose Comparison System

This guide provides comprehensive instructions for deploying the Pose Comparison System in various environments.

## Quick Start Deployment

### Local Development

1. **Activate Virtual Environment**
   ```bash
   cd pose_comparison_app
   source venv/bin/activate
   ```

2. **Start the Application**
   ```bash
   python src/main.py
   ```

3. **Access the Application**
   - Open browser to `http://localhost:5000`
   - Allow camera permissions when prompted

### Production Deployment

For production environments, follow these steps:

1. **Install Production Dependencies**
   ```bash
   pip install gunicorn
   ```

2. **Create Production Configuration**
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex())')
   ```

3. **Run with Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 src.main:app
   ```

## Cloud Platform Deployment

### Heroku Deployment

1. **Create Procfile**
   ```
   web: gunicorn src.main:app
   ```

2. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   git add .
   git commit -m "Deploy pose comparison app"
   git push heroku main
   ```

### AWS Deployment

#### Using Elastic Beanstalk

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize and Deploy**
   ```bash
   eb init
   eb create pose-comparison-env
   eb deploy
   ```

#### Using EC2

1. **Launch EC2 Instance** (Ubuntu 22.04 recommended)
2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv nginx
   ```

3. **Deploy Application**
   ```bash
   git clone your-repo
   cd pose_comparison_app
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Configure Nginx**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

### Google Cloud Platform

#### Using App Engine

1. **Create app.yaml**
   ```yaml
   runtime: python311
   
   env_variables:
     FLASK_ENV: production
   
   automatic_scaling:
     min_instances: 1
     max_instances: 10
   ```

2. **Deploy**
   ```bash
   gcloud app deploy
   ```

## Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.main:app"]
```

### Build and Run

```bash
docker build -t pose-comparison .
docker run -p 5000:5000 pose-comparison
```

### Docker Compose

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./data:/app/data
```

## Environment Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment | development |
| SECRET_KEY | Flask secret key | auto-generated |
| PORT | Application port | 5000 |
| HOST | Application host | 0.0.0.0 |

### Configuration File

Create `config.py` for advanced configuration:

```python
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False
```

## Security Configuration

### HTTPS Setup

For production, always use HTTPS:

1. **Obtain SSL Certificate** (Let's Encrypt recommended)
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

2. **Update Nginx Configuration**
   ```nginx
   server {
       listen 443 ssl;
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/private.key;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header X-Forwarded-Proto https;
       }
   }
   ```

### Firewall Configuration

```bash
# Allow HTTP and HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Allow SSH (if needed)
sudo ufw allow 22

# Enable firewall
sudo ufw enable
```

## Monitoring and Logging

### Application Logging

Configure logging in `src/main.py`:

```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
```

### System Monitoring

Use tools like:
- **Prometheus + Grafana** for metrics
- **ELK Stack** for log analysis
- **New Relic** or **DataDog** for APM

## Performance Optimization

### Application Optimization

1. **Enable Gzip Compression**
   ```python
   from flask_compress import Compress
   Compress(app)
   ```

2. **Implement Caching**
   ```python
   from flask_caching import Cache
   cache = Cache(app, config={'CACHE_TYPE': 'simple'})
   ```

3. **Optimize MediaPipe Settings**
   ```python
   # In pose_comparison.py
   self.pose = self.mp_pose.Pose(
       static_image_mode=False,
       model_complexity=1,  # Reduce for better performance
       min_detection_confidence=0.5,
       min_tracking_confidence=0.5
   )
   ```

### Infrastructure Optimization

1. **Use CDN** for static assets
2. **Implement Load Balancing** for multiple instances
3. **Database Optimization** if using external database
4. **Memory Management** for large-scale deployments

## Backup and Recovery

### Data Backup

1. **Database Backup**
   ```bash
   # For SQLite
   cp src/database/app.db backup/app_$(date +%Y%m%d).db
   ```

2. **CSV Data Backup**
   ```bash
   # Backup CSV files
   tar -czf backup/csv_data_$(date +%Y%m%d).tar.gz src/static/*.csv
   ```

### Automated Backup Script

```bash
#!/bin/bash
BACKUP_DIR="/path/to/backup"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
cp src/database/app.db $BACKUP_DIR/app_$DATE.db

# Backup CSV files
tar -czf $BACKUP_DIR/csv_data_$DATE.tar.gz src/static/*.csv

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "*.db" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

## Troubleshooting Deployment Issues

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 5000
   lsof -i :5000
   # Kill the process
   kill -9 <PID>
   ```

2. **Permission Denied**
   ```bash
   # Fix file permissions
   chmod +x src/main.py
   chown -R www-data:www-data /path/to/app
   ```

3. **Module Not Found**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   # Reinstall requirements
   pip install -r requirements.txt
   ```

4. **Camera Access Issues**
   - Ensure HTTPS is used for remote access
   - Check browser permissions
   - Verify camera device availability

### Log Analysis

Check application logs for errors:

```bash
# Application logs
tail -f logs/app.log

# System logs
sudo journalctl -u your-service-name -f

# Nginx logs
sudo tail -f /var/log/nginx/error.log
```

## Scaling Considerations

### Horizontal Scaling

1. **Load Balancer Configuration**
   ```nginx
   upstream pose_app {
       server 127.0.0.1:5000;
       server 127.0.0.1:5001;
       server 127.0.0.1:5002;
   }
   
   server {
       location / {
           proxy_pass http://pose_app;
       }
   }
   ```

2. **Session Management**
   - Use external session storage (Redis)
   - Implement stateless design

### Vertical Scaling

1. **Increase Server Resources**
   - More CPU cores for MediaPipe processing
   - Additional RAM for concurrent users
   - Faster storage for file operations

2. **Optimize Application**
   - Implement connection pooling
   - Use async processing for heavy operations
   - Cache frequently accessed data

## Maintenance

### Regular Maintenance Tasks

1. **Update Dependencies**
   ```bash
   pip list --outdated
   pip install --upgrade package-name
   ```

2. **Clean Up Logs**
   ```bash
   # Rotate logs
   logrotate /etc/logrotate.d/pose-app
   ```

3. **Monitor Disk Space**
   ```bash
   df -h
   du -sh /path/to/app/*
   ```

4. **Security Updates**
   ```bash
   sudo apt update && sudo apt upgrade
   ```

### Health Checks

Implement health check endpoint:

```python
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}
```

Monitor with external tools:
- **Pingdom** for uptime monitoring
- **StatusCake** for performance monitoring
- **Custom scripts** for application-specific checks

This deployment guide provides comprehensive coverage for deploying the Pose Comparison System in various environments. Choose the deployment method that best fits your requirements and infrastructure.

