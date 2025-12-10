# Cloud Run Deployment Configuration

## Prerequisites

1. Google Cloud SDK installed
2. Docker installed
3. Model file (`food101_classifier.pth`) in `ml-model/saved_models/`

## Deploy to Cloud Run

### Step 1: Set up Google Cloud Project

```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### Step 2: Build and Push Docker Image

```bash
# Set variables
PROJECT_ID="your-project-id"
IMAGE_NAME="food-nutrition-detector"
REGION="us-central1"  # or your preferred region

# Build the image
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME
```

### Step 3: Deploy to Cloud Run

```bash
# Deploy with the following settings:
gcloud run deploy food-nutrition-api \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --port 8080 \
  --set-env-vars PORT=8080
```

### Step 4: Update CORS Settings (if needed)

If you need to allow specific origins:

```bash
gcloud run services update food-nutrition-api \
  --region $REGION \
  --set-env-vars ALLOWED_ORIGINS="https://yourdomain.com,https://www.yourdomain.com"
```

## Important Configuration Notes

### Memory and CPU

- **Memory**: 2Gi (minimum recommended for PyTorch model)
- **CPU**: 2 (for faster inference)
- Adjust based on your model size and traffic

### Timeout

- **timeout**: 300 seconds (5 minutes)
- Cloud Run default is 60s, but model loading may take longer
- First request (cold start) needs extra time

### Port

- **PORT**: 8080 (Cloud Run default)
- The app automatically reads from `PORT` environment variable

### Model File

- Ensure `food101_classifier.pth` is in `ml-model/saved_models/`
- The model is included in the Docker image
- File size should be reasonable (< 500MB recommended)

## Testing Locally

Test the Docker container locally before deploying:

```bash
# Build the image
docker build -t food-nutrition-detector .

# Run locally
docker run -p 8080:8080 -e PORT=8080 food-nutrition-detector

# Test the API
curl http://localhost:8080/api/health
```

## Troubleshooting

### Error: Container failed to start

**Cause**: Container not listening on PORT within timeout

**Solutions**:

1. Ensure model file exists in `ml-model/saved_models/`
2. Increase timeout: `--timeout 300`
3. Increase memory: `--memory 2Gi`
4. Check logs: `gcloud run logs read food-nutrition-api --region $REGION`

### Error: Out of memory

**Cause**: PyTorch model too large for allocated memory

**Solutions**:

1. Increase memory: `--memory 4Gi`
2. Use model quantization to reduce size
3. Use a smaller base model

### Error: Cold start timeout

**Cause**: First request takes too long to load model

**Solutions**:

1. Set minimum instances: `--min-instances 1`
2. Increase timeout: `--timeout 300`
3. Use startup CPU boost: `--cpu-boost` (if available)

## Cost Optimization

- Set `--min-instances 0` for development (only pay when used)
- Set `--min-instances 1` for production (avoid cold starts)
- Use `--max-instances` to limit concurrent containers
- Monitor usage in Google Cloud Console

## Get Service URL

```bash
gcloud run services describe food-nutrition-api \
  --region $REGION \
  --format 'value(status.url)'
```

## View Logs

```bash
# Stream logs
gcloud run logs tail food-nutrition-api --region $REGION

# Read recent logs
gcloud run logs read food-nutrition-api --region $REGION --limit 50
```

## Environment Variables

You can add more environment variables as needed:

```bash
gcloud run services update food-nutrition-api \
  --region $REGION \
  --set-env-vars PORT=8080,DEBUG=false,MAX_UPLOAD_SIZE=10485760
```
