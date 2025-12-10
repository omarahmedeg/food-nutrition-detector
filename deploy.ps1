# Google Cloud Run Deployment Script (PowerShell)
# This script builds and deploys the Food Nutrition Detector API to Cloud Run

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Food Nutrition Detector - Cloud Run Deployment" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if required files exist
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

if (-not (Test-Path "ml-model\saved_models\food101_classifier.pth")) {
    Write-Host "ERROR: Model file not found!" -ForegroundColor Red
    Write-Host "Please ensure ml-model\saved_models\food101_classifier.pth exists"
    exit 1
}

if (-not (Test-Path "Dockerfile")) {
    Write-Host "ERROR: Dockerfile not found!" -ForegroundColor Red
    exit 1
}

# Get project ID
$PROJECT_ID = Read-Host "Enter your Google Cloud Project ID"

if ([string]::IsNullOrWhiteSpace($PROJECT_ID)) {
    Write-Host "ERROR: Project ID cannot be empty" -ForegroundColor Red
    exit 1
}

# Configuration
$IMAGE_NAME = "food-nutrition-detector"
$SERVICE_NAME = "food-nutrition-api"
$REGION = "us-central1"  # Change if needed

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Project ID: $PROJECT_ID"
Write-Host "  Image Name: $IMAGE_NAME"
Write-Host "  Service Name: $SERVICE_NAME"
Write-Host "  Region: $REGION"
Write-Host ""

# Set project
Write-Host "Setting Google Cloud project..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID

# Build Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed" -ForegroundColor Red
    exit 1
}

# Push to Google Container Registry
Write-Host "Pushing image to Google Container Registry..." -ForegroundColor Yellow
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker push failed" -ForegroundColor Red
    exit 1
}

# Deploy to Cloud Run
Write-Host "Deploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $SERVICE_NAME `
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2 `
  --timeout 300 `
  --max-instances 10 `
  --port 8080 `
  --set-env-vars PORT=8080

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Deployment successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Getting service URL..." -ForegroundColor Yellow
    $SERVICE_URL = gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
    Write-Host ""
    Write-Host "Your API is deployed at:" -ForegroundColor Green
    Write-Host $SERVICE_URL -ForegroundColor Green
    Write-Host ""
    Write-Host "Test your API:" -ForegroundColor Yellow
    Write-Host "  Health check: curl $SERVICE_URL/api/health"
    Write-Host "  API docs: $SERVICE_URL"
} else {
    Write-Host "Deployment failed!" -ForegroundColor Red
    Write-Host "Check logs with: gcloud run logs read $SERVICE_NAME --region $REGION"
    exit 1
}

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
