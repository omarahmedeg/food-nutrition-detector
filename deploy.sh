#!/bin/bash

# Google Cloud Run Deployment Script
# This script builds and deploys the Food Nutrition Detector API to Cloud Run

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if required files exist
echo "Checking prerequisites..."

if [ ! -f "ml-model/saved_models/food101_classifier.pth" ]; then
    echo -e "${RED}ERROR: Model file not found!${NC}"
    echo "Please ensure ml-model/saved_models/food101_classifier.pth exists"
    exit 1
fi

if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}ERROR: Dockerfile not found!${NC}"
    exit 1
fi

# Get project ID
echo -e "${YELLOW}Enter your Google Cloud Project ID:${NC}"
read PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}ERROR: Project ID cannot be empty${NC}"
    exit 1
fi

# Configuration
IMAGE_NAME="food-nutrition-detector"
SERVICE_NAME="food-nutrition-api"
REGION="us-central1"  # Change if needed

echo -e "${GREEN}Configuration:${NC}"
echo "  Project ID: $PROJECT_ID"
echo "  Image Name: $IMAGE_NAME"
echo "  Service Name: $SERVICE_NAME"
echo "  Region: $REGION"
echo ""

# Set project
echo -e "${YELLOW}Setting Google Cloud project...${NC}"
gcloud config set project $PROJECT_ID

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Docker build failed${NC}"
    exit 1
fi

# Push to Google Container Registry
echo -e "${YELLOW}Pushing image to Google Container Registry...${NC}"
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Docker push failed${NC}"
    exit 1
fi

# Deploy to Cloud Run
echo -e "${YELLOW}Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
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

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Deployment successful!${NC}"
    echo ""
    echo -e "${GREEN}Getting service URL...${NC}"
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')
    echo ""
    echo -e "${GREEN}Your API is deployed at:${NC}"
    echo -e "${GREEN}$SERVICE_URL${NC}"
    echo ""
    echo -e "${YELLOW}Test your API:${NC}"
    echo "  Health check: curl $SERVICE_URL/api/health"
    echo "  API docs: $SERVICE_URL"
else
    echo -e "${RED}Deployment failed!${NC}"
    echo "Check logs with: gcloud run logs read $SERVICE_NAME --region $REGION"
    exit 1
fi
