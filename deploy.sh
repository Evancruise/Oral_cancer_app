#!/bin/bash

#安裝後，記得登入：
#gcloud auth login
#gcloud config set project your-project-id

PROJECT_ID="evan-ml-project"  # 請確認是否是你的 GCP 專案 ID
SERVICE_NAME="evan-ml-project"
REGION="asia-east1"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"  # 推送到 GCR
PORT=8080
MEMORY="1Gi"
TIMEOUT="600s"
MAX_INSTANCES="3"

echo "📦 建立 Docker 映像..."
gcloud builds submit --tag $IMAGE

echo "☁️ 部署到 Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --port $PORT \
  --memory $MEMORY \
  --timeout $TIMEOUT \
  --max-instances $MAX_INSTANCES \
  --allow-unauthenticated

echo "✅ 部署完成！可從 GCP Console 或以下網址查看："
gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)"