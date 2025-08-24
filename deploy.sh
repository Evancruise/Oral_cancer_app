#!/bin/bash
set -e

# ===== 修改成你的專案資訊 =====
PROJECT_ID="oral-cancer-ai"
REGION="asia-east1"
REPO="flask-app-repo"
IMAGE_NAME="flask-app"
TAG="test"

echo "⚡ 使用專案: $PROJECT_ID"
echo "⚡ 區域: $REGION"
echo "⚡ Repository: $REPO"
echo "⚡ Image: $IMAGE_NAME:$TAG"
echo "-----------------------------------------"

# 1. 建立 Artifact Registry Repository (如果不存在才建)
if gcloud artifacts repositories describe $REPO --location=$REGION --project=$PROJECT_ID >/dev/null 2>&1; then
  echo "✅ Repository $REPO 已存在"
else
  echo "📦 建立 Artifact Registry repository..."
  gcloud artifacts repositories create $REPO \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repo for $IMAGE_NAME" \
    --project=$PROJECT_ID
  echo "✅ Repository $REPO 建立完成"
fi
echo "-----------------------------------------"

# 2. 配置 Docker 登入 Artifact Registry
echo "🔑 設定 Artifact Registry docker 認證..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev -q

# 3. Build Image
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"
echo "🐳 建立 Docker image: $IMAGE_URI"
docker build -t $IMAGE_URI .

# 4. Push Image
echo "📤 推送到 Artifact Registry..."
docker push $IMAGE_URI

# 5. 驗證 Repo 內容
echo "🔎 列出 Artifact Registry images:"
gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}

echo "🎉 測試完成！Image 已經 push 到 Artifact Registry: $IMAGE_URI"