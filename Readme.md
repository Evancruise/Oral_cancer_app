# Multi-Agent System — DINOv2 (Swin-Transformer–based visual analyzer) + Mask R-CNN (image analyzer) + RAG-based LLM (data retriever) with PyTorch

This project combines a visual analyzer, instance segmentation, and retrieval-augmented generation (RAG) to build a practical multi-agent pipeline for medical imaging and knowledge grounding.

---

## Table of Contents
- [Features](#features)
- [Use Without Installation](#use-without-installation)
- [Data Preparation](#data-preparation)
- [Step-by-Step Inspection (Notebooks)](#step-by-step-inspection-notebooks)
- [LLM + RAG Integration](#llm--rag-integration)
- [Quick Start](#quick-start)
- [Model Deployment](#model-deployment)
- [Image / Static Asset Versioning](#image--static-asset-versioning)
- [Push to Registries](#push-to-registries)
- [K8s Manifests](#k8s-manifests)
- [GitHub Actions (CI/CD)](#github-actions-cicd)
- [Nginx Config](#nginx-config)
- [Run with Docker & .env](#run-with-docker--env)
- [Citation](#citation)
- [App Screenshot](#app-screenshot)
- [References](#references)
- [Projects Using This Model](#projects-using-this-model)

---

## Features
- Mask R-CNN (FPN + ResNet-101) implementation under `mrcnn/`
- ParallelModel for multi-GPU training
- MS COCO AP evaluation utilities
- Step-by-step inspection notebooks (data/model/weights)
- Minimal RAG + LLM API (FastAPI/Flask), with mock LLM or external TGI/vLLM via env
- Dockerfiles and GitHub Actions CI/CD examples

---

## Use Without Installation
If you only need to run predictions, see **Quick Start** below. For training/evaluation, follow **Data Preparation** and **Notebooks** sections.

---

## Data Preparation
Datasets are split into **train/val/test**. Use `annotation_platform.py` for annotations.

```
annotations/
dataset/
 ├─ all/
 │  ├─ annotations_mask/
 │  ├─ annotations_pt/
 │  └─ images/
 ├─ test/
 │  ├─ annotations/
 │  └─ images/
 ├─ train/
 │  ├─ annotations/
 │  └─ images/
 └─ val/
    ├─ annotations/
    └─ images/
annotation_platform.py   # annotation tool
```

**Repository includes**
- Mask R-CNN (FPN + ResNet-101) under `mrcnn/`
- ParallelModel for multi-GPU training
- COCO-style AP evaluation
- Example training on custom datasets

> If you use this code in research, please consider citing (BibTeX below).

---

## Step-by-Step Inspection (Notebooks)
Three Jupyter notebooks provide visual, incremental inspection:

- `inspect_data.ipynb` — data, anchors, RPN steps  
- `inspect_model.ipynb` — proposals, box refinement, masks  
- `inspect_weights.ipynb` — weight histograms, health checks  

### 1) Anchor Sorting & Filtering
Visualizes RPN steps, positive/negative anchors, and anchor box refinement.

### 2) Bounding-Box Refinement
Shows final detection boxes (dotted) and refinements (solid).

### 3) Mask Generation
Generated masks (scaled and placed correctly):

![label](training_dataset/000001_00_label.png)
![mask](training_dataset/000001_01_mask.png)

### 4) Layer Activations
Check for pathologies (all-zero, noise).

### 5) Weight Histograms
Included in `inspect_weights.ipynb`.

### 6) TensorBoard Logging
Loss curves & checkpoints per epoch.  
![tensorboard](logs/detection_tensorboard.jpg)

### 7) Putting It All Together
![result](predicted_result/000001_00.png)

### 8) Metrics: Confusion Matrix / PR Curve
![cm](current_confusion_matrix.png)
![pr](precision_recall_curve_dinov2_yolov5.png)

---

## LLM + RAG Integration
**Minimal RAG + LLM API (FastAPI/Flask):**
- Lightweight retrieval with TF-IDF (scikit-learn)
- Mock LLM by default; optionally call TGI/vLLM via environment variables
- Dockerfile included
- GitHub Actions CI/CD workflow included

---

## Quick Start

### [1] Create virtualenv & Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### [2] Run app
```bash
python app_entry.py
```

### Example request
```bash
curl -X POST "http://localhost:8000/rag/answer" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is oral leukoplakia screening suggestion?"}'
```

---

## Model Deployment

### [1] 模型版本管理（上傳至 GCS）
1) 建立 bucket 並上傳模型：
```bash
gcloud storage buckets create gs://model-bucket-20250820 --location=asia-east1
gcloud storage cp dinov2_token_segmentation_final.pth \
  gs://model-bucket-20250820/models/
```

2) 查看檔案：
```bash
gcloud storage ls gs://model-bucket-20250820/models/
```

### [2] 建立 Docker 映像
> 你可以用 **Option A: Google Cloud Build** 或 **Option B: 本地 Docker**。

**Option A — Cloud Build → GCR**
```bash
gcloud config set project YOUR_PROJECT_ID
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/flask-app:latest
```

**Option B — Local Docker → GCR**
```bash
# Build
docker build -t gcr.io/YOUR_PROJECT_ID/flask-app:latest .

# (可選) 登入並推送
gcloud auth configure-docker
docker push gcr.io/YOUR_PROJECT_ID/flask-app:latest
```

> 說明：`gcr.io/<PROJECT_ID>/<IMAGE_NAME>:<TAG>`

### [3] 本地測試
請確保容器內服務監聽 `8000`（與 Quick Start 一致）：
```bash
docker run -p 8000:8000 gcr.io/YOUR_PROJECT_ID/flask-app:latest

curl -X POST "http://localhost:8000/rag/answer" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is leukoplakia?"}'
```

### [4] 部署到 Cloud Run
```bash
gcloud run deploy flask-dino-service \
  --image gcr.io/YOUR_PROJECT_ID/flask-app:latest \
  --platform managed \
  --region asia-east1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_BUCKET=model-bucket-20250820,MODEL_BLOB=models/dinov2_token_segmentation_final.pth
```

(若使用本地建置) 推送並確認：
```bash
docker push gcr.io/YOUR_PROJECT_ID/flask-app:latest
gcloud container images list-tags gcr.io/YOUR_PROJECT_ID/flask-app
```

---

## Image / Static Asset Versioning

### Method 1 — `docker run` with volume
```bash
docker run -d \
  -p 5000:5000 \
  -v C:\Users\Evan\Desktop\master\Side_project\OralCancerAPP_main\static\images:/app/static/images \
  --name flask-oral-images \
  oralcancer_ai_template
```

### Method 2 — `docker compose`
```bash
docker compose up --build
# or specify file
docker compose -f infra/docker-compose.yml up --build
```

---

## Push to Registries

### [A] DockerHub
```bash
docker login -u <USERNAME>
docker tag rag-ai-api:latest <USERNAME>/rag-ai-api:latest
docker push <USERNAME>/rag-ai-api:latest
```

### [B] GCP Artifact Registry
> 建議改用 Artifact Registry（較新），路徑格式：  
> `asia-east1-docker.pkg.dev/<PROJECT_ID>/<REPO>/<IMAGE>:<TAG>`

```bash
# 設定 Artifact Registry 認證
gcloud auth configure-docker asia-east1-docker.pkg.dev

# 重新標記
docker tag rag-ai-api:latest \
  asia-east1-docker.pkg.dev/<PROJECT_ID>/<REPO>/rag-ai-api:latest

# 推送
docker push \
  asia-east1-docker.pkg.dev/<PROJECT_ID>/<REPO>/rag-ai-api:latest
```

---

## K8s Manifests

### `docker-compose.yaml` (for local dev)
```yaml
version: "3.9"
services:
  flask-oral-images:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./uploads/images:/app/static/images

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./uploads/images:/usr/share/nginx/html/images
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
```

### `deployment.yaml` (Kubernetes)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-ai-api
  namespace: staging
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-ai-api
  template:
    metadata:
      labels:
        app: rag-ai-api
    spec:
      containers:
      - name: rag-ai-api
        image: <USERNAME>/rag-ai-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: USE_TGI
          value: "0"
---
apiVersion: v1
kind: Service
metadata:
  name: rag-ai-service
  namespace: staging
spec:
  selector:
    app: rag-ai-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### 部署到 **staging**
```bash
kubectl create namespace staging
kubectl apply -f deployment.yaml
kubectl get pods -n staging
kubectl get svc  -n staging
```

### 滾動更新 / 回滾
```bash
# Rolling update
kubectl set image deployment/rag-ai-api rag-ai-api=<USERNAME>/rag-ai-api:v2 -n staging
kubectl rollout status deployment/rag-ai-api -n staging

# Rollback
kubectl rollout undo deployment/rag-ai-api -n staging
```

### 部署到 **production**（staging 驗證後）
```bash
kubectl create namespace production
kubectl apply -f deployment.yaml --namespace=production
```

---

## GitHub Actions (CI/CD)

`.github/workflows/deploy.yml`
```yaml
name: CI/CD

on:
  push:
    branches: [ "main" ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Docker login
        run: echo "${{ secrets.DOCKER_PASS }}" | docker login -u ${{ secrets.DOCKER_USER }} --password-stdin

      - name: Build image
        run: docker build -t ${{ secrets.DOCKER_USER }}/rag-ai-api:${{ github.sha }} .

      - name: Push image
        run: docker push ${{ secrets.DOCKER_USER }}/rag-ai-api:${{ github.sha }}

      - name: Deploy to K8s (Staging)
        uses: azure/k8s-deploy@v4
        with:
          manifests: |
            ./k8s/deployment.yaml
          images: |
            ${{ secrets.DOCKER_USER }}/rag-ai-api:${{ github.sha }}
          namespace: staging
```

---

## Nginx Config
`nginx.conf`
```nginx
server {
    listen 80;

    location /images/ {
        alias /usr/share/nginx/html/images/;
        autoindex on;
    }

    location / {
        proxy_pass http://flask-oral-images:5000;
    }
}
```

---

## Run with Docker & .env
```bash
docker run --env-file .env -p 5000:5000 oralcancer_ai_template
```

---

## Citation
If this repository helps your research, please cite:

```bibtex
@misc{matterport_maskrcnn_2017,
  title        = {Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author       = {Waleed Abdulla},
  year         = {2017},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/matterport/Mask_RCNN}}
}
```

---

## App Screenshot
![app](app_screenshot.png)

---

## References

### [Usiigaci: Label-free Cell Tracking in Phase Contrast Microscopy](https://github.com/oist/usiigaci)  
A project from Japan to automatically track cells in a microfluidics platform. Paper is pending, but the source code is released.

---

## Projects Using This Model
If you extend this model to other datasets or build projects that use it, we’d love to hear from you!