@echo off
set PROJECT_ID=evan-ml-project
set SERVICE_NAME=evan-ml-project
set REGION=asia-east1
set IMAGE=gcr.io/%PROJECT_ID%/%SERVICE_NAME%

echo Building Docker image...
gcloud builds submit --tag %IMAGE%

echo Deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% ^
  --image %IMAGE% ^
  --platform managed ^
  --region %REGION% ^
  --allow-unauthenticated ^
  --port 8080 ^
  --memory 1Gi ^
  --timeout 600s ^
  --max-instances 2

pause
