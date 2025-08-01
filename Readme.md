# [1] 模型版本管理
# 1. 建立 Docker
docker build -t oralcancer_ai_template .

# [2] 圖片版本管理
# 1. 啟動 Docker 時掛載 Volume 

## 方式1: 使用 docker run 掛載目錄
docker run -d \
  -p 5000:5000 \
  -v C:\Users\Evan\Desktop\master\Side_project\OralCancerAPP_v3\static\images:/app/static/images \
  --name flask-oral-images \
  oralcancer_ai_template

## 方式2: 執行 docker 指令 (搭配 dockercompose.yaml)
docker compose up --build
docker compose -f infra/docker-compose.yml up --build (可以用-f parser來指定用特定的yml，向這個範例當中的docker-compose.yml是自定義的yaml檔案)
```
version: '1'
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

nginx.conf 
```
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

# 透過以下路徑訪問靜態圖
# http://localhost/images/2025-07-17/test.jpg

# 運行 Docker
docker run --env-file .env -p 5000:5000 oralcancer_ai_template