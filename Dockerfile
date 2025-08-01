# Base image
FROM python:3.9-slim

# Set workdir
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy files
COPY . .

# Construct directory to store temp images
RUN mkdir -p static/images

# Expose port
EXPOSE 8080

# Use gunicorn to serve Flask
CMD ["python", "app.py"]

