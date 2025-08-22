# Base image
FROM python:3.10

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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
ENV PORT=8080

# Use gunicorn to serve Flask
# CMD ["python", "app.py"]
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "-b 0.0.0.0:${PORT}", "app:app"]
