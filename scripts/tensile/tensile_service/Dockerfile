FROM python:3.10-slim

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create storage directory
RUN mkdir -p storage

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]