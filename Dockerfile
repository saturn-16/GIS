FROM python:3.11-slim

# Install system libraries needed by OpenCV and matplotlib
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Hugging Face Spaces requires port 7860
EXPOSE 7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
