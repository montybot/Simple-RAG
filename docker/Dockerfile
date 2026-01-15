# docker/Dockerfile
FROM python:3.12-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY docker/requirements.txt .

# Install UV and Python packages in a single RUN layer
# This ensures UV is available immediately after installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install -r requirements.txt  
# --no-cache

# Set PATH for UV (for any future commands)
ENV PATH="/root/.local/bin:${PATH}"

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/indices /app/logs

# Expose API and Streamlit ports
EXPOSE 8000 8001

# Entry point - Start both FastAPI and Streamlit
CMD ["/bin/bash", "/app/scripts/start_services.sh"]
