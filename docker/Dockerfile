# docker/Dockerfile
FROM python:3.12-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Création du répertoire de travail
WORKDIR /app

# Copie des dépendances
COPY docker/requirements.txt .

# Installation des packages Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copie du code source
COPY src/ ./src/
COPY scripts/ ./scripts/

# Création des répertoires de données
RUN mkdir -p /app/data/raw /app/data/processed /app/data/indices /app/logs

# Exposition du port API
EXPOSE 8000

# Point d'entrée
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
