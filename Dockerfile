# ──────────────────────────────────────────────────────────────
# ML Auto-Tagging Service — Dockerfile
# Base: python:3.11-slim
# Port: 8002
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Install curl untuk healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements dulu (caching layer)
COPY ml_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code ml_service & ml_app (termasuk models)
COPY ml_service/ ./ml_service/
COPY ml_app/ ./ml_app/

# Working directory di ml_service agar import relative benar
WORKDIR /app/ml_service

EXPOSE 8002

# Production: tanpa --reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
