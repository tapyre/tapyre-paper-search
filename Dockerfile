FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl tzdata ca-certificates netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt && pip install gunicorn

COPY . .

# User anlegen
RUN useradd -m -u 1000 appuser

# Logs- UND State-Verzeichnis anlegen und auf appuser setzen
RUN mkdir -p /app/logs /app/state \
 && chown -R appuser:appuser /app/logs /app/state

# Line-Endings fixen & Entrypoints ausführbar machen
RUN sed -i 's/\r$//' docker/entrypoints/run_api.sh \
 && sed -i 's/\r$//' docker/entrypoints/run_pipeline.sh \
 && chmod +x docker/entrypoints/run_api.sh docker/entrypoints/run_pipeline.sh

USER appuser

CMD ["bash", "-lc", "python -m src.pipeline"]
