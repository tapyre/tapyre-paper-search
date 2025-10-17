#!/usr/bin/env bash
set -euo pipefail

# --- Warten auf MySQL ---
for i in $(seq 1 90); do
  if nc -z "${MYSQL_HOST:-mysql}" "${MYSQL_PORT:-3306}" 2>/dev/null; then
    echo "MySQL up"
    break
  fi
  sleep 1
  if [ "$i" -eq 90 ]; then
    echo "ERROR: MySQL not reachable after 90s" >&2
    exit 1
  fi
done

# --- Warten auf Qdrant ---
for i in $(seq 1 90); do
  if nc -z "${QDRANT_HOST:-qdrant}" "${QDRANT_HTTP_PORT:-6333}" 2>/dev/null; then
    echo "Qdrant up"
    break
  fi
  sleep 1
  if [ "$i" -eq 90 ]; then
    echo "ERROR: Qdrant not reachable after 90s" >&2
    exit 1
  fi
done

exec gunicorn -w ${GUNICORN_WORKERS:-3} -k ${GUNICORN_WORKER_CLASS:-gthread} \
  --threads ${GUNICORN_THREADS:-4} \
  --timeout ${GUNICORN_TIMEOUT:-120} \
  --bind 0.0.0.0:${API_PORT:-8000} \
  src.wsgi:app
