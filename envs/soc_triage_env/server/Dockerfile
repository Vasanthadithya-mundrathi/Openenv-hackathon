FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY envs/soc_triage_env/ /app/envs/soc_triage_env/
COPY baseline.py /app/baseline.py

RUN pip install --no-cache-dir -r /app/envs/soc_triage_env/server/requirements.txt && \
    pip install --no-cache-dir /app/envs/soc_triage_env

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "soc_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
