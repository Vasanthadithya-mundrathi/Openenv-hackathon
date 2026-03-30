FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY envs/soc_triage_env/ /app/

RUN pip install --no-cache-dir --retries 10 --default-timeout 300 -r /app/server/requirements.txt && \
  pip install --no-cache-dir --retries 10 --default-timeout 300 /app

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "soc_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
