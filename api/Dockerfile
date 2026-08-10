FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    pymongo==4.8.0 \
    prometheus-client==0.20.0

COPY grafana_bridge.py .

CMD ["uvicorn", "grafana_bridge:app", "--host", "0.0.0.0", "--port", "8010"]