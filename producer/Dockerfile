FROM python:3.11-slim

WORKDIR /app

# Dependances du producteur
RUN pip install --no-cache-dir \
    "confluent-kafka[avro,schemaregistry]==2.5.0" \
    fastavro==1.9.4 \
    pandas==2.2.2 \
    prometheus-client==0.20.0 \
    requests==2.32.3

# Copier le code, le schema et les donnees
COPY producer.py .
COPY schemas/ ./schemas/
COPY data/ ./data/

CMD ["python", "producer.py"]