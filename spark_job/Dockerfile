FROM python:3.11-slim-bookworm

WORKDIR /app

# ----- Java (requis par Spark) -----
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless procps && \
    rm -rf /var/lib/apt/lists/*

# JAVA_HOME pour que Spark trouve la JVM
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# ----- PySpark + libs ML (memes versions qu'a l'entrainement) -----
RUN pip install --no-cache-dir \
    pyspark==3.5.1 \
    scikit-learn==1.5.2 \
    xgboost==2.1.1 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    joblib==1.4.2 \
    fastavro==1.9.4 \
    pyarrow==17.0.0 \
    pymongo==4.8.0

COPY streaming_job.py scoring.py mongo_sink.py ./
COPY schemas/ ./schemas/
COPY model/ ./model/
RUN mkdir -p /app/checkpoints
CMD ["python", "streaming_job.py"]

