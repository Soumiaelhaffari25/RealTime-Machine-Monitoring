# 🛠️ Real-Time Machine Monitoring — NASA C-MAPSS

End-to-end **machine monitoring** pipeline for a fleet of aircraft engines, based on the **NASA C-MAPSS** dataset (Turbofan Engine Degradation Simulation). The system ingests sensor data in real time, predicts each engine's **Remaining Useful Life (RUL)**, detects anomalous behavior, and surfaces everything through dashboards and alerts.

The project spans the full chain: **streaming**, **machine learning**, **storage**, **observability**, **Kubernetes orchestration**, and **CI/CD**.

---

## 📑 Table of Contents

- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Machine Learning Models](#-machine-learning-models)
- [Quick Start (Docker Compose)](#-quick-start-docker-compose)
- [Kubernetes Deployment](#-kubernetes-deployment)
- [Observability](#-observability)
- [CI/CD](#-cicd)
- [Testing](#-testing)
- [Dataset](#-dataset)

---

## 🏗️ Architecture

The pipeline follows a **decoupled, streaming-oriented architecture**: each component communicates through Kafka or internal services, which keeps it robust and scalable.

```
┌───────────┐   Avro    ┌─────────┐          ┌──────────┐   scoring   ┌───────────┐
│ Producer  │ ────────> │  Kafka  │ ───────> │  Spark   │ ──────────> │  MongoDB  │
│ (sensors) │           │ (topic) │          │Streaming │     ML      │ (storage) │
└───────────┘           └─────────┘          └──────────┘             └─────┬─────┘
                             │                                              │
                       ┌─────┴──────┐                                       │
                       │   Schema   │                                 ┌─────┴─────┐
                       │  Registry  │                                 │    API    │
                       └────────────┘                                 │  FastAPI  │
                                                                      └─────┬─────┘
                                                                            │ /metrics
                                                                      ┌─────┴──────┐
                                                                      │ Prometheus │
                                                                      └─────┬──────┘
                                                                            │
                                                                      ┌─────┴──────┐
                                                                      │  Grafana   │
                                                                      │ dashboards │
                                                                      │ + alerts   │
                                                                      └────────────┘
```

**Data flow:**
1. The **producer** simulates a fleet of engines and publishes sensor readings (**Avro** format) to **Kafka**.
2. **Spark Structured Streaming** consumes the stream, applies the ML models (RUL + anomaly detection), and writes the scored results to **MongoDB**.
3. The **FastAPI** service exposes fleet data over REST and publishes **Prometheus** metrics.
4. **Prometheus** collects the metrics; **Grafana** visualizes them (business & technical dashboards) and triggers **alerts**.

---

## 🧰 Tech Stack

| Domain | Technologies |
|---|---|
| **Streaming** | Apache Kafka (KRaft), Avro, Schema Registry |
| **Processing** | Apache Spark (Structured Streaming) |
| **Machine Learning** | scikit-learn, XGBoost, MLflow |
| **Storage** | MongoDB |
| **API** | FastAPI, Uvicorn |
| **Observability** | Prometheus, Grafana |
| **Orchestration** | Kubernetes (Kind), Strimzi (Kafka operator), Helm |
| **CI/CD** | GitHub Actions, Docker |
| **Language** | Python 3.11 |

---

## ✨ Features

- **Realistic fleet simulation**: staggered-age renewal model (engines reaching end of life are replaced), keeping the fleet varied and stable over time.
- **Real-time RUL prediction** (remaining useful life in cycles).
- **Unsupervised anomaly detection** on sensor behavior.
- **Data-driven criticality thresholds**: CRITICAL (RUL < 30), WATCH (30–60), OK (≥ 60).
- **Two Grafana dashboards**: one business-facing (fleet health) and one technical (pipeline health).
- **Alerting system**: degraded fleet, imminent engine failure, pipeline outage.
- **Full Kubernetes deployment** with a Kafka operator and persistent storage.
- **CI/CD pipeline**: automated tests and Docker image builds on every push.

---

## 📂 Project Structure

```
.
├── .github/workflows/      # CI/CD pipelines (GitHub Actions)
│   ├── ci.yml              # Tests + code quality
│   └── cd.yml              # Docker image builds
├── api/                    # FastAPI service (MongoDB → REST bridge + Prometheus metrics)
│   ├── Dockerfile
│   └── grafana_bridge.py
├── dashboards/             # Grafana dashboards (importable JSON)
│   ├── grafana_business.json
│   └── grafana_technical.json
├── k8s/                    # Kubernetes manifests
│   ├── kind-cluster.yaml   # Local cluster configuration
│   ├── kafka.yaml          # Kafka cluster (Strimzi)
│   ├── kafka-topic.yaml    # sensor-readings topic
│   ├── schema-registry.yaml
│   ├── mongodb.yaml        # StatefulSet + persistent volume
│   ├── producer.yaml
│   ├── spark.yaml
│   ├── api.yaml
│   ├── prometheus.yaml
│   └── grafana.yaml
├── ml/                     # Model training 
│   ├── build_rul_cmapss.py       # Data preparation + feature engineering
│   ├── train_rul_cmapss.py       # RUL model training
│   ├── train_detection_cmapss.py # Anomaly detection training
│   ├── explore.py                # Dataset exploration
│   ├── demo_pred.py              # Prediction demo
│   └── requirements-ml.txt
├── producer/               # Real-time producer
│   ├── Dockerfile
│   ├── producer.py
│   └── schemas/reading.avsc
├── spark_job/              # Spark scoring job
│   ├── Dockerfile
│   ├── streaming_job.py    # Kafka consumption + scoring + Mongo write
│   ├── scoring.py          # Model loading + prediction
│   ├── mongo_sink.py       # MongoDB writer
│   ├── setup_indexes.py
│   └── schemas/reading.avsc
├── prometheus/             # Prometheus configuration + alerts
│   ├── prometheus.yml
│   ├── alert_rules.yml
│   └── alertmanager.yml
├── tests/                  # Unit tests (pytest)
│   └── test_state_from_rul.py
├── docker-compose.yml      # Full local deployment
├── requirements.txt
└── README.md
```

---

## 🤖 Machine Learning Models

Models are trained **offline** on the C-MAPSS FD001 dataset, with experiment tracking via **MLflow**. RUL is capped at 125 cycles (standard C-MAPSS convention), and the scaler is fit on the training data only to prevent data leakage.

### RUL Prediction (regression)

Two models were compared fairly (RandomizedSearchCV, cross-validation):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| **RandomForest** ✅ | 0.826 | 11.87 | 16.74 |
| XGBoost | 0.825 | — | 16.75 |

**RandomForest** was selected (near-identical, slightly better performance).

### Anomaly Detection (unsupervised)

Two approaches compared via a separation score (% anomalies at end of life − % at start of life):

| Model | Separation score |
|---|---|
| **Isolation Forest** ✅ | 89.7 |
| Local Outlier Factor | 89.4 |

**Isolation Forest** was selected. The two models cross-validate each other in production: engines with the lowest RUL are also the ones flagged as anomalous.

### Regenerating the models

> ℹ️ Trained models (`.pkl`) are not versioned (large files). To regenerate them:

```bash
pip install -r ml/requirements-ml.txt
python ml/build_rul_cmapss.py
python ml/train_rul_cmapss.py
python ml/train_detection_cmapss.py
```

---

## 🚀 Quick Start (Docker Compose)

The simplest way to run the entire pipeline locally.

### Prerequisites
- Docker & Docker Compose
- Python 3.11 (for the producer and API run locally)
- The C-MAPSS FD001 dataset in `ml/data/` (downloaded separately)

### Run

```bash
# 1. Start the infrastructure (Kafka, Schema Registry, MongoDB, Spark, Prometheus, Grafana)
docker compose up -d

# 2. Generate the ML models (if not already done)
python ml/build_rul_cmapss.py && python ml/train_rul_cmapss.py && python ml/train_detection_cmapss.py

# 3. Start the producer
python producer/producer.py

# 4. Start the API
uvicorn api.grafana_bridge:app --host 0.0.0.0 --port 8010
```

### Access the interfaces
| Service | URL |
|---|---|
| Grafana | http://localhost:3000 (admin / admin) |
| Prometheus | http://localhost:9090 |
| API (docs) | http://localhost:8010/docs |

---

## ☸️ Kubernetes Deployment

The full pipeline can be deployed on a Kubernetes cluster. Kafka is managed by the **Strimzi operator**, MongoDB by a **StatefulSet** with a persistent volume, and the applications (producer, API, Spark) are containerized.

### Prerequisites
- Docker, [Kind](https://kind.sigs.k8s.io/), kubectl, Helm

### Deploy

```bash
# 1. Create the local cluster
kind create cluster --config k8s/kind-cluster.yaml

# 2. Create the namespace
kubectl create namespace cmapss

# 3. Install the Kafka operator (Strimzi)
helm repo add strimzi https://strimzi.io/charts/
helm install strimzi-operator strimzi/strimzi-kafka-operator -n cmapss --set crds.enabled=true

# 4. Deploy the infrastructure
kubectl apply -f k8s/kafka.yaml
kubectl apply -f k8s/kafka-topic.yaml
kubectl apply -f k8s/schema-registry.yaml
kubectl apply -f k8s/mongodb.yaml

# 5. Build and load the application images
docker build -t cmapss-producer:1.0 ./producer && kind load docker-image cmapss-producer:1.0 --name cmapss
docker build -t cmapss-api:1.0 ./api && kind load docker-image cmapss-api:1.0 --name cmapss
docker build -t cmapss-spark:1.0 ./spark_job && kind load docker-image cmapss-spark:1.0 --name cmapss

# 6. Deploy the applications and monitoring
kubectl apply -f k8s/producer.yaml
kubectl apply -f k8s/spark.yaml
kubectl apply -f k8s/api.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml

# 7. Verify
kubectl get pods -n cmapss
```

### Access (via NodePort)
| Service | URL |
|---|---|
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| API | via `kubectl port-forward -n cmapss deploy/api 8010:8010` |

---

## 📊 Observability

### Grafana Dashboards

- **Business dashboard** (`grafana_business.json`): fleet distribution, RUL per engine, detailed engine table, time-series trends.
- **Technical dashboard** (`grafana_technical.json`): ingestion throughput, message volume, end-to-end pipeline health (sent vs. stored throughput).

Import: in Grafana, add a **Prometheus** data source (`http://prometheus:9090`), then import the JSON files from the `dashboards/` folder.

### Alerts

Three alert rules are configured:
- **Too many critical engines** (`fleet_critical_engines > 8`)
- **Imminent engine failure** (`fleet_rul_min < 5`)
- **Producer down** (`up{job="cmapss-producer"} < 1`)

---

## 🔄 CI/CD

Automated pipeline via **GitHub Actions**, triggered on every push to `main`:

- **CI** (`ci.yml`): linting (flake8) + unit tests (pytest).
- **CD** (`cd.yml`): Docker image builds (producer, API) to validate buildability.

---

## 🧪 Testing

Unit tests with **pytest**:

```bash
pip install pytest
pytest tests/ -v
```

Tests cover, among other things, the engine classification logic based on criticality thresholds (`state_from_rul`).

---

## 📚 Dataset

This project uses the **NASA C-MAPSS** dataset (Turbofan Engine Degradation Simulation Data Set), subset **FD001** (100 engines, 21 sensors). The dataset is publicly available (NASA Prognostics Center of Excellence / Kaggle).

---

## Author
Soumia El Haffari
