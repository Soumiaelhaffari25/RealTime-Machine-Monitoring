import os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from prometheus_client import (
    Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry,
)

#Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "monitoring")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "scored_readings")

RUL_CRITICAL = int(os.getenv("RUL_CRITICAL", "30"))
RUL_WARNING = int(os.getenv("RUL_WARNING", "60"))

app = FastAPI(title="CMAPSS Monitoring API")

# CORS : autorise les appels navigateur (dashboard, tests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

client = MongoClient(MONGO_URI)
collection = client[MONGO_DB][MONGO_COLLECTION]


def state_from_rul(rul):
    """Determine l'etat d'un moteur a partir de son RUL predit."""
    if rul < RUL_CRITICAL:
        return "CRITIQUE"
    elif rul < RUL_WARNING:
        return "A_SURVEILLER"
    return "OK"


def get_fleet_latest():
    """Dernier cycle de chaque moteur, trie par RUL croissant (plus critiques d'abord)."""
    pipeline = [
        {"$sort": {"unit_id": 1, "cycle": 1}},
        {"$group": {
            "_id": "$unit_id",
            "unit_id": {"$last": "$unit_id"},
            "cycle": {"$last": "$cycle"},
            "rul_pred": {"$last": "$rul_pred"},
            "is_anomaly": {"$last": "$is_anomaly"},
        }},
        {"$sort": {"rul_pred": 1}},
    ]
    fleet = list(collection.aggregate(pipeline))
    for m in fleet:
        m.pop("_id", None)
        m["state"] = state_from_rul(m["rul_pred"])
    return fleet


#Endpoints JSON
@app.get("/health")
def health():
    """Verifie que l'API et MongoDB repondent."""
    return {"status": "ok", "documents": collection.estimated_document_count()}


@app.get("/fleet/latest")
def fleet_latest():
    """Dernier cycle connu de chaque moteur, avec RUL, etat, anomalie."""
    return get_fleet_latest()


@app.get("/fleet/summary")
def fleet_summary():
    """Repartition de la flotte par etat (pour KPI et camembert)."""
    fleet = get_fleet_latest()
    summary = {"CRITIQUE": 0, "A_SURVEILLER": 0, "OK": 0}
    for m in fleet:
        summary[m["state"]] += 1
    return {
        "total": len(fleet),
        "critique": summary["CRITIQUE"],
        "a_surveiller": summary["A_SURVEILLER"],
        "ok": summary["OK"],
        "anomalies": sum(1 for m in fleet if m["is_anomaly"]),
    }


@app.get("/engine/{unit_id}/history")
def engine_history(unit_id: int, limit: int = 200):
    """Historique des cycles d'un moteur (evolution du RUL dans le temps)."""
    cursor = (
        collection.find(
            {"unit_id": unit_id},
            {"_id": 0, "cycle": 1, "rul_pred": 1, "is_anomaly": 1},
        ).sort("cycle", 1).limit(limit)
    )
    return list(cursor)


@app.get("/engines/critical")
def critical_engines():
    """Liste des moteurs actuellement en etat CRITIQUE."""
    return [m for m in get_fleet_latest() if m["state"] == "CRITIQUE"]


#Endpoint Prometheus
@app.get("/metrics")
def metrics():
    """Expose les metriques metier au format Prometheus (scrape par Prometheus)."""
    registry = CollectorRegistry()

    # Metriques agregees de la flotte
    g_total = Gauge("fleet_total_engines", "Nombre total de moteurs suivis", registry=registry)
    g_crit = Gauge("fleet_critical_engines", "Moteurs en etat CRITIQUE (RUL<30)", registry=registry)
    g_warn = Gauge("fleet_warning_engines", "Moteurs A SURVEILLER (30<=RUL<60)", registry=registry)
    g_ok = Gauge("fleet_ok_engines", "Moteurs sains (RUL>=60)", registry=registry)
    g_anom = Gauge("fleet_anomalies", "Moteurs avec anomalie detectee", registry=registry)
    g_rul_avg = Gauge("fleet_rul_avg", "RUL moyen de la flotte", registry=registry)
    g_rul_min = Gauge("fleet_rul_min", "RUL minimum de la flotte", registry=registry)
    g_total_docs = Gauge("pipeline_documents_total", "Total de cycles scores stockes dans MongoDB", registry=registry)

    # Metriques par moteur (label unit_id) pour le tableau et les barres
    g_rul_engine = Gauge("engine_rul", "RUL predit par moteur", ["unit_id"], registry=registry)
    g_anom_engine = Gauge("engine_anomaly", "Anomalie par moteur (1=oui, 0=non)", ["unit_id"], registry=registry)
    g_cycle_engine = Gauge("engine_cycle", "Cycle courant par moteur", ["unit_id"], registry=registry)

    fleet = get_fleet_latest()
    total = len(fleet)
    crit = sum(1 for m in fleet if m["state"] == "CRITIQUE")
    warn = sum(1 for m in fleet if m["state"] == "A_SURVEILLER")
    ok = sum(1 for m in fleet if m["state"] == "OK")
    anom = sum(1 for m in fleet if m["is_anomaly"])
    ruls = [m["rul_pred"] for m in fleet] or [0]

    g_total.set(total)
    g_crit.set(crit)
    g_warn.set(warn)
    g_ok.set(ok)
    g_anom.set(anom)
    g_rul_avg.set(sum(ruls) / len(ruls))
    g_rul_min.set(min(ruls))
    g_total_docs.set(collection.estimated_document_count())

    for m in fleet:
        uid = str(m["unit_id"])
        g_rul_engine.labels(unit_id=uid).set(m["rul_pred"])
        g_anom_engine.labels(unit_id=uid).set(1 if m["is_anomaly"] else 0)
        g_cycle_engine.labels(unit_id=uid).set(m["cycle"])

    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)