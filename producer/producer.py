import os
import time
import random
import pandas as pd

from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import SerializationContext, MessageField
from prometheus_client import start_http_server, Counter, Gauge

# ------------------- Configuration -------------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
SCHEMA_REGISTRY_URL = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")
TOPIC = os.getenv("TOPIC", "sensor-readings")
DATA_FILE = os.getenv("DATA_FILE", "ml/data/train_FD001.txt")
SCHEMA_FILE = os.getenv("SCHEMA_FILE", "producer/schemas/reading.avsc")

TICK_SECONDS = float(os.getenv("TICK_SECONDS", "1"))
FLEET_SIZE = int(os.getenv("FLEET_SIZE", "30"))
MAX_AGE_CYCLES = int(os.getenv("MAX_AGE_CYCLES", "130"))
PROM_PORT = int(os.getenv("PROM_PORT", "8000"))
SEED = int(os.getenv("SEED", "42"))

COLS = ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"] + \
       [f"sensor_{i}" for i in range(1, 22)]

#Metriques Prometheus 
MESSAGES_SENT = Counter("producer_messages_sent_total", "Messages envoyes")
MESSAGES_FAILED = Counter("producer_messages_failed_total", "Messages en echec")
ACTIVE_ENGINES = Gauge("producer_active_engines", "Postes moteurs actifs")
REPLACEMENTS = Counter("producer_replacements_total", "Moteurs remplaces (fin de vie)")
# Seuil de remplacement preventif : on remplace un moteur quand son RUL reel
# descend sous ce seuil (maintenance predictive : intervention avant panne).
REPLACE_AT_RUL = int(os.getenv("REPLACE_AT_RUL", "15"))


def load_schema(path):
    with open(path) as f:
        return f.read()


def load_data(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def delivery_report(err, msg):
    if err is not None:
        MESSAGES_FAILED.inc()


def build_record(row, poste_id):
    record = {
        "unit_id": int(poste_id),
        "cycle": int(row["cycle"]),
        "event_time": int(time.time() * 1000),
        "setting_1": float(row["setting_1"]),
        "setting_2": float(row["setting_2"]),
        "setting_3": float(row["setting_3"]),
    }
    for i in range(1, 22):
        record[f"sensor_{i}"] = float(row[f"sensor_{i}"])
    return record


def main():
    random.seed(SEED)   # reproductible
    start_http_server(PROM_PORT)
    print(f"Metriques Prometheus sur http://localhost:{PROM_PORT}/metrics")

    schema_str = load_schema(SCHEMA_FILE)
    sr_client = SchemaRegistryClient({"url": SCHEMA_REGISTRY_URL})
    serializer = AvroSerializer(sr_client, schema_str)
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
    ctx = SerializationContext(TOPIC, MessageField.VALUE)

    df = load_data(DATA_FILE)
    source_ids = sorted(df["unit_id"].unique())
    trajectories = {
        u: df[df["unit_id"] == u].sort_values("cycle").to_dict("records")
        for u in source_ids
    }
    print(f"Flotte de {FLEET_SIZE} postes | {len(source_ids)} trajectoires disponibles")

    def new_trajectory():
        return random.choice(source_ids)

    # --- Initialisation a AGES ECHELONNES UNIFORMEMENT ---
    # Le poste i demarre a une fraction i/FLEET_SIZE de la vie de sa trajectoire.
    fleet = {}
    for i, poste in enumerate(range(1, FLEET_SIZE + 1)):
        traj_id = new_trajectory()
        cycles = trajectories[traj_id]
        # Age initial etale uniformement dans la fenetre [0, MAX_AGE_CYCLES]
        start = int((i / FLEET_SIZE) * MAX_AGE_CYCLES)
        start = min(start, len(cycles) - 1)
        fleet[poste] = {"traj": cycles, "idx": start}

    ACTIVE_ENGINES.set(FLEET_SIZE)
    tick = 0
    total_sent = 0

    print("Flux continu a renouvellement (flotte equilibree en permanence)")
    print("Ctrl+C pour arreter\n")
    while True:
        replaced = []
        for poste, state in fleet.items():
            cycles = state["traj"]
            idx = state["idx"]

            record = build_record(cycles[idx], poste)
            payload = serializer(record, ctx)
            producer.produce(topic=TOPIC, key=str(poste), value=payload,
                             on_delivery=delivery_report)
            MESSAGES_SENT.inc()
            total_sent += 1

            state["idx"] += 1
            # Remplacement quand le moteur atteint l'age max OU la fin de sa trajectoire.
            age = state["idx"]
            if age >= MAX_AGE_CYCLES or state["idx"] >= len(cycles):
                traj_id = new_trajectory()
                fleet[poste] = {"traj": trajectories[traj_id], "idx": 0}
                REPLACEMENTS.inc()
                replaced.append(poste)
        producer.poll(0)

        if tick % 15 == 0 or replaced:
            msg = f"tick {tick:4d} | {FLEET_SIZE} postes | total {total_sent}"
            if replaced:
                msg += f" | remplaces: {replaced}"
            print(msg)

        tick += 1
        time.sleep(TICK_SECONDS)


if __name__ == "__main__":
    main()