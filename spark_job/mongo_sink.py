import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
MONGO_DB = os.getenv("MONGO_DB", "monitoring")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "scored_readings")

# Client Mongo reutilise entre les batchs (un par worker)
_client = None


def _get_collection():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[MONGO_DB][MONGO_COLLECTION]


def write_batch(batch_df, batch_id):
    """Ecrit un micro-batch Spark dans MongoDB."""
    # Convertir le DataFrame Spark en liste de dicts
    rows = [row.asDict() for row in batch_df.collect()]
    if not rows:
        return
    collection = _get_collection()
    collection.insert_many(rows)
    print(f"[MongoDB] batch {batch_id} : {len(rows)} cycles ecrits")