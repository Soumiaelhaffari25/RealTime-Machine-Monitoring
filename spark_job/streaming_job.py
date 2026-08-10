import os
import io
import json
import warnings
import fastavro

# Faire taire les warnings sklearn sur les noms de features (inoffensifs)
warnings.filterwarnings("ignore", category=UserWarning)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import (
    StructType, StructField, IntegerType, LongType, FloatType, BooleanType,
)

from scoring import score_batch
from mongo_sink import write_batch

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
TOPIC = os.getenv("TOPIC", "sensor-readings")
SCHEMA_FILE = os.getenv("SCHEMA_FILE", "/app/schemas/reading.avsc")
CHECKPOINT = os.getenv("CHECKPOINT", "/app/checkpoints/scoring")

with open(SCHEMA_FILE) as f:
    AVRO_SCHEMA = json.load(f)

PARSED_SCHEMA = StructType([
    StructField("unit_id", IntegerType()),
    StructField("cycle", IntegerType()),
    StructField("event_time", LongType()),
    StructField("setting_1", FloatType()),
    StructField("setting_2", FloatType()),
    StructField("setting_3", FloatType()),
] + [StructField(f"sensor_{i}", FloatType()) for i in range(1, 22)])

SCORED_SCHEMA = StructType(
    PARSED_SCHEMA.fields + [
        StructField("rul_pred", FloatType()),
        StructField("is_anomaly", BooleanType()),
    ]
)


def deserialize_avro(raw_bytes):
    if raw_bytes is None or len(raw_bytes) <= 5:
        return None
    try:
        return fastavro.schemaless_reader(io.BytesIO(raw_bytes[5:]), AVRO_SCHEMA)
    except Exception:
        return None


def score_partition(iterator):
    for pdf in iterator:
        yield score_batch(pdf)


def main():
    spark = (
        SparkSession.builder
        .appName("cmapss-streaming")
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
        .config("spark.sql.shuffle.partitions", "3")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")   # moins de bruit
    print(">>> Spark demarre : scoring + ecriture MongoDB...")

    deserialize_udf = udf(deserialize_avro, PARSED_SCHEMA)

    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        raw
        .select(deserialize_udf(col("value")).alias("data"))
        .filter(col("data").isNotNull())
        .select("data.*")
    )

    scored = parsed.mapInPandas(score_partition, schema=SCORED_SCHEMA)

    # Ecriture MongoDB via foreachBatch + checkpoints (exactly-once)
    query = (
        scored.writeStream
        .foreachBatch(write_batch)
        .option("checkpointLocation", CHECKPOINT)
        .outputMode("append")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()