import os
import json
import time
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

DATA_DIR = os.getenv("DATA_DIR", "ml/data")
MODEL_DIR = os.getenv("MODEL_DIR", "ml/model")
HEALTHY_RUL = int(os.getenv("HEALTHY_RUL", "100"))
CONTAMINATION = float(os.getenv("CONTAMINATION", "0.1"))

mlflow.set_tracking_uri(os.getenv("MLFLOW_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("cmapss-detection")


def evaluate_and_log(train, features, model, name):
    healthy = train[train["RUL"] >= HEALTHY_RUL]
    t0 = time.time()
    model.fit(healthy[features])
    fit_time = time.time() - t0

    preds = model.predict(train[features])
    df = train.copy()
    df["is_anomaly"] = (preds == -1)

    pct_eol = 100 * df[df["RUL"] < 30]["is_anomaly"].mean()
    pct_early = 100 * df[df["RUL"] >= 100]["is_anomaly"].mean()
    separation = pct_eol - pct_early
    metrics = {
        "pct_end_of_life": pct_eol,
        "pct_early_life": pct_early,
        "separation": separation,
        "mean_rul_anomalies": df[df["is_anomaly"]]["RUL"].mean(),
        "mean_rul_normal": df[~df["is_anomaly"]]["RUL"].mean(),
        "fit_time_s": fit_time,
    }

    with mlflow.start_run(run_name=name):
        mlflow.log_param("healthy_rul_threshold", HEALTHY_RUL)
        mlflow.log_param("contamination", CONTAMINATION)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, name="model")

    print(f"[{name}] separation={separation:.1f} pts "
          f"(fin={pct_eol:.1f}% debut={pct_early:.1f}%) fit={fit_time:.2f}s")
    return metrics, model


def main():
    train = pd.read_parquet(os.path.join(DATA_DIR, "cmapss_train.parquet"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        features = json.load(f)
    print(f"Cycles sains (RUL >= {HEALTHY_RUL}) : "
          f"{(train['RUL'] >= HEALTHY_RUL).sum()} / {len(train)}")

    candidates = [
        ("IsolationForest",
         IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                         random_state=42, n_jobs=-1)),
        ("LocalOutlierFactor",
         LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION,
                            novelty=True)),
    ]

    results, models = {}, {}
    for name, mdl in candidates:
        m, fitted = evaluate_and_log(train, features, mdl, name)
        results[name], models[name] = m, fitted

    best_name = max(results, key=lambda n: results[n]["separation"])
    print(f"\n>>> Meilleur score de separation : {best_name} "
          f"({results[best_name]['separation']:.1f} pts)")

    joblib.dump(models[best_name], os.path.join(MODEL_DIR, "detection_model.pkl"))
    with open(os.path.join(MODEL_DIR, "detection_metrics.json"), "w") as f:
        json.dump({"selected": best_name, "comparison": results}, f, indent=2)
    print(f"Modele de detection sauvegarde : {MODEL_DIR}/detection_model.pkl")


if __name__ == "__main__":
    main()