import os
import json
import joblib
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

DATA_DIR = os.getenv("DATA_DIR", "ml/data")
MODEL_DIR = os.getenv("MODEL_DIR", "ml/model")
N_ITER = int(os.getenv("N_ITER", "40"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))

# MLflow : dossier de stockage local des experiences
mlflow.set_tracking_uri(os.getenv("MLFLOW_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("cmapss-rul")


def load_datasets():
    train = pd.read_parquet(os.path.join(DATA_DIR, "cmapss_train.parquet"))
    test = pd.read_parquet(os.path.join(DATA_DIR, "cmapss_test.parquet"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        features = json.load(f)
    return train, test, features


def evaluate(model, X_test, y_test):
    pred = np.clip(model.predict(X_test), 0, None)
    return {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }


def tune_and_log(estimator, param_space, X_train, y_train, X_test, y_test, label):
    """Regle le modele, l'evalue, et logue le tout dans un run MLflow."""
    print(f"\nReglage {label} : {N_ITER} combinaisons x {CV_FOLDS} plis...")
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_space,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=CV_FOLDS,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    metrics = evaluate(best, X_test, y_test)

    # ----- Logging MLflow (flavor adaptee au type de modele) -----
    with mlflow.start_run(run_name=label):
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_rmse", -search.best_score_)
        mlflow.log_metric("test_rmse", metrics["rmse"])
        mlflow.log_metric("test_mae", metrics["mae"])
        mlflow.log_metric("test_r2", metrics["r2"])
        # XGBoost a sa propre flavor ; sinon flavor sklearn
        if isinstance(best, XGBRegressor):
            mlflow.xgboost.log_model(best, name="model")
        else:
            mlflow.sklearn.log_model(best, name="model")

    print(f"[{label}] R²={metrics['r2']:.3f} MAE={metrics['mae']:.2f} "
          f"RMSE={metrics['rmse']:.2f}")
    return best, search.best_params_, metrics

def main():
    train, test, features = load_datasets()
    X_train, y_train = train[features], train["RUL"]
    X_test, y_test = test[features], test["RUL"]
    print(f"Train : {X_train.shape} | Test : {X_test.shape} | {len(features)} features")

    rf_space = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 1.0],
    }
    rf_best, rf_params, rf_metrics = tune_and_log(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_space, X_train, y_train, X_test, y_test, "RandomForest",
    )

    xgb_space = {
        "n_estimators": [200, 300, 400, 500, 700],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.3],
    }
    xgb_best, xgb_params, xgb_metrics = tune_and_log(
        XGBRegressor(random_state=42, n_jobs=-1),
        xgb_space, X_train, y_train, X_test, y_test, "XGBoost",
    )

    #Selection du meilleur (RMSE test le plus bas)
    if rf_metrics["rmse"] <= xgb_metrics["rmse"]:
        best_model, best_name, best_metrics = rf_best, "RandomForest", rf_metrics
    else:
        best_model, best_name, best_metrics = xgb_best, "XGBoost", xgb_metrics
    print(f"\n>>> Modele retenu : {best_name} (RMSE={best_metrics['rmse']:.2f})")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "rul_model.pkl"))
    with open(os.path.join(MODEL_DIR, "rul_metrics.json"), "w") as f:
        json.dump({
            "selected": {"name": best_name, **best_metrics},
            "rf": rf_metrics, "xgb": xgb_metrics,
            "best_rf_params": rf_params, "best_xgb_params": xgb_params,
        }, f, indent=2)
    print(f"Modele RUL sauvegarde : {MODEL_DIR}/rul_model.pkl")


if __name__ == "__main__":
    main()