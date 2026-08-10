import os
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")

# ----- Cache module-level : charge les artefacts une seule fois par worker -----
_scaler = None
_rul_model = None
_detection_model = None
_feature_cols = None


def _load_artifacts():
    """Charge scaler + modeles + liste de features (une fois, mis en cache)."""
    global _scaler, _rul_model, _detection_model, _feature_cols
    if _scaler is None:
        _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        _rul_model = joblib.load(os.path.join(MODEL_DIR, "rul_model.pkl"))
        _detection_model = joblib.load(os.path.join(MODEL_DIR, "detection_model.pkl"))
        with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
            _feature_cols = json.load(f)
    return _scaler, _rul_model, _detection_model, _feature_cols


def score_batch(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Score un lot de cycles.
    Entree : DataFrame pandas avec les colonnes brutes (unit_id, cycle, settings, sensors).
    Sortie : meme DataFrame + colonnes rul_pred et is_anomaly.
    """
    scaler, rul_model, detection_model, feature_cols = _load_artifacts()

    if pdf.empty:
        pdf["rul_pred"] = []
        pdf["is_anomaly"] = []
        return pdf

    # 1. Normaliser les features EXACTEMENT comme a l'entrainement
    X = scaler.transform(pdf[feature_cols])

    # 2. Prediction du RUL (RandomForest), bornee a >= 0
    rul_pred = np.clip(rul_model.predict(X), 0, None)

    # 3. Detection d'anomalie (Isolation Forest) : -1 anomalie, 1 normal
    anomaly = detection_model.predict(X)
    is_anomaly = (anomaly == -1)

    # 4. Ajouter les resultats au DataFrame
    result = pdf.copy()
    result["rul_pred"] = rul_pred.astype(float)
    result["is_anomaly"] = is_anomaly.astype(bool)
    return result