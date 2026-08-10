import os
import json
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Configuration
DATA_DIR = os.getenv("DATA_DIR", "ml/data")
MODEL_DIR = os.getenv("MODEL_DIR", "ml/model")
RUL_CAP = int(os.getenv("RUL_CAP", "125"))

# 6 capteurs constants identifies a la Phase 0 (a retirer)
CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_10",
                    "sensor_16", "sensor_18", "sensor_19"]

COLS = ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"] + \
       [f"sensor_{i}" for i in range(1, 22)]

# Features utilisees par le modele : 3 reglages + 15 capteurs utiles
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" not in CONSTANT_SENSORS]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
FEATURE_COLS = SETTING_COLS + SENSOR_COLS


def load_raw(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def add_rul_train(df):
    """RUL = dernier cycle du moteur - cycle courant, plafonne a RUL_CAP."""
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = max_cycle - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
    return df


def build_test_labels(test_df, rul_file):
    """
    Le protocole officiel C-MAPSS evalue sur le DERNIER cycle de chaque moteur test.
    RUL_FD001.txt donne le vrai RUL a ce dernier cycle.
    On extrait donc la derniere ligne de chaque moteur + son vrai RUL.
    """
    true_rul = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["RUL_true"])
    true_rul["unit_id"] = true_rul.index + 1   # moteurs 1..100 dans l'ordre

    # Derniere ligne observee de chaque moteur test
    last_rows = test_df.sort_values("cycle").groupby("unit_id").tail(1).copy()
    last_rows = last_rows.merge(true_rul, on="unit_id")
    # RUL cible plafonne de la meme facon que le train
    last_rows["RUL"] = last_rows["RUL_true"].clip(upper=RUL_CAP)
    return last_rows


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Charger les donnees brutes
    train = load_raw(os.path.join(DATA_DIR, "train_FD001.txt"))
    test = load_raw(os.path.join(DATA_DIR, "test_FD001.txt"))
    print(f"Train brut : {train.shape} | Test brut : {test.shape}")

    # 2. RUL du train (plafonne)
    train = add_rul_train(train)
    print(f"RUL train : min={train['RUL'].min()} max={train['RUL'].max()} "
          f"(plafond={RUL_CAP})")

    # 3. Labels du test (dernier cycle + vrai RUL)
    test_last = build_test_labels(test, os.path.join(DATA_DIR, "RUL_FD001.txt"))
    print(f"Test evalue sur {len(test_last)} moteurs (dernier cycle de chacun)")

    # 4. Scaler min-max ajuste sur le TRAIN uniquement (features seulement)
    scaler = MinMaxScaler()
    scaler.fit(train[FEATURE_COLS])

    # 5. Appliquer le scaler (train complet + dernier cycle du test)
    train_scaled = train.copy()
    train_scaled[FEATURE_COLS] = scaler.transform(train[FEATURE_COLS])

    test_scaled = test_last.copy()
    test_scaled[FEATURE_COLS] = scaler.transform(test_last[FEATURE_COLS])

    # 6. Sauvegarder les datasets (features + RUL + identifiants)
    keep = ["unit_id", "cycle"] + FEATURE_COLS + ["RUL"]
    train_scaled[keep].to_parquet(os.path.join(DATA_DIR, "cmapss_train.parquet"))
    test_scaled[keep].to_parquet(os.path.join(DATA_DIR, "cmapss_test.parquet"))

    # 7. Sauvegarder le scaler + la liste des features (contrat de pretraitement)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print("\nArtefacts sauvegardes :")
    print(f"  {DATA_DIR}/cmapss_train.parquet  ({len(train_scaled)} lignes)")
    print(f"  {DATA_DIR}/cmapss_test.parquet   ({len(test_scaled)} lignes)")
    print(f"  {MODEL_DIR}/scaler.pkl")
    print(f"  {MODEL_DIR}/feature_cols.json    ({len(FEATURE_COLS)} features)")


if __name__ == "__main__":
    main()