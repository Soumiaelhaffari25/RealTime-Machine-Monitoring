import pandas as pd

# 26 colonnes, pas d'en-tête, séparateur = espaces
cols = ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"] + \
       [f"sensor_{i}" for i in range(1, 22)]

df = pd.read_csv(
    "ml/data/train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=cols,
)

print("Shape :", df.shape)                        # attendu ~ (20631, 26)
print("Nb moteurs :", df["unit_id"].nunique())    # attendu 100
print("Cycles par moteur (min/max) :",
      df.groupby("unit_id")["cycle"].max().min(),
      df.groupby("unit_id")["cycle"].max().max())
print(df.head())

# Repérer les capteurs constants (à retirer plus tard)
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
constants = [c for c in sensor_cols if df[c].nunique() == 1]
print("\nCapteurs constants (à retirer) :", constants)