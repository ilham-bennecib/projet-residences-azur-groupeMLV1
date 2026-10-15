"""Entraîne le pipeline final et le sauvegarde dans model.pkl.

Usage : python train.py
        python train.py --data ../data/reservations_azur.csv

Protocole (choisi dans le notebook) : lignes identiques supprimées, découpage chronologique
sur la date de réservation (80 % les plus anciennes pour l'entraînement, 20 % les plus récentes
pour le test). Le modèle est évalué sur ce test, puis réentraîné sur toutes les données avant
la sauvegarde.
"""
import argparse
import json
import platform
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from features import (add_features, RAW_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                      EXCLUDED, INPUT_FIELDS)

DATA_PATH = "../data/reservations_azur.csv"
TARGET = "annule"
SEUIL = round(6 / (100 / 3), 2)   # appeler si p x 1/3 x 100 > 6 €, soit p > 0,18
SEED = 0


def build_pipeline():
    """Renvoie l'objet complet (préparation + modèle) : c'est lui qui est sauvegardé."""
    preprocess = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                              sparse_output=False), CATEGORICAL_FEATURES),
    ])
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=SEED)
    return make_pipeline(FunctionTransformer(add_features), preprocess, model)


def load(path):
    df = pd.read_csv(path)
    data = df.drop(columns="id_reservation").drop_duplicates()
    return data.sort_values("date_reservation").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()

    data = load(args.data)
    coupe = int(len(data) * 0.8)
    train, test = data.iloc[:coupe], data.iloc[coupe:]

    # 1. Évaluation honnête sur les réservations les plus récentes
    pipe = build_pipeline()
    t0 = time.perf_counter()
    pipe.fit(train[RAW_COLUMNS], train[TARGET])
    duree = time.perf_counter() - t0
    proba = pipe.predict_proba(test[RAW_COLUMNS])[:, 1]
    y = test[TARGET].to_numpy()
    appel = proba >= SEUIL
    metrics = {
        "auprc_test": round(average_precision_score(y, proba), 4),
        "auc_test": round(roc_auc_score(y, proba), 4),
        "precision_seuil": round(precision_score(y, appel), 4),
        "rappel_seuil": round(recall_score(y, appel), 4),
        "gain_net_test_euros": round(float((y[appel] * 100 / 3).sum() - 6 * appel.sum()), 0),
        "temps_entrainement_s": round(duree, 1),
    }
    print("Évaluation sur le test chronologique :", metrics)

    # 2. Modèle de production : réentraîné sur toutes les données
    final = build_pipeline().fit(data[RAW_COLUMNS], data[TARGET])
    joblib.dump(final, "model.pkl")

    meta = {
        "version": "1.0.0",
        "date": str(date.today()),
        "modele": "HistGradientBoostingClassifier (max_iter=400, learning_rate=0.08)",
        "seuil_par_defaut": SEUIL,
        "variables_brutes": RAW_COLUMNS,
        "champs_saisis": INPUT_FIELDS,
        "variables_exclues": EXCLUDED,
        "protocole": "dédoublonnage, découpage chronologique 80/20 sur date_reservation",
        "nb_reservations": int(len(data)),
        "metriques": metrics,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    with open("model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Pipeline sauvegardé dans model.pkl, métadonnées dans model_meta.json")
