"""API FastAPI de prédiction d'annulation.

Lancement : uvicorn api:api --reload   (documentation interactive sur http://127.0.0.1:8000/docs)
Charge le pipeline sauvegardé model.pkl, qui a besoin de features.py (même dossier).
"""
import json
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

import features  # nécessaire : model.pkl appelle features.add_features
from features import build_row

DOSSIER = Path(__file__).resolve().parent
model = joblib.load(DOSSIER / "model.pkl")            # chargé une seule fois, au démarrage
meta = json.loads((DOSSIER / "model_meta.json").read_text(encoding="utf-8"))

api = FastAPI(title="Résidences Azur : risque d'annulation", version=meta["version"])


class Reservation(BaseModel):
    """Schéma d'entrée : mêmes champs que l'application. Toute valeur hors bornes renvoie 422."""
    date_reservation: date
    date_arrivee: date
    nuits_week_end: int = Field(ge=0, le=20)
    nuits_semaine: int = Field(ge=0, le=60)
    nb_adultes: int = Field(ge=1, le=10)
    pays_client: str = Field(min_length=2, max_length=3, pattern="^[A-Za-z]+$")
    origine_commerciale: Literal["Agence en ligne", "Agence classique", "Groupes", "Direct",
                                 "Entreprise", "Offert", "Compagnie aérienne"]
    garantie: Literal["Aucune", "Non remboursable", "Remboursable"]
    nb_demandes_speciales: int = Field(ge=0, le=5)
    id_agence: Optional[int] = Field(None, ge=1, le=9999,
                                     description="Code agence ; absent = agence la plus fréquente pour l'origine")

    @model_validator(mode="after")
    def coherence(self):
        if self.date_arrivee < self.date_reservation:
            raise ValueError("date_arrivee doit être postérieure ou égale à date_reservation")
        if (self.date_arrivee - self.date_reservation).days > 800:
            raise ValueError("l'arrivée est à plus de 800 jours de la réservation")
        if self.nuits_week_end + self.nuits_semaine == 0:
            raise ValueError("le séjour doit compter au moins une nuit")
        return self

    model_config = {"json_schema_extra": {"examples": [{
        "date_reservation": "2026-10-15", "date_arrivee": "2027-05-02", "nuits_week_end": 2,
        "nuits_semaine": 5, "nb_adultes": 2, "pays_client": "FRA",
        "origine_commerciale": "Agence en ligne", "garantie": "Aucune",
        "nb_demandes_speciales": 0, "id_agence": 1270}]}}


@api.get("/health")
def health():
    """Le service est-il en vie, et avec quel modèle ?"""
    return {"status": "ok", "model_version": meta["version"]}


@api.post("/predict")
def predict(r: Reservation):
    """Probabilité d'annulation et recommandation au seuil par défaut."""
    proba = float(model.predict_proba(build_row(r.model_dump()))[0, 1])
    seuil = meta["seuil_par_defaut"]
    return {"proba_annulation": round(proba, 4),
            "recommandation": "appeler" if proba >= seuil else "ne pas appeler",
            "seuil": seuil, "model_version": meta["version"]}
