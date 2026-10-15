"""Tests du service (lancement : pytest)."""
import pytest
from fastapi.testclient import TestClient

from api import api
from app import analyze

client = TestClient(api)

VALIDE = {
    "date_reservation": "2026-10-15", "date_arrivee": "2027-05-02", "nuits_week_end": 2,
    "nuits_semaine": 5, "nb_adultes": 2, "pays_client": "FRA",
    "origine_commerciale": "Agence en ligne", "garantie": "Aucune",
    "nb_demandes_speciales": 0, "id_agence": 1270,
}


# ---------- Application Gradio ----------
def test_app_renvoie_une_probabilite():
    proba, decision = analyze(*VALIDE.values(), 0.18)
    assert 0 <= proba <= 1
    assert decision.startswith(("APPELER", "Ne pas appeler"))


def test_app_seuil_change_la_decision():
    _, d_bas = analyze(*VALIDE.values(), 0.0)
    _, d_haut = analyze(*VALIDE.values(), 1.0)
    assert d_bas.startswith("APPELER") and d_haut.startswith("Ne pas appeler")


def test_app_refuse_une_arrivee_avant_la_reservation():
    invalide = {**VALIDE, "date_arrivee": "2026-01-01"}
    with pytest.raises(ValueError, match="date d'arrivée"):
        analyze(*invalide.values(), 0.18)


def test_app_agence_vide_acceptee():
    proba, _ = analyze(*{**VALIDE, "id_agence": ""}.values(), 0.18)
    assert 0 <= proba <= 1


def test_app_refuse_zero_adulte():
    with pytest.raises(ValueError, match="nb_adultes"):
        analyze(*{**VALIDE, "nb_adultes": 0}.values(), 0.18)


# ---------- API FastAPI ----------
def test_api_health():
    assert client.get("/health").json()["status"] == "ok"


def test_api_reservation_valide():
    r = client.post("/predict", json=VALIDE)
    assert r.status_code == 200
    assert 0 <= r.json()["proba_annulation"] <= 1


@pytest.mark.parametrize("modif", [
    {"nb_adultes": 0},                         # hors bornes
    {"garantie": "Gratuite"},                  # catégorie inconnue
    {"date_arrivee": "2026-01-01"},            # arrivée avant la réservation
    {"nuits_semaine": "beaucoup"},             # mauvais type
    {"id_agence": 0},                          # code agence hors bornes
])
def test_api_reservation_invalide(modif):
    assert client.post("/predict", json={**VALIDE, **modif}).status_code == 422


def test_api_agence_optionnelle():
    sans_agence = {k: v for k, v in VALIDE.items() if k != "id_agence"}
    assert client.post("/predict", json=sans_agence).status_code == 200


def test_api_champ_manquant():
    incomplet = {k: v for k, v in VALIDE.items() if k != "garantie"}
    assert client.post("/predict", json=incomplet).status_code == 422
