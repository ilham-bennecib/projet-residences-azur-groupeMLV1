"""Application Gradio : faut-il appeler le client pour confirmer sa réservation ?

Lancement : python app.py  (puis ouvrir http://127.0.0.1:7860)
Charge le pipeline sauvegardé model.pkl, qui a besoin de features.py (même dossier).
"""
import json
from datetime import date, timedelta
from pathlib import Path

import gradio as gr
import joblib

import features  # nécessaire : model.pkl appelle features.add_features
from features import INPUT_FIELDS, ORIGINES, GARANTIES, PAYS, build_row, validate_input

DOSSIER = Path(__file__).resolve().parent
model = joblib.load(DOSSIER / "model.pkl")            # chargé une seule fois, au démarrage
meta = json.loads((DOSSIER / "model_meta.json").read_text(encoding="utf-8"))
SEUIL_DEFAUT = meta["seuil_par_defaut"]


def predict_proba(values: dict) -> float:
    """Probabilité d'annulation d'une réservation valide."""
    return float(model.predict_proba(build_row(values))[0, 1])


def analyze(date_reservation, date_arrivee, nuits_week_end, nuits_semaine, nb_adultes, pays_client,
            origine_commerciale, garantie, nb_demandes_speciales, id_agence, seuil):
    """Valide la saisie, calcule la probabilité et la recommandation. Lève ValueError si invalide."""
    values = dict(zip(INPUT_FIELDS, [date_reservation, date_arrivee, nuits_week_end, nuits_semaine,
                                     nb_adultes, pays_client, origine_commerciale, garantie,
                                     nb_demandes_speciales, id_agence]))
    if values["pays_client"] == "Autre":
        values["pays_client"] = "XXX"            # pays rare : regroupé avec les modalités peu fréquentes
    errors = validate_input(values)
    if errors:
        raise ValueError(" ".join(errors))
    proba = predict_proba(values)
    if proba >= seuil:
        decision = f"APPELER le client (risque {proba:.0%} ≥ seuil {seuil:.0%})"
    else:
        decision = f"Ne pas appeler (risque {proba:.0%} < seuil {seuil:.0%})"
    return round(proba, 3), decision


def interface_fn(*args):
    try:
        return analyze(*args)
    except ValueError as err:
        raise gr.Error(str(err))   # message clair affiché à l'utilisateur, pas de prédiction


aujourd_hui = date.today()
demo = gr.Interface(
    fn=interface_fn,
    inputs=[
        gr.Textbox(value=str(aujourd_hui), label="Date de réservation (AAAA-MM-JJ)"),
        gr.Textbox(value=str(aujourd_hui + timedelta(days=60)), label="Date d'arrivée (AAAA-MM-JJ)"),
        gr.Number(value=1, precision=0, label="Nuits de week-end"),
        gr.Number(value=2, precision=0, label="Nuits de semaine"),
        gr.Number(value=2, precision=0, label="Nombre d'adultes"),
        gr.Dropdown(PAYS + ["Autre"], value="FRA", label="Pays du client"),
        gr.Dropdown(ORIGINES, value="Agence en ligne", label="Origine commerciale"),
        gr.Dropdown(GARANTIES, value="Aucune", label="Garantie"),
        gr.Number(value=0, precision=0, label="Demandes spéciales"),
        gr.Textbox(value="1270", label="Code agence (vide = agence la plus fréquente pour cette origine)"),
        gr.Slider(0, 1, value=SEUIL_DEFAUT, step=0.01,
                  label=f"Seuil d'appel (défaut {SEUIL_DEFAUT} : appel rentable si risque × 33 € > 6 €)"),
    ],
    outputs=[gr.Number(label="Probabilité d'annulation"), gr.Textbox(label="Recommandation")],
    examples=[
        [str(aujourd_hui), str(aujourd_hui + timedelta(days=200)), 2, 5, 2, "FRA", "Agence en ligne", "Aucune", 0, "1270", SEUIL_DEFAUT],
        [str(aujourd_hui), str(aujourd_hui + timedelta(days=5)), 1, 1, 2, "GBR", "Direct", "Aucune", 2, "", SEUIL_DEFAUT],
    ],
    title="Résidences Azur : faut-il confirmer cette réservation par téléphone ?",
    description="Saisissez la réservation. Le modèle estime la probabilité d'annulation ; "
                "au-dessus du seuil, un appel de confirmation est rentable (6 € l'appel, 100 € par annulation évitée, "
                "un appel réussi sur trois). Les autres informations prennent leur valeur la plus fréquente.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
