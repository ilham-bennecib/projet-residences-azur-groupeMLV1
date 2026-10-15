"""Variables dérivées et valeurs par défaut, partagées par train.py, app.py, api.py et le notebook.

Le pipeline sauvegardé (model.pkl) appelle `add_features` via un FunctionTransformer :
ce fichier doit donc être présent à côté de model.pkl pour que le chargement fonctionne.
"""
import numpy as np
import pandas as pd

# Variables brutes conservées après l'audit (disponibles au moment de la réservation et utiles)
RAW_COLUMNS = [
    "site", "date_reservation", "date_arrivee", "nuits_week_end", "nuits_semaine",
    "nb_adultes", "nb_enfants", "nb_bebes", "formule", "pays_client",
    "origine_commerciale", "canal_vente", "client_fidele", "annulations_passees",
    "sejours_honores_passes", "logement_demande", "garantie", "id_agence", "id_entreprise",
    "profil_client", "tarif_nuit", "nb_demandes_speciales",
]

# Colonnes exclues après l'audit (voir audit_variables.md)
EXCLUDED = {
    "id_reservation": "identifiant, aucune information",
    "statut_dossier": "fuite : statut final connu après le séjour",
    "date_statut": "fuite : date du statut final",
    "logement_attribue": "fuite : attribué à l'arrivée",
    "nb_modifications": "inconnu à l'enregistrement (modifications ultérieures)",
    "jours_attente_confirmation": "s'accumule après l'enregistrement",
    "nb_places_parking": "fuite : demandé à l'arrivée (0 % d'annulation si > 0)",
    "code_promo": "aucun lien avec la cible (32 à 35 % d'annulation pour chaque code)",
    "indice_affinite": "aucun lien avec la cible (corrélation -0,007), origine inconnue",
}

# Colonnes produites par add_features
NUMERIC_FEATURES = [
    "delai_jours", "nb_nuits", "part_week_end", "nb_personnes", "avec_enfants",
    "nuits_week_end", "nuits_semaine", "nb_adultes",
    "client_fidele", "annulations_passees", "sejours_honores_passes", "a_deja_annule",
    "tarif_nuit", "tarif_par_personne", "nb_demandes_speciales",
    "via_agence", "via_entreprise", "mois_sin", "mois_cos", "jour_arrivee_sin", "jour_arrivee_cos",
]
CATEGORICAL_FEATURES = [
    "site", "formule", "pays_client", "origine_commerciale", "canal_vente",
    "logement_demande", "garantie", "profil_client", "agence", "saison",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme les colonnes brutes en variables exploitables par le modèle."""
    X = df.copy()
    resa = pd.to_datetime(X["date_reservation"], errors="coerce")
    arrivee = pd.to_datetime(X["date_arrivee"], errors="coerce")

    # 1. Délai entre réservation et arrivée : la variable dérivée la plus informative
    X["delai_jours"] = (arrivee - resa).dt.days.clip(lower=0)

    # 2. Durée et composition du séjour
    X["nb_nuits"] = X["nuits_week_end"] + X["nuits_semaine"]
    X["part_week_end"] = X["nuits_week_end"] / X["nb_nuits"].replace(0, np.nan)
    X["part_week_end"] = X["part_week_end"].fillna(0)
    X["nb_enfants"] = X["nb_enfants"].fillna(0)
    X["nb_personnes"] = X["nb_adultes"] + X["nb_enfants"] + X["nb_bebes"]
    X["avec_enfants"] = ((X["nb_enfants"] + X["nb_bebes"]) > 0).astype(int)

    # 3. Historique du client
    X["a_deja_annule"] = (X["annulations_passees"] > 0).astype(int)

    # 4. Tarif nettoyé (valeurs négatives ou absurdes ramenées dans [0 ; 1000])
    X["tarif_nuit"] = X["tarif_nuit"].clip(0, 1000)
    X["tarif_par_personne"] = X["tarif_nuit"] / X["nb_personnes"].replace(0, np.nan)
    X["tarif_par_personne"] = X["tarif_par_personne"].fillna(X["tarif_nuit"])

    # 5. Réservation passée par une agence ou une entreprise ; agence comme catégorie
    X["via_agence"] = X["id_agence"].notna().astype(int)
    X["via_entreprise"] = X["id_entreprise"].notna().astype(int)
    X["agence"] = X["id_agence"].fillna(0).astype(int).astype(str)

    # 6. Calendrier de l'arrivée, encodé de façon cyclique (décembre est voisin de janvier)
    mois = arrivee.dt.month.fillna(1)
    jour = arrivee.dt.dayofweek.fillna(0)
    X["mois_sin"] = np.sin(2 * np.pi * mois / 12)
    X["mois_cos"] = np.cos(2 * np.pi * mois / 12)
    X["jour_arrivee_sin"] = np.sin(2 * np.pi * jour / 7)
    X["jour_arrivee_cos"] = np.cos(2 * np.pi * jour / 7)
    X["saison"] = mois.map({12: "hiver", 1: "hiver", 2: "hiver", 3: "printemps", 4: "printemps",
                            5: "printemps", 6: "ete", 7: "ete", 8: "ete"}).fillna("automne")

    X["pays_client"] = X["pays_client"].fillna("Inconnu")
    return X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


# Champs saisis dans l'application et l'API : 10 champs, parmi les plus influents du modèle
# (importance par permutation, section 10 du notebook) et faciles à connaître au téléphone.
# Le code agence est la 2e variable la plus influente : il est saisi (optionnel). S'il est laissé
# vide, on prend l'agence la plus fréquente pour l'origine commerciale (aucune pour une vente directe).
INPUT_FIELDS = [
    "date_reservation", "date_arrivee", "nuits_week_end", "nuits_semaine", "nb_adultes",
    "pays_client", "origine_commerciale", "garantie", "nb_demandes_speciales", "id_agence",
]

# Listes de valeurs acceptées (celles vues à l'entraînement)
ORIGINES = ["Agence en ligne", "Agence classique", "Groupes", "Direct", "Entreprise",
            "Offert", "Compagnie aérienne"]
GARANTIES = ["Aucune", "Non remboursable", "Remboursable"]

# Valeurs par défaut des autres variables : la plus fréquente (catégories)
# ou la médiane (nombres) des données d'entraînement dédoublonnées, fixées ici.
DEFAULTS = {
    "site": "Urbaine", "nb_enfants": 0, "nb_bebes": 0, "formule": "Petit-déjeuner",
    "client_fidele": 0, "annulations_passees": 0, "sejours_honores_passes": 0, "logement_demande": "L07",
    "id_entreprise": np.nan, "profil_client": "Individuel", "tarif_nuit": 116,
}
# Canal et agence les plus fréquents pour chaque origine commerciale (cohérence de la saisie)
CANAL_PAR_ORIGINE = {"Direct": "Direct", "Offert": "Direct", "Entreprise": "Entreprise",
                     "Compagnie aérienne": "Entreprise"}
AGENCE_PAR_ORIGINE = {"Agence en ligne": 1270.0, "Agence classique": 1159.0, "Groupes": 1861.0}


PAYS = ["FRA", "GBR", "PRT", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD", "USA", "CHE",
        "AUT", "SWE", "CHN"]


def validate_input(values: dict) -> list:
    """Renvoie la liste des erreurs de saisie (liste vide si la réservation est valide)."""
    errors = []
    try:
        resa = pd.Timestamp(str(values["date_reservation"]))
        arrivee = pd.Timestamp(str(values["date_arrivee"]))
        if arrivee < resa:
            errors.append("La date d'arrivée doit être postérieure ou égale à la date de réservation.")
        elif (arrivee - resa).days > 800:
            errors.append("L'arrivée est à plus de 800 jours de la réservation.")
    except (ValueError, TypeError):
        errors.append("Les dates doivent être au format AAAA-MM-JJ (par exemple 2026-11-20).")
    bornes = {"nuits_week_end": (0, 20), "nuits_semaine": (0, 60), "nb_adultes": (1, 10),
              "nb_demandes_speciales": (0, 5)}
    for champ, (lo, hi) in bornes.items():
        try:
            v = float(values.get(champ))
            ok = v == int(v) and lo <= int(v) <= hi
        except (TypeError, ValueError):
            ok = False
        if not ok:
            errors.append(f"{champ} doit être un entier entre {lo} et {hi}.")
    if not errors and int(values["nuits_week_end"]) + int(values["nuits_semaine"]) == 0:
        errors.append("Le séjour doit compter au moins une nuit.")
    if values.get("origine_commerciale") not in ORIGINES:
        errors.append(f"Origine commerciale inconnue : choisir parmi {', '.join(ORIGINES)}.")
    if values.get("garantie") not in GARANTIES:
        errors.append(f"Garantie inconnue : choisir parmi {', '.join(GARANTIES)}.")
    agence = values.get("id_agence")
    if agence not in (None, ""):
        try:
            a = float(agence)
            ok = a == int(a) and 1 <= a <= 9999
        except (TypeError, ValueError):
            ok = False
        if not ok:
            errors.append("Le code agence doit être un entier entre 1 et 9999 (ou vide).")
    pays = str(values.get("pays_client", "")).strip()
    if not (2 <= len(pays) <= 3 and pays.isalpha()):
        errors.append("Le pays doit être un code de 2 ou 3 lettres (par exemple FRA).")
    return errors


def build_row(values: dict) -> pd.DataFrame:
    """Complète une saisie (champs INPUT_FIELDS) avec les valeurs par défaut."""
    row = {**DEFAULTS, **values}
    origine = row["origine_commerciale"]
    row["canal_vente"] = CANAL_PAR_ORIGINE.get(origine, "Intermédiaire")
    agence = values.get("id_agence")
    if agence in (None, ""):
        row["id_agence"] = AGENCE_PAR_ORIGINE.get(origine, np.nan)   # non saisie : valeur la plus fréquente
    else:
        row["id_agence"] = float(agence)
    row["date_reservation"] = str(row["date_reservation"])
    row["date_arrivee"] = str(row["date_arrivee"])
    row["pays_client"] = str(row["pays_client"]).strip().upper()
    return pd.DataFrame([{c: row[c] for c in RAW_COLUMNS}])
