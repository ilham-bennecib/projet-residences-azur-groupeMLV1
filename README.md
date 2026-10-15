# Résidences Azur : anticiper les annulations (groupe 1, MLV)

Le jeu de données doit se trouver dans `../data/reservations_azur.csv` (dossier `data/` à côté de ce dossier).

```bash
pip install -r requirements.txt
python train.py            # entraîne, évalue et sauvegarde model.pkl + model_meta.json (≈ 30 s)
python app.py              # interface Gradio : http://127.0.0.1:7860
uvicorn api:api --reload   # API FastAPI : http://127.0.0.1:8000/docs
pytest                     # 14 tests (application et API)
```

| Fichier | Contenu |
|---|---|
| `notebook.ipynb` | démarche complète, exécutée : audit, EDA, variables, échelle de méthodes, évaluation, seuil, diagnostic, interprétation |
| `audit_variables.md` | une ligne par variable : disponible à la réservation ? utile ? décision |
| `features.py` | variables dérivées, champs saisis, valeurs par défaut, validation (importé par le pipeline, l'application et l'API) |
| `train.py` | pipeline final (préparation + gradient boosting), évaluation chronologique, sauvegarde |
| `model.pkl`, `model_meta.json` | pipeline entier sauvegardé et ses métadonnées |
| `app.py`, `api.py`, `test_app.py` | application Gradio, API FastAPI, tests pytest |
| `note_methodologique.md`, `soutenance.pdf` | note de 3 pages et support de 7 slides |

`model.pkl` a besoin de `features.py` dans le même dossier. En cas d'avertissement de version de scikit-learn au chargement, relancer `python train.py`.
