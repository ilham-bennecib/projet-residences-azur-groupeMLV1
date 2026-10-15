# Projet de groupe : anticiper les annulations de séjour des Résidences Azur

IA / Machine Learning · IPSSI · jeudi 15 octobre 2026

## 1. Contexte et objectif

**Résidences Azur** loue des logements à la nuitée sur deux sites : une **résidence urbaine** et une **résidence littoral**. Environ **un séjour sur trois est annulé**, souvent peu avant l'arrivée, et le logement reste vide.

**Votre objectif :** construire un modèle qui estime, pour chaque réservation, la **probabilité d'annulation**, puis une **application web** qui aide l'équipe commerciale à décider quelles réservations doivent être **confirmées par un appel**.

**Moment de la prédiction :** le modèle est utilisé **à l'instant où la réservation est enregistrée**. Il ne peut donc s'appuyer que sur ce que Résidences Azur sait à ce moment-là.

Ordre de grandeur économique, à titre indicatif : un appel de confirmation coûte **6 €** et évite l'annulation dans **un cas sur trois** ; une annulation évitée vaut en moyenne une nuit, soit environ **100 €**. Le seuil de décision doit pouvoir être **réglé dans l'application**.

## 2. Données

Fichier : `reservations_azur.csv`, dans le dossier `data/` placé à côté de `projet_squelette/`. Depuis le squelette, le chemin est `../data/reservations_azur.csv` (valeur par défaut de `train.py`). Une ligne correspond à une réservation. La variable à prédire est `annule` (1 si le séjour a été annulé, 0 s'il a eu lieu).

| Variable | Description | Disponible au moment de la réservation ? |
|---|---|---|
| `id_reservation` | Identifiant du dossier | *à remplir* |
| `site` | Résidence urbaine ou littoral | *à remplir* |
| `annule` | **Cible** : séjour annulé (1) ou honoré (0) | sans objet |
| `date_reservation` | Date d'enregistrement de la réservation | *à remplir* |
| `date_arrivee` | Date d'arrivée prévue | *à remplir* |
| `nuits_week_end`, `nuits_semaine` | Nombre de nuits de week-end et de semaine | *à remplir* |
| `nb_adultes`, `nb_enfants`, `nb_bebes` | Composition du séjour | *à remplir* |
| `formule` | Formule de repas | *à remplir* |
| `pays_client` | Pays d'origine du client | *à remplir* |
| `origine_commerciale`, `canal_vente` | Origine et canal de la réservation | *à remplir* |
| `client_fidele` | Client déjà venu | *à remplir* |
| `annulations_passees`, `sejours_honores_passes` | Historique du client | *à remplir* |
| `logement_demande` | Type de logement demandé | *à remplir* |
| `logement_attribue` | Type de logement attribué | *à remplir* |
| `nb_modifications` | Nombre de modifications du dossier | *à remplir* |
| `garantie` | Aucune, non remboursable ou remboursable | *à remplir* |
| `id_agence`, `id_entreprise` | Agence ou entreprise ayant réservé (codes anonymisés) | *à remplir* |
| `jours_attente_confirmation` | Jours d'attente avant confirmation | *à remplir* |
| `profil_client` | Individuel, contrat, groupe… | *à remplir* |
| `tarif_nuit` | Tarif moyen par nuit | *à remplir* |
| `nb_places_parking`, `nb_demandes_speciales` | Demandes du client | *à remplir* |
| `code_promo` | Code promotionnel associé au dossier | *à remplir* |
| `indice_affinite` | Indice calculé par le service marketing | *à remplir* |
| `statut_dossier`, `date_statut` | Statut final du dossier et date de ce statut | *à remplir* |

## 3. La démarche demandée, étape par étape

Chaque étape produit une trace dans le notebook et un paragraphe dans la note.

| # | Étape | Ce qu'on attend |
|---|---|---|
| 1 | **Cadrage** | Définir la métrique et la décision métier (qui appelle, avec quel seuil) |
| 2 | **Audit des variables** | Compléter le tableau ci-dessus : pour chaque variable, est-elle disponible **au moment de la réservation** ? Est-elle **utile** ? Justifier chaque exclusion |
| 3 | **Exploration (EDA)** | Taux d'annulation par segment, valeurs manquantes (et ce qu'elles signifient), valeurs aberrantes, lignes identiques, évolution dans le temps. Au moins 3 constats justifiés |
| 4 | **Ingénierie des variables** | Les dates sont brutes : en déduire des variables exploitables. Créer au moins **trois variables dérivées pertinentes**, regrouper les modalités rares, encoder, traiter les valeurs manquantes. Justifier chaque choix |
| 5 | **Sélection des variables** | Comparer les performances avec et sans un groupe de variables, ou utiliser une importance. Conserver les variables **utiles et utilisables** |
| 6 | **Modélisation par paliers** | Voir la section 4 |
| 7 | **Évaluation rigoureuse** | Protocole honnête : gérer les lignes identiques, choisir entre découpage aléatoire et chronologique, validation croisée. Métrique, matrice de confusion, ROC, **courbe précision-rappel (AUPRC)**, **seuil** déduit du coût et du gain |
| 8 | **Diagnostic** | Courbe d'apprentissage ou écart entre entraînement et test : montrer l'absence de sur et de sous-apprentissage |
| 9 | **Interprétation** | Variables les plus influentes, expliquées à un responsable commercial |
| 10 | **Mise en production** | Pipeline entier sauvegardé et application (voir section 5) |

## 4. L'échelle de méthodes

Le but est d'**observer la progression** en changeant de méthode, pas seulement de viser le meilleur score. Présentez un **tableau récapitulatif** (même découpage, même métrique) avec, au minimum :

1. une **baseline** ;
2. une **régression logistique** (avec et sans variables dérivées) ;
3. un **arbre de décision** : profondeur faible, profondeur choisie par validation, profondeur illimitée (qu'observe-t-on ?) ;
4. une **forêt aléatoire** ;
5. un **gradient boosting** : `HistGradientBoostingClassifier` de scikit-learn suffit ; XGBoost ou LightGBM sont possibles, à ajouter alors au `requirements.txt` ;
6. *(bonus)* un **perceptron multicouche** (PyTorch ou scikit-learn) : bat-il les arbres ? pourquoi ?

Pour chaque palier : le score, le temps d'entraînement, et **une phrase qui explique l'écart** avec le palier précédent.

## 5. Application, API et livrables

Le groupe implémente lui-même l'application Gradio et l'API FastAPI, en s'appuyant sur le TD 3 (`17_td_deploiement`), le notebook `16_deploiement_modele` et les slides de déploiement. Les deux chargent le même pipeline sauvegardé.

**Application Gradio** (`app.py`) : saisie d'une réservation, probabilité d'annulation, recommandation « appeler / ne pas appeler », **curseur de seuil**. Les entrées sont validées.

**Champs de saisie** : inutile de demander toutes les variables. Proposez **5 à 10 champs**, les plus influents de votre modèle (par exemple dates, nombre de nuits et de personnes, garantie, canal). Les autres variables prennent une **valeur par défaut** (la plus fréquente, ou la médiane de l'entraînement), fixée dans le code.

**API FastAPI** (`api.py`) : endpoint `POST /predict` qui reçoit une réservation en JSON (schéma Pydantic validé, mêmes champs que l'application) et renvoie la probabilité ; une entrée invalide renvoie le code 422.

**Attention au chargement de `model.pkl`** : si vos variables dérivées sont calculées par une fonction à vous dans le pipeline (`FunctionTransformer`), le fichier sauvegardé contient seulement le **nom** de cette fonction. Placez-la dans un fichier `.py` (par exemple `features.py`) importé par `train.py`, `app.py` et `api.py`, et rendez ce fichier. Vérifiez avant le dépôt : dans un terminal neuf, `python app.py` doit démarrer depuis le dossier rendu.

Une **archive par groupe**, nommée `groupeN_CLASSE.zip` (par exemple `groupe3_MLV.zip`), déposée sur Teams **avant 15h30**. Elle contient les fichiers ci-dessous ; les documents sont rendus en **PDF** ou en **Markdown** (`.md`).

| Livrable | Contenu |
|---|---|
| `notebook.ipynb` | Rendu **exécuté**, sorties visibles, et qui s'exécute de haut en bas sans erreur. La démarche complète : **exploration des données (EDA)**, ingénierie des variables, échelle de méthodes, évaluation |
| `audit_variables` (`.md`, `.csv` ou `.pdf`) | Un tableau, **une ligne par variable** : disponible au moment de la réservation ? utile ? justification de chaque exclusion (le tableau de la section 2, complété) |
| `model.pkl` | Pipeline entier sauvegardé (préparation et modèle dans un seul objet) |
| `features.py` (si besoin) | Vos fonctions de variables dérivées, utilisées dans le pipeline |
| `app.py`, `api.py` | Application Gradio (`python app.py`) et API FastAPI (`uvicorn api:api`), **à implémenter par le groupe** |
| `requirements.txt` | Dépendances **aux versions figées** (et version de Python indiquée) |
| `note_methodologique` (`.md` ou `.pdf`) | **2 à 3 pages**, à partir du modèle `note_methodologique.md` : problème, exploration des données, variables, méthodes, résultats, limites, veille, usage d'assistants de code |
| `soutenance` (`.pdf` ou `.pptx`) | **7 slides au maximum** |
| `test_app.py` | Au moins un test `pytest` pour l'application et un pour l'API |

### Bonus (valorisés, non obligatoires)
**Deep Learning (jusqu'à 2 points)** : un réseau de neurones (PyTorch ou scikit-learn) comparé aux modèles classiques. Aussi valorisés : un réglage d'hyperparamètres par validation croisée, une mise en ligne (Hugging Face Spaces).

## 6. Groupes et rôles

Chaque groupe est composé de 3 ou 4 étudiants de la **même classe**. Quatre rôles : données et variables, modélisation et évaluation, application et tests, documentation et soutenance. Chaque membre tient **un rôle principal** (dans un groupe de 3, un membre en cumule deux), mais **tous doivent pouvoir répondre à une question sur n'importe quelle partie.**

### Organisation de la journée

| Horaire | Où en être |
|---|---|
| 10h15 à 10h45 | Lecture du cahier des charges, répartition des rôles, chargement des données |
| 10h45 à 13h00 | Cadrage, audit, exploration, variables dérivées, échelle de méthodes. **À 13h00 : tableau de l'échelle rempli et un `model.pkl` sauvegardé** |
| 14h00 à 15h30 | Seuil, diagnostic, application, API, tests, note méthodologique, support de soutenance. **Dépôt avant 15h30** |
| 16h00 | Soutenances |

Ne gardez pas l'application pour la fin : une application simple qui fonctionne vaut mieux qu'un excellent modèle sans application.

## 7. Soutenance

15 minutes : **10 min** de présentation et de démonstration en direct, **5 min** de questions. Chaque membre répond à au moins une question. L'ordre de passage est tiré au sort. Soyez prêts à justifier : *pourquoi cette variable, pourquoi ce modèle, pourquoi ce seuil.*

## 8. Évaluation (sur 20)

| Critère | Points |
|---|---|
| **Exploration (EDA)** : constats justifiés (segments, valeurs manquantes, valeurs aberrantes, lignes identiques) | 2 |
| **Variables** : audit de disponibilité, repérage de la fuite d'information, variables dérivées pertinentes, sélection justifiée | 3 |
| **Modélisation** : échelle de méthodes cohérente, écarts expliqués, réglage raisonné | 3 |
| **Évaluation** : protocole honnête (lignes identiques, découpage), métrique, seuil économique, diagnostic de fitting | 3 |
| **Application Gradio** : fonctionne depuis le dossier rendu, pipeline entier, entrées validées, un test | 3 |
| **API FastAPI** : endpoint `/predict` au schéma validé, erreurs gérées, un test | 2 |
| **Note méthodologique** : clarté, limites, veille (2 à 3 pages) | 2 |
| **Soutenance et questions** : clarté, démonstration, apport individuel | 2 |

**Bonus Deep Learning** : jusqu'à 2 points. Les bonus valorisent la note et la soutenance. La note individuelle peut être ajustée selon les réponses aux questions.

## 9. Règles

- Le travail doit être **le vôtre** : vous serez interrogés sur vos choix.
- Les assistants de code (ChatGPT, Copilot…) sont **autorisés**. Le groupe **déclare** dans la note méthodologique ce qu'il a généré avec eux, et chaque membre doit pouvoir expliquer tout le code rendu.
- La note de projet est distincte de la note du QCM.
