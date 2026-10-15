# Note méthodologique : anticiper les annulations des Résidences Azur

Groupe : 1  ·  Classe : MLV  ·  Membres et rôle principal : Romain LIENARD (données et variables ; application et tests), Florent NGOR (modélisation et évaluation), ilham KADRI (note et soutenance)

## 1. Problème et décision métier

À l'instant où une réservation est enregistrée, le modèle estime sa **probabilité d'annulation**. L'équipe commerciale appelle les réservations à risque ; le modèle ne décide pas seul.

**Seuil économique.** Un appel coûte 6 € et évite l'annulation une fois sur trois ; une annulation évitée vaut 100 €. Appeler rapporte en moyenne p × 1/3 × 100 − 6 €, positif si **p > 0,18**. C'est le seuil par défaut de l'application, réglable par un curseur.

**Métrique principale : l'AUPRC** (aire sous la courbe précision-rappel), qui juge la capacité à placer les annulations en tête quel que soit le seuil, et reste honnête quand les annulations sont minoritaires (24,5 % après dédoublonnage). On suit aussi l'AUC ROC et le **gain net en euros** au seuil choisi.

## 2. Données et variables

**Exploration (constats principaux).** (1) 28 204 lignes identiques (25 %), très annulées (53 % contre 23,5 %) : surtout des réservations de groupe en plusieurs exemplaires. (2) Le délai entre réservation et arrivée est le premier signal : 8 % d'annulation à moins d'une semaine, plus de 50 % au-delà d'un an. (3) Segments contrastés : site urbain 37,5 % contre littoral 24 %, origine « Groupes » 57 % contre « Direct » 13 %, garantie « non remboursable » 99 %. (4) Les manquants de `id_agence` (14 %) et `id_entreprise` (94 %) signifient « pas d'intermédiaire ». (5) Tarifs aberrants (1 917 ≤ 0, un à 6 372 €). (6) Le taux d'annulation varie de 27 % à 39 % selon le trimestre.

**Audit** (tableau complet dans `audit_variables.md`). Exclues pour **fuite** : `statut_dossier` et `date_statut` (la réponse elle-même), `logement_attribue` (attribué à l'arrivée : 4,5 % d'annulation quand il diffère du logement demandé), `nb_places_parking` (0 % d'annulation dès qu'une place est demandée), `nb_modifications` et `jours_attente_confirmation` (s'accumulent après l'enregistrement). Exclues car **inutiles** : `id_reservation`, `code_promo` (tous les codes entre 31,5 % et 35 %), `indice_affinite` (corrélation −0,007). **Gardées avec réserve** : `pays_client` et `nb_demandes_speciales`, très influentes, à faire confirmer par le métier.

**Variables dérivées** (`features.py`, appelées dans le pipeline) : délai en jours, nombre de nuits et part de week-end, nombre de personnes et présence d'enfants, tarif par personne, « a déjà annulé », passage par une agence ou une entreprise, mois et jour d'arrivée encodés sur un cercle (sinus, cosinus) et saison. Modalités rares (moins de 50 réservations) regroupées par `OneHotEncoder(min_frequency=50)`, qui gère aussi une catégorie inconnue.

**Sélection.** Sans les variables dérivées, l'AUPRC du boosting tombe de 0,600 à 0,512 ; sans le délai seul, à 0,577 ; sans pays ni demandes spéciales, à 0,484.

## 3. Méthodes

Même découpage, même métrique (test chronologique, 16 797 réservations).

| Palier | AUPRC test | AUC | Temps | Écart expliqué |
|---|---|---|---|---|
| Baseline (taux moyen) | 0,246 | 0,500 | 0 s | Aucun classement |
| Logistique sans dérivées | 0,495 | 0,756 | 2 s | Les variables brutes portent déjà l'essentiel |
| Logistique avec dérivées | 0,531 | 0,792 | 5 s | +0,036 grâce au délai, à la saison, aux intermédiaires |
| Arbre, profondeur 3 | 0,404 | 0,718 | 1 s | Trop simple : sous-apprentissage |
| Arbre, profondeur 10 (validation) | 0,540 | 0,800 | 2 s | Capte des interactions, dépasse la logistique |
| Arbre illimité | 0,342 | 0,636 | 2 s | 1,000 à l'entraînement : apprend par cœur |
| Forêt aléatoire (300 arbres) | 0,588 | 0,830 | 16 s | Moyenne d'arbres : moins de variance |
| **Gradient boosting (HistGB)** | **0,600** | **0,836** | 11 s | Arbres successifs qui corrigent les erreurs : moins de biais |
| MLP 64×64 (bonus) | 0,591 | 0,819 | 13 s | Ne bat pas les arbres sur des données en tableau |

Le boosting est retenu : meilleur score, plus rapide que la forêt, écart entraînement-test plus faible. Trois réglages comparés par validation croisée chronologique donnent 0,618 à 0,626 (écart-type 0,07) : différence non significative, on garde le plus simple.

## 4. Résultats et évaluation

**Protocole.** Lignes identiques supprimées **avant** tout découpage (sinon une même réservation peut être à la fois en entraînement et en test). **Découpage chronologique** sur la date de réservation : 80 % les plus anciennes pour l'entraînement, 20 % les plus récentes pour le test, comme en production. Réglages (profondeur, seuil) choisis sur les 20 % les plus récents de l'entraînement ; test utilisé une seule fois. Validation croisée chronologique (`TimeSeriesSplit`, 4 plis) : AUPRC 0,622 ± 0,074.

**Deux vérifications du protocole.** (1) Un protocole naïf (découpage aléatoire, doublons conservés) afficherait une AUPRC d'environ 0,90 : sans dédoublonnage ni découpage dans le temps, le score est gonflé. (2) Notre test ne contient que les réservations récentes déjà arrivées, donc surtout des courts délais (délai médian 29 jours contre 48 sur l'ensemble). En découpant plutôt sur la **date d'arrivée** (entraînement 2021-2022, test 2023), le classement des modèles est identique (boosting 0,645, logistique 0,625, hasard 0,285) : nos conclusions ne dépendent pas du découpage.

**Test.** AUPRC 0,600 (hasard 0,246), AUC 0,836. Au seuil 0,18 : 6 069 appels sur 16 797, précision 0,50, rappel 0,73, **gain net 64 519 €**, contre 36 785 € en appelant tout le monde. Le modèle **sous-estime** le risque des réservations récentes (0,14 prédit pour 0,24 observé entre 0,1 et 0,18) : le seuil choisi sur la validation, **0,11**, donne **70 657 €** sur le test. On le recommande ; le curseur permet de l'appliquer.

**Diagnostic.** Courbe d'apprentissage : l'AUPRC de validation plafonne vers 0,67 dès 27 000 réservations, l'écart avec l'entraînement se stabilise vers 0,17. Pas de sous-apprentissage (bien au-dessus du hasard), sur-apprentissage contenu ; plus de données anciennes n'aideraient guère.

**Interprétation** (importance par permutation) : pays du client, agence, demandes spéciales, dates (délai et saison), historique du client, origine commerciale, garantie. Message commercial : *une réservation lointaine, par une agence en ligne, sans demande spéciale, d'un client sans historique, est à appeler en priorité.*

## 5. Application et tests

`python train.py` entraîne, évalue et sauvegarde le pipeline entier (`model.pkl` + `model_meta.json`). `python app.py` lance l'interface Gradio (http://127.0.0.1:7860) : 10 champs (dates, nuits, adultes, pays, origine, garantie, demandes spéciales, code agence, ce dernier optionnel : vide = agence la plus fréquente pour l'origine), les autres variables prenant leur valeur la plus fréquente ; probabilité, recommandation, curseur de seuil (0,18 par défaut) ; saisies impossibles refusées avec un message clair. `uvicorn api:api` lance l'API : `POST /predict` (schéma Pydantic avec types, bornes, listes de valeurs et cohérence des dates ; entrée invalide → 422), `GET /health`. `pytest` lance 14 tests : probabilité entre 0 et 1, effet du seuil, agence non saisie acceptée, refus d'entrées invalides dans l'application ; réservation valide → 200, agence optionnelle, valeurs hors bornes, code agence invalide, catégorie inconnue, dates incohérentes, mauvais type et champ manquant → 422 pour l'API.

## 6. Limites et veille

- **Disponibilité de `pays_client` et `nb_demandes_speciales`** à confirmer : si elles sont complétées après la réservation, le score est surestimé.
- **Dérive** : le taux d'annulation varie selon la période et le modèle sous-estime le risque récent. À surveiller (indice PSI sur les variables, taux d'annulation réel) et ré-entraîner régulièrement ; recalibrer les probabilités (`CalibratedClassifierCV`).
- **Valeurs par défaut** dans l'application : les variables non saisies (historique du client, formule, tarif…) prennent leur valeur la plus fréquente ; pour un client connu, saisir son historique affinerait la prédiction.
- **Équité** : le pays pèse fortement ; vérifier que les erreurs ne pénalisent pas un groupe de clients (RGPD : un humain décide de l'appel).
- **Coût** : entraînement en 11 s, prédiction en quelques millisecondes : compatible avec un usage en temps réel.
- **Avec plus de temps** : calibration, réglage plus large par validation croisée, mise en ligne sur Hugging Face Spaces, suivi des expériences (MLflow).

## 7. Usage d'assistants de code

Le groupe a utilisé **Claude (Anthropic)** pour générer une première version du code (`features.py`, `train.py`, `app.py`, `api.py`, `test_app.py`, notebook) et des documents (audit, note, support). Le groupe a relu, exécuté et vérifié chaque partie : le notebook s'exécute de haut en bas, `pytest` passe (14 tests), l'application et l'API démarrent depuis le dossier rendu. Modifications propres au groupe : vérification ligne à ligne de l'audit des variables (fuites et justifications) à partir des chiffres recalculés dans le notebook, relecture et correction du chemin des données et de la cohérence des métriques entre le notebook, la note et le support de soutenance, ajustement des dépendances de `requirements.txt`. Chaque membre est en mesure d'expliquer et de justifier l'ensemble du code rendu.
