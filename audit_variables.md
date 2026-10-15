# Audit des variables : Résidences Azur

Question posée pour chaque colonne : **est-elle connue à l'instant où la réservation est enregistrée ?** Puis : **est-elle utile ?** Les chiffres viennent de la section 2 et 3 du notebook (112 188 réservations brutes).

| Variable | Disponible à la réservation ? | Utile ? | Décision | Justification |
|---|---|---|---|---|
| `id_reservation` | Oui | Non | **Exclue** | Identifiant unique, ne porte aucune information sur le comportement |
| `site` | Oui | Oui | Gardée | Urbaine 37,5 % d'annulation, littoral 24,3 % |
| `annule` | Sans objet | Cible | Cible | Variable à prédire |
| `date_reservation` | Oui | Oui (via dérivées) | Gardée | Sert à calculer le délai avant l'arrivée ; aussi utilisée pour le découpage chronologique |
| `date_arrivee` | Oui | Oui (via dérivées) | Gardée | Délai, mois et jour d'arrivée (saisonnalité) |
| `nuits_week_end`, `nuits_semaine` | Oui | Oui | Gardées | Durée et type de séjour ; dérivées `nb_nuits`, `part_week_end` |
| `nb_adultes`, `nb_enfants`, `nb_bebes` | Oui | Oui | Gardées | Composition du séjour ; dérivées `nb_personnes`, `avec_enfants` ; 3 enfants manquants mis à 0 |
| `formule` | Oui | Faible | Gardée | Pension complète 55 %, autres formules 31 à 33 % |
| `pays_client` | Oui, **à confirmer** | Oui, fort | Gardée avec réserve | FRA 52 % contre 14 à 32 % pour les autres pays. Si le pays n'était complété qu'à l'arrivée, ce serait une fuite : point à vérifier avec le métier (sans pays ni demandes, AUPRC 0,600 → 0,484). 474 manquants → « Inconnu » ; pays rares regroupés |
| `origine_commerciale` | Oui | Oui | Gardée | Groupes 57 %, Direct 13 % |
| `canal_vente` | Oui | Moyenne | Gardée | Redondant en partie avec l'origine |
| `client_fidele` | Oui | Oui | Gardée | Clients fidèles 12 % contre 34 % |
| `annulations_passees`, `sejours_honores_passes` | Oui | Oui | Gardées | Historique du client ; dérivée `a_deja_annule` |
| `logement_demande` | Oui | Faible | Gardée | Type demandé au moment de la réservation |
| `logement_attribue` | **Non** | — | **Exclue (fuite)** | Attribué à l'arrivée : s'il diffère du logement demandé, 4,5 % d'annulation contre 37,3 %. Le client était présent |
| `nb_modifications` | **Non** | — | **Exclue (fuite)** | Les modifications arrivent après l'enregistrement (0 au moment de la réservation) ; 37 % d'annulation à 0 modification, 12 à 18 % au-delà |
| `garantie` | Oui | Oui | Gardée | Non remboursable 99 % d'annulation (surprenant, à faire confirmer, mais connue à la réservation) |
| `id_agence` | Oui | Oui | Gardée | Manquant = pas d'agence (21 % contre 35 %) → `via_agence` ; agences rares regroupées |
| `id_entreprise` | Oui | Oui | Gardée | Manquant à 94 % = pas d'entreprise → `via_entreprise` (15 % d'annulation avec entreprise) |
| `jours_attente_confirmation` | **Non** | — | **Exclue** | S'accumule après l'enregistrement : inconnu à l'instant de la réservation |
| `profil_client` | Oui | Oui | Gardée | Individuel 36,5 %, groupe 8,6 % |
| `tarif_nuit` | Oui | Oui | Gardée | Nettoyée : 1 917 tarifs ≤ 0 et un tarif de 6 372 € ramenés dans [0 ; 1 000] ; dérivée `tarif_par_personne` |
| `nb_places_parking` | **Non** | — | **Exclue (fuite)** | 0 % d'annulation dès qu'une place est demandée : saisi à l'arrivée |
| `nb_demandes_speciales` | Oui, **à confirmer** | Oui, fort | Gardée avec réserve | 43 % d'annulation sans demande, 5 % avec cinq. Peuvent être ajoutées après la réservation : à confirmer |
| `code_promo` | Oui | **Non** | **Exclue** | 12 codes, tous entre 31,5 % et 35 % d'annulation (moyenne 33 %) : aucun signal |
| `indice_affinite` | Inconnu | **Non** | **Exclue** | Corrélation −0,007 avec la cible ; mode et date de calcul inconnus |
| `statut_dossier` | **Non** | — | **Exclue (fuite)** | C'est la réponse : « Annulé » et « Absent non prévenu » = 100 % d'annulation, « Séjour effectué » = 0 % |
| `date_statut` | **Non** | — | **Exclue (fuite)** | Date du statut final, connue après le séjour ou l'annulation |

**Variables dérivées créées** (dans `features.py`) : `delai_jours`, `nb_nuits`, `part_week_end`, `nb_personnes`, `avec_enfants`, `a_deja_annule`, `tarif_par_personne`, `via_agence`, `via_entreprise`, `agence`, `mois_sin`/`mois_cos`, `jour_arrivee_sin`/`jour_arrivee_cos`, `saison`.
