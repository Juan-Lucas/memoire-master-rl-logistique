"""Constantes physiques partagées par les modèles de trajet et de carburant.

Regroupe les coefficients utilisés à la fois par ``graph_model`` (temps de
trajet) et ``fuel_model`` (consommation de carburant) afin d'éviter
toute divergence entre les deux modèles.
"""

from __future__ import annotations

# Pénalité par point de pourcentage de pente : une pente de x % réduit la
# vitesse moyenne d'environ x * SLOPE_PENALTY_PER_PCT et augmente la
# consommation de carburant dans la même proportion.
SLOPE_PENALTY_PER_PCT = 0.03

# ---------------------------------------------------------------------------
# Carburant (T14, M20)
# ---------------------------------------------------------------------------
# Référence publiée : Cat 785D (charge utile ≈ 136 t, le plus proche des 140 t du
# simulateur), Caterpillar Performance Handbook (édition 2022, tables de consommation
# issues de la télématique VisionLink) : 54,2–81,4 L/h (usage léger), 81,4–108,5 L/h
# (usage moyen), 108,5–135,6 L/h (usage intensif).
#
# Valeurs d'origine du simulateur, non calibrées : 0,65 L/km chargé, 0,45 L/km à vide,
# 10 L/h au ralenti. Mesure (code de T13, Affectation fixe, nominal, 30 graines) :
# 14,635 L/h par camion en moyenne sur le poste, soit environ 6,5 fois moins que la
# plage « usage moyen ».
#
# Calibration : toutes les consommations sont multipliées par un même facteur, pour
# viser 95 L/h, le milieu de la plage « usage moyen ». Un facteur unique conserve les
# rapports entre trajet chargé, trajet à vide et ralenti : les écarts relatifs entre
# politiques (en %) sont inchangés, seules les valeurs absolues (L/t, L/cycle)
# deviennent réalistes.
FUEL_REFERENCE_L_PER_H = 95.0
FUEL_UNCALIBRATED_L_PER_H = 14.635
FUEL_CALIBRATION_FACTOR = FUEL_REFERENCE_L_PER_H / FUEL_UNCALIBRATED_L_PER_H  # ≈ 6,49

FUEL_L_PER_KM_LOADED = 0.65 * FUEL_CALIBRATION_FACTOR  # ≈ 4,22 L/km
FUEL_L_PER_KM_EMPTY = 0.45 * FUEL_CALIBRATION_FACTOR  # ≈ 2,92 L/km
FUEL_IDLE_L_PER_H = 10.0 * FUEL_CALIBRATION_FACTOR  # ≈ 64,9 L/h (file, chargement, panne)
# Surconsommation sur piste dégradée : +50 % de (1 − état de la piste). Sans unité,
# non calibrée (aucune source trouvée) : à signaler comme hypothèse.
FUEL_ROAD_STATE_PENALTY = 0.5

# Normalisation du carburant dans l'observation de l'agent (``observation_builder``).
# Multipliée par le même facteur que les consommations : l'agent voit exactement les
# mêmes valeurs qu'avant la calibration, donc les modèles déjà entraînés restent valables.
FUEL_OBS_MAX_L = 500.0 * FUEL_CALIBRATION_FACTOR
