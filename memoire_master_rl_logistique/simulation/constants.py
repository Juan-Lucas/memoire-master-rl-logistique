"""Constantes physiques partagées par les modèles de trajet et de carburant.

Regroupe les coefficients utilisés à la fois par ``graph_model`` (temps de
trajet) et ``fuel_model`` (consommation de carburant) afin d'éviter
toute divergence entre les deux modèles.
"""

from __future__ import annotations

# Pénalité par point de pourcentage de pente : une pente de x % réduit la
# vitesse moyenne d'environ x * SLOPE_PENALTY_PER_PCT. (Jusqu'au 22 octobre 2026, elle
# majorait aussi la consommation dans la même proportion ; le carburant suit maintenant
# les mesures de Kubler, voir plus bas.)
SLOPE_PENALTY_PER_PCT = 0.03

# ---------------------------------------------------------------------------
# Durées de service et pannes (lues par MineEnv, MTWT et le programme linéaire)
# ---------------------------------------------------------------------------
# Chargement à la pelle et déversement à la verse : lois log-normales de moyenne et
# d'écart-type donnés (MineEnv._sample_duration).
LOAD_TIME_MEAN_MIN = 2.0
LOAD_TIME_STD_MIN = 0.3
UNLOAD_TIME_MEAN_MIN = 1.0
UNLOAD_TIME_STD_MIN = 0.2
# Durée d'une réparation après une panne de camion : uniforme sur cet intervalle.
REPAIR_RANGE_MIN = (10.0, 30.0)
MEAN_REPAIR_MIN = sum(REPAIR_RANGE_MIN) / 2

# ---------------------------------------------------------------------------
# Carburant (T14, M20 ; modèle revu le 22 octobre 2026, avant la campagne)
# ---------------------------------------------------------------------------
# Camion de référence : Cat 785D (charge cible 133 t dans le Handbook, charge nominale
# 145 t dans la fiche technique ; le simulateur en transporte 140). Sources, rangées dans
# references/7-donnees-equipements/ :
# - Caterpillar Performance Handbook, édition 49 (SEBD0351-49, 2019), section 25,
#   « Hourly Fuel Consumption Tables » (moyennes de la télématique VisionLink), 785D :
#   54,2–81,4 L/h (usage léger), 81,4–108,5 L/h (moyen), 108,5–135,6 L/h (intensif).
# - Kubler (2015), mesures sur un Cat 785D en exploitation (Queensland) : ralenti
#   11,3 L/h (écart-type 1,71 ; § 5.3) ; consommation en rampe en fonction de la pente x
#   (fraction, 0,01 à 0,18 à vide et à 0,23 en charge), en L/km (§ 5.1.3) :
#   à vide  f(x) = −1661 x³ + 727,7 x² + 6,809 x + 4,562
#   chargé  f(x) = −803,8 x³ + 603,87 x² + 80,825 x + 7,638
#
# Modèle : ralenti mesuré par Kubler ; roulage = courbes de Kubler multipliées par un
# facteur d'échelle FUEL_TRAVEL_SCALE, choisi pour que l'Affectation fixe en nominal
# consomme 95 L/h par camion (milieu de la plage « usage moyen » du Handbook) sur les
# graines de SÉLECTION (fuel_scale.py, 22/10/2026 : ralenti 1,96 L/h, roulage brut
# 208,33 L/h). Sans ce facteur, le simulateur consommerait 211 L/h, au-delà du maximum
# publié (135,6 L/h) : toutes ses pistes montent, y compris au retour (pente de 1 à 8 %),
# et il ne modélise ni descente ni roue libre. Le facteur garde les rapports mesurés
# (chargé / à vide, effet de la pente) et ramène le niveau à la moyenne publiée.
#
# Avant le 22/10/2026 : 0,65 / 0,45 L/km × (1 + 0,03 × pente) et 10 L/h au ralenti, tout
# multiplié par 6,49 pour viser 95 L/h. Le ralenti valait donc 64,9 L/h, près de 6 fois
# la mesure de Kubler, et l'effet de la pente était bien plus faible que mesuré.
FUEL_REFERENCE_L_PER_H = 95.0
FUEL_IDLE_L_PER_H = 11.3  # file, chargement, déversement, panne, ATTENDRE
FUEL_KUBLER_LOADED = (7.638, 80.825, 603.87, -803.8)  # coefficients de x⁰ à x³, L/km
FUEL_KUBLER_EMPTY = (4.562, 6.809, 727.7, -1661.0)
FUEL_KUBLER_MAX_GRADE = 0.18  # domaine mesuré (à vide) ; pentes du simulateur ≤ 0,08
FUEL_TRAVEL_SCALE = 0.4466
# Surconsommation sur piste dégradée : +50 % de (1 − état de la piste). Sans unité,
# non calibrée (aucune source trouvée) : à signaler comme hypothèse.
FUEL_ROAD_STATE_PENALTY = 0.5

# Normalisation du carburant du camion qui décide, dans l'observation (T18) : la
# consommation maximale publiée pour le Cat 785D (usage intensif, 135,6 L/h) sur un
# poste de 8 h, soit ≈ 1 085 L.
FUEL_MAX_L_PER_H = 135.6
FUEL_OBS_MAX_L = FUEL_MAX_L_PER_H * 8.0
