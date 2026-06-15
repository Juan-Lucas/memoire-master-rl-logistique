"""Constantes physiques partagées par les modèles de trajet et de carburant.

Regroupe les coefficients utilisés à la fois par ``graph_model`` (temps de
trajet, Eq. 3.2) et ``fuel_model`` (consommation de carburant) afin d'éviter
toute divergence entre les deux modèles.
"""

from __future__ import annotations

# Pénalité par point de pourcentage de pente : une pente de x % réduit la
# vitesse moyenne d'environ x * SLOPE_PENALTY_PER_PCT et augmente la
# consommation de carburant dans la même proportion.
SLOPE_PENALTY_PER_PCT = 0.03
