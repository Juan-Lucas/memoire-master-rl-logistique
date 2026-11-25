"""Inférence entre modèles (M8 du rapport du 17/11/2026) : d apparié, bootstrap hiérarchique."""

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.analysis.model_inference import (
    cohens_dz,
    hierarchical_bootstrap_ci,
)
from memoire_master_rl_logistique.analysis.statistical_tests import cohens_d


def test_d_apparie():
    """d_z = moyenne / écart type des différences ; il dépasse le d groupé quand les deux
    séries varient ensemble (réplications appariées), comme dans la campagne."""
    rng = np.random.default_rng(0)
    common = rng.normal(0, 50, 30)              # aléa partagé par les deux politiques
    a, b = 1000 + common + rng.normal(0, 5, 30), 990 + common + rng.normal(0, 5, 30)
    diff = a - b
    assert cohens_dz(diff) == pytest.approx(diff.mean() / diff.std(ddof=1))
    assert abs(cohens_dz(diff)) > abs(cohens_d(a, b))
    assert np.isnan(cohens_dz(np.ones(5)))


def test_bootstrap_hierarchique():
    """Sans variabilité entre graines d'entraînement, l'IC hiérarchique a la largeur de l'IC
    usuel ; avec une graine très différente des autres, il s'élargit nettement."""
    rng = np.random.default_rng(1)
    seeds = list(range(42, 72))
    ref = pd.Series(1000 + rng.normal(0, 10, 30), index=seeds)
    noise = rng.normal(0, 10, 30)
    same = pd.DataFrame([ref.to_numpy() - 20 + noise] * 5, columns=seeds)
    low, high = hierarchical_bootstrap_ci(same, ref, n_boot=2_000)
    assert low < -20 < high and high - low < 10
    spread = same.copy()
    spread.iloc[4] -= 150                       # un modèle nettement moins bon
    low2, high2 = hierarchical_bootstrap_ci(spread, ref, n_boot=2_000)
    assert high2 - low2 > 5 * (high - low)
