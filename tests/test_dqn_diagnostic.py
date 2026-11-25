"""Diagnostic de DQN par les valeurs Q (M6 du rapport du 17/11/2026)."""

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.analysis import dqn_diagnostic as dd
from memoire_master_rl_logistique.rl.time_discount import per_minute_gamma
from memoire_master_rl_logistique.simulation.reward import RewardWeights


class FakeModel:
    """Modèle minimal : renvoie des valeurs Q fixées pour chaque observation."""

    def __init__(self, table):
        self.table = table

    class policy:  # noqa: N801
        @staticmethod
        def obs_to_tensor(obs):
            import torch as th
            return th.as_tensor(obs, dtype=th.float32).reshape(1, -1), None

    def q_net(self, tensor):
        import torch as th
        key = int(tensor.numpy().ravel()[0])
        return th.as_tensor(self.table[key], dtype=th.float32).reshape(1, -1)


def test_ecart_et_attendre_en_tete():
    """Écart = Q(ATTENDRE) − meilleure affectation ; ATTENDRE en tête quand il est positif."""
    model = FakeModel({0: [10.0, 9.0, 10.2], 1: [10.0, 9.0, 9.7]})
    states = [np.array([0.0]), np.array([1.0])]
    d = dd.diagnose(model, states, wait_action=2, gamma=0.995, weights=RewardWeights())
    assert d["gap"].tolist() == pytest.approx([0.2, -0.3], abs=1e-5)
    assert d["wait_first"].tolist() == [True, False]
    assert d["scale"].tolist() == pytest.approx([10.2, 10.0], abs=1e-5)


def test_ecart_attendu():
    """Écart attendu d'une attente de 0,2 min : actualisation de la valeur et coût d'ATTENDRE
    (w4 × 0,2 / 30 + w1 × 0,2 / 23,2)."""
    w = RewardWeights()
    cost = w.wait * 0.2 / 30 + w.production * 0.2 / 23.2
    assert dd.wait_cost(w) == pytest.approx(cost)
    model = FakeModel({0: [100.0, 50.0, 99.0]})
    d = dd.diagnose(model, [np.array([0.0])], wait_action=2, gamma=0.995, weights=w)
    g02 = per_minute_gamma(0.995) ** 0.2
    assert d["expected_gap"].iloc[0] == pytest.approx(-(1 - g02) * 100.0 - cost)


def test_resume():
    states = pd.DataFrame({"train_seed": [1, 1, 2], "states_of": ["own"] * 3,
                           "gap": [0.1, 0.3, -0.2], "scale": [100.0, 100.0, 100.0],
                           "wait_first": [True, True, False], "expected_gap": [-0.1] * 3})
    s = dd.summarise(states).set_index("train_seed")
    assert s.loc[1, "wait_first_share"] == 1.0
    assert s.loc[1, "median_gap"] == pytest.approx(0.2)
    assert s.loc[2, "median_relative_gap_pct"] == pytest.approx(-0.2)
