"""Ablation sur w4 : PPO et DQN seulement, dans le dossier de la campagne."""

import pandas as pd
import pytest

from memoire_master_rl_logistique.experiments import ablation_study
from memoire_master_rl_logistique.experiments.ablation_study import (
    ABLATION_SCENARIOS,
    ablation_table,
    run_ablation,
)


def test_regimes_de_l_ablation():
    assert ABLATION_SCENARIOS == ["ablation_w4_00", "ablation_w4_01", "ablation_w4_05"]


def test_lancement_dans_le_dossier_de_la_campagne(monkeypatch, tmp_path):
    """Seuls les régimes d'ablation sont lancés ; le nominal (w4 = 0,3) vient de la campagne."""
    calls = []
    monkeypatch.setattr(ablation_study, "run_all_benchmarks",
                        lambda out, **kw: calls.append((out, kw)) or tmp_path / "b.csv")

    assert run_ablation(tmp_path, n_workers=4) == tmp_path / "b.csv"
    assert calls == [(str(tmp_path), {"scenario_names": ABLATION_SCENARIOS, "n_workers": 4})]


def _rows(scenario, policy, mode, values):
    return [{"scenario": scenario, "policy": policy, "mode": mode, "train_seed": ts,
             "seed": 42, "productivity_tph": v, "wait_share_pct": 0.0}
            for ts, v in enumerate(values, start=1)]


def test_tableau_w4(tmp_path):
    """w4 lu dans la configuration ; au point nominal, PPO et DQN spécialisés et les
    heuristiques, sans Q-Learning ni SARSA ; les agents transférés sont ignorés."""
    campaign = pd.DataFrame(
        _rows("nominal", "PPO", "spécialisé", [4400, 4420])
        + _rows("nominal", "Q-Learning", "spécialisé", [4100])
        + _rows("nominal", "MTWT", None, [4443])
        + _rows("high_load", "PPO", "transféré", [1]))
    ablation = pd.DataFrame(
        _rows("ablation_w4_00", "PPO", "spécialisé", [4300, 4320])
        + _rows("ablation_w4_05", "DQN", "spécialisé", [3000]))
    paths = [tmp_path / "campagne.csv", tmp_path / "ablation.csv"]
    campaign.to_csv(paths[0], index=False)
    ablation.to_csv(paths[1], index=False)

    table = ablation_table(paths)

    means = table[("productivity_tph", "mean")]
    assert means.to_dict() == {(0.0, "PPO"): 4310.0, (0.3, "MTWT"): 4443.0,
                               (0.3, "PPO"): 4410.0, (0.5, "DQN"): 3000.0}
    assert table.loc[(0.3, "PPO"), ("productivity_tph", "std")] == pytest.approx(14.14, abs=0.01)
