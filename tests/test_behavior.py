"""Tests de T12 bis : comportement des politiques à partir du journal des cycles."""

from pathlib import Path

import pandas as pd
import pytest

from memoire_master_rl_logistique.analysis.behavior import (
    behaviour_latex,
    behaviour_runs,
    behaviour_summary,
    episode_behaviour,
)
from memoire_master_rl_logistique.experiments.policies import agent_policies, heuristic_policies
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS
from memoire_master_rl_logistique.plots.behaviour import main as plot_main
from memoire_master_rl_logistique.simulation.entities import CycleRecord

V1_MODELS = Path(__file__).parent.parent / "data" / "results_v1_article"
SHOVELS = ["S1", "S2", "S3"]


def cycle(shovel, start, end):
    return CycleRecord("cycle", "T1", start, end, shovel_id=shovel, dump_id="D1",
                       load_min=end - start)


def test_episode_behaviour():
    """3 cycles à S1, 1 à S2 et 1 ATTENDRE : 75 / 25 / 0 % des cycles, 20 % d'ATTENDRE."""
    log = [cycle("S1", 0, 20), cycle("S1", 20, 40), cycle("S2", 40, 70), cycle("S1", 70, 90),
           CycleRecord("wait", "T1", 90, 95, voluntary_wait_min=5)]

    row = episode_behaviour(log, SHOVELS)

    assert (row["share_S1_pct"], row["share_S2_pct"], row["share_S3_pct"]) == (75, 25, 0)
    assert row["wait_share_pct"] == 20
    assert row["mean_cycle_min"] == pytest.approx(22.5)
    assert row["n_decisions"] == 5


def test_heuristiques_sur_les_trois_regimes():
    """Test du plan (heuristiques) : Pelle la plus proche et Plus court chemin font 100 % de
    leurs cycles à la pelle 1 dans les trois régimes ; aucune heuristique n'utilise ATTENDRE."""
    summary = behaviour_summary(behaviour_runs(heuristic_policies, MAIN_SCENARIOS, seeds=[42, 43]))
    table = summary.set_index(["policy", "scenario"])

    for scenario in MAIN_SCENARIOS:
        for policy in ("Nearest", "ShortestPath"):
            assert table.loc[(policy, scenario), "share_S1_pct"] == 100
        for policy in ("FIFO", "Fixed"):
            assert max(table.loc[(policy, scenario), [f"share_{s}_pct" for s in SHOVELS]]) < 40
    assert (summary["wait_share_pct"] == 0).all()


def test_tableau_latex():
    """Une ligne par politique : « P1 / P2 / P3 & ATTENDRE » pour chaque régime."""
    summary = pd.DataFrame([
        {"scenario": s, "policy": "Nearest", "share_S1_pct": 100.0, "share_S2_pct": 0.0,
         "share_S3_pct": 0.0, "wait_share_pct": 0.0} for s in MAIN_SCENARIOS
    ])

    line = behaviour_latex(summary, {"Nearest": "Pelle la plus proche"})

    assert line == "Pelle la plus proche & " + " & ".join(["100 / 0 / 0 & 0"] * 3) + " \\\\\n"


def test_figure(tmp_path):
    """La figure se construit à partir du CSV de synthèse."""
    runs = behaviour_runs(heuristic_policies, ["nominal"], seeds=[42])
    csv = tmp_path / "behaviour_summary.csv"
    behaviour_summary(runs).to_csv(csv, index=False)

    out = plot_main(csv, tmp_path / "fig")

    assert out.exists()


@pytest.mark.skip(reason="historique : réussi le 15 octobre 2026 (commit d1cec48, 3 min 30). "
                         "Depuis T18, l'observation a 18 entrées : les DQN et PPO v1 (147 entrées) "
                         "ne peuvent plus être rejoués.")
def test_modeles_v1_attendre_en_forte_charge():
    """Test du plan (agents) : avec les modèles v1, en forte charge, sur 30 graines,
    PPO choisit ATTENDRE dans ≈ 87 % de ses décisions et DQN dans ≈ 64 %.

    Historique : valable jusqu'à T18 (observation de taille fixe), qui rend les modèles
    DQN et PPO v1 illisibles. Le résultat est consigné dans le journal (T12 bis).

    Toujours ignoré, même avec ``RUN_SLOW=1`` : gardé comme trace du test du plan. Pour le
    rejouer, revenir au commit d1cec48. Le même contrôle sur les modèles de la campagne v2
    est prévu en T31.
    """
    def deep_agents(scenario):
        return agent_policies(scenario, V1_MODELS, allow_legacy=True, algorithms=("DQN", "PPO"))

    with pytest.warns(UserWarning):
        summary = behaviour_summary(behaviour_runs(deep_agents, ["high_load"]))
    wait = summary.set_index("policy")["wait_share_pct"]

    assert wait["PPO"] == pytest.approx(87, abs=0.5)
    assert wait["DQN"] == pytest.approx(64, abs=0.5)
