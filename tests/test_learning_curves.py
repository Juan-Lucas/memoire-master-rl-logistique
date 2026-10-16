"""Tests de T30 (M7, N7) : courbes d'apprentissage et de diagnostic agrégées sur les graines,
lues dans les journaux des modèles évalués (plus de réentraînement séparé)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.analysis import learning_curves as lc
from memoire_master_rl_logistique.analysis import sample_efficiency as se
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.plots import (
    convergence_deeprl,
    convergence_qlearning_sarsa,
    exploration_exploitation,
    policy_value_loss,
    td_error,
)
from memoire_master_rl_logistique.rl.model_paths import model_dir
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1, 2)
REGIME = "nominal"
EPISODE_MINUTES = 60.0
BUDGETS = {"Q-Learning": 6, "SARSA": 6, "DQN": 3_000, "PPO": 2_048}


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """Mini-campagne : 4 agents × 2 graines, dans les dossiers évalués par run_benchmark."""
    root = tmp_path_factory.mktemp("campagne")
    for seed in SEEDS:
        common = dict(seed=seed, episode_minutes=EPISODE_MINUTES)
        for algo, train in (("Q-Learning", train_q_learning), ("SARSA", train_sarsa)):
            train(n_episodes=BUDGETS[algo], output_dir=str(model_dir(root, algo, REGIME, seed)),
                  **common)
        for algo, train in (("DQN", train_dqn), ("PPO", train_ppo)):
            train(total_timesteps=BUDGETS[algo], progress_bar=False,
                  output_dir=str(model_dir(root, algo, REGIME, seed)), **common)
    return root


# --- Journaux écrits par les entraînements ------------------------------------------

def test_journal_par_episode_des_agents_profonds(campaign):
    """Chaque épisode garde le pas atteint (axe commun des graines) et sa récompense."""
    for algo in ("DQN", "PPO"):
        log = pd.read_csv(model_dir(campaign, algo, REGIME, 1) / "training_kpis.csv")
        assert list(log.columns) == ["episode", "timesteps", "tonnage_t", "wait_min", "fuel_l",
                                     "reward"]
        assert len(log) >= 3
        assert log["timesteps"].is_monotonic_increasing
        assert log["timesteps"].iloc[-1] <= BUDGETS[algo]
        assert log["reward"].notna().all()


def test_journal_tabulaire_avec_productivite(campaign):
    log = pd.read_csv(model_dir(campaign, "SARSA", REGIME, 2) / "training_diagnostics.csv")
    assert list(log.columns) == ["episode", "total_reward", "epsilon", "mean_abs_td_error",
                                 "tonnage_t"]
    assert len(log) == BUDGETS["SARSA"] and (log["tonnage_t"] > 0).all()


# --- Résumé d'un run sur une grille commune -----------------------------------------

def test_moyenne_par_intervalle():
    log = pd.DataFrame({"x": [0, 1, 2, 3, 6, 10], "value": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]})
    values = lc.binned(log, budget=10, n_bins=5)
    # [0,2) → 1, 3 ; [2,4) → 5, 7 ; [4,6) vide ; [6,8) → 9 ; [8,10] → 11 (x = budget inclus)
    np.testing.assert_allclose(values, [2.0, 6.0, np.nan, 9.0, 11.0])


def test_moyenne_et_intervalle_des_runs():
    matrix = np.array([[1.0, 10.0, np.nan], [3.0, 10.0, 5.0], [2.0, 10.0, np.nan]])
    summary = lc.summarize(np.array([0.5, 1.5, 2.5]), matrix, n_boot=2_000)
    assert summary["n_runs"].tolist() == [3, 3, 1]
    np.testing.assert_allclose(summary["mean"], [2.0, 10.0, 5.0])
    assert summary.loc[0, "ci_low"] < 2.0 < summary.loc[0, "ci_high"]
    assert summary.loc[1, "ci_low"] == summary.loc[1, "ci_high"] == 10.0
    assert np.isnan(summary.loc[2, "ci_low"])  # un seul run : pas d'intervalle


# --- Courbes de la mini-campagne ----------------------------------------------------

def test_les_courbes_viennent_des_modeles_evalues(campaign):
    """N7 : le dossier lu est celui que run_benchmark évalue (model_dir), pas un *_losscurve."""
    dirs = lc.run_dirs(campaign, "PPO", REGIME, SEEDS)
    assert dirs == {s: model_dir(campaign, "PPO", REGIME, s) for s in SEEDS}


def test_courbe_de_productivite(campaign):
    for algo in ("Q-Learning", "SARSA", "DQN", "PPO"):
        x, matrix, budget = lc.curve_matrix(campaign, algo, REGIME, "productivity_tph", SEEDS,
                                            n_bins=4)
        assert budget == BUDGETS[algo] and matrix.shape == (2, 4)

    # Q-Learning, 6 épisodes en 3 intervalles : chaque point = moyenne de 2 épisodes, en t/h.
    _, matrix, _ = lc.curve_matrix(campaign, "Q-Learning", REGIME, "productivity_tph", SEEDS,
                                   n_bins=3)
    log = pd.read_csv(model_dir(campaign, "Q-Learning", REGIME, 1) / "training_diagnostics.csv")
    expected = (log["tonnage_t"] / (EPISODE_MINUTES / 60)).to_numpy().reshape(3, 2).mean(axis=1)
    np.testing.assert_allclose(matrix[0], expected, rtol=1e-6)

    curve = lc.learning_curve(campaign, "Q-Learning", REGIME, "productivity_tph", SEEDS,
                              n_bins=3, n_boot=500)
    assert curve["n_runs"].tolist() == [2, 2, 2]
    np.testing.assert_allclose(curve["mean"], matrix.mean(axis=0))
    np.testing.assert_allclose(curve["progress_pct"], [100 / 6, 50, 500 / 6])


def test_toutes_les_courbes_de_diagnostic(campaign):
    for name, sources in lc.CURVES.items():
        for algo in sources:
            curve = lc.learning_curve(campaign, algo, REGIME, name, SEEDS, n_bins=4, n_boot=200)
            assert curve["n_runs"].max() == 2, (name, algo)
    eps = lc.learning_curve(campaign, "SARSA", REGIME, "epsilon", SEEDS, n_bins=6, n_boot=200)
    assert eps["mean"].iloc[0] == pytest.approx(1.0)  # ε de départ du YAML
    assert eps["mean"].is_monotonic_decreasing
    entropy = lc.learning_curve(campaign, "PPO", REGIME, "entropy", SEEDS, n_bins=2, n_boot=200)
    assert (entropy["mean"].dropna() > 0).all()  # entropie = − perte d'entropie de SB3


def test_courbe_inexistante_pour_un_algorithme(campaign):
    with pytest.raises(ValueError, match="td_error"):
        lc.learning_curve(campaign, "PPO", REGIME, "td_error", SEEDS)


def test_run_manquant(campaign):
    with pytest.raises(FileNotFoundError, match="seed3"):
        lc.curve_matrix(campaign, "DQN", REGIME, "productivity_tph", (1, 3))


def test_budgets_differents_refuses(tmp_path):
    for seed, episodes in ((1, 2), (2, 3)):
        train_q_learning(n_episodes=episodes, seed=seed, episode_minutes=30.0,
                         output_dir=str(model_dir(tmp_path, "Q-Learning", REGIME, seed)))
    with pytest.raises(ValueError, match="budgets"):
        lc.curve_matrix(tmp_path, "Q-Learning", REGIME, "reward", SEEDS)


def test_journal_d_avant_t30_refuse(campaign, tmp_path):
    """Un training_kpis.csv sans colonne timesteps (modèles v1) : message explicite."""
    old = tmp_path / "ppo_nominal_seed1"
    old.mkdir()
    src = model_dir(campaign, "PPO", REGIME, 1)
    (old / "metadata.json").write_text((src / "metadata.json").read_text(encoding="utf-8"),
                                       encoding="utf-8")
    pd.read_csv(src / "training_kpis.csv").drop(columns=["timesteps", "reward"]).to_csv(
        old / "training_kpis.csv", index=False)
    with pytest.raises(ValueError, match="T30"):
        lc.run_log(old, "PPO", "productivity_tph")


def test_export_csv_sans_ecrasement(campaign, tmp_path):
    out = tmp_path / "courbes.csv"
    table = lc.main([str(campaign), str(out), "--scenarios", REGIME, "--train-seeds", "1", "2"])
    assert set(table["algorithm"]) == {"Q-Learning", "SARSA", "DQN", "PPO"}
    assert len(table) == 4 * lc.N_BINS
    with pytest.raises(FileExistsError):
        lc.main([str(campaign), str(out), "--scenarios", REGIME])


# --- Efficacité d'échantillonnage ---------------------------------------------------

def test_point_de_convergence():
    x = np.arange(10) + 0.5
    values = np.array([0, 50, 90, 96, 99, np.nan, 100, 97, 101, 100.0])
    # Reste à ±5 % de 100 à partir de l'indice 3 (x = 3,5), le nan est ignoré.
    assert se.convergence_point(x, values, budget=10) == pytest.approx(35.0)
    assert se.convergence_point(x, np.full(10, 7.0), budget=10) == pytest.approx(5.0)


def test_efficacite_par_graine(campaign):
    table = se.efficiency_table(campaign, scenarios=[REGIME], train_seeds=SEEDS)
    assert len(table) == 4 and (table["n_train_seeds"] == 2).all()
    assert ((table["fastest"] <= table["pct_budget_95"])
            & (table["pct_budget_95"] <= table["slowest"])).all()


# --- Figures -------------------------------------------------------------------------

FIGURES = [
    (convergence_qlearning_sarsa, ["learning_curves_nominal.png"]),
    (convergence_deeprl, ["learning_curves_deeprl_nominal.png"]),
    (td_error, ["td_error_nominal.png"]),
    (policy_value_loss, ["loss_dqn_nominal.png", "loss_ppo_nominal.png"]),
    (exploration_exploitation, ["exploration_exploitation_nominal.png"]),
]


@pytest.mark.parametrize(("module", "names"), FIGURES,
                         ids=[m.__name__.rsplit(".", 1)[-1] for m, _ in FIGURES])
def test_figures_ecrites_sans_ecrasement(campaign, tmp_path, module, names):
    paths = module.main(campaign, tmp_path, scenarios=[REGIME], train_seeds=SEEDS)
    assert [p.name for p in paths] == names
    assert all(p.stat().st_size > 0 for p in paths)
    with pytest.raises(FileExistsError):
        module.main(campaign, tmp_path, scenarios=[REGIME], train_seeds=SEEDS)


def test_intervalles_vides_relies_dans_la_figure():
    """Un intervalle sans valeur ne coupe pas la courbe (sinon des points isolés
    disparaissent, comme l'entropie de PPO journalisée tous les 1 024 pas)."""
    import matplotlib.pyplot as plt

    from memoire_master_rl_logistique.plots.common import plot_band

    curve = pd.DataFrame({"x": [0.5, 1.5, 2.5, 3.5], "n_runs": [2, 0, 1, 2],
                          "mean": [1.0, np.nan, 2.0, 3.0],
                          "ci_low": [0.5, np.nan, np.nan, 2.5],
                          "ci_high": [1.5, np.nan, np.nan, 3.5]})
    fig, ax = plt.subplots()
    plot_band(ax, curve, "#000000", "test")
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.5, 2.5, 3.5])
    assert "moyenne de 2 graines" in ax.lines[0].get_label()
    plt.close(fig)


def test_regime_absent_ne_produit_aucune_figure(campaign, tmp_path):
    with pytest.raises(FileNotFoundError):
        td_error.main(campaign, tmp_path / "figs", scenarios=[REGIME, "high_load"],
                      train_seeds=SEEDS)
    assert {p.name for p in (tmp_path / "figs").iterdir()} <= {"td_error_nominal.png"}


# --- Réentraînements séparés supprimés (N7) ------------------------------------------

def test_plus_de_reentrainement_pour_les_figures():
    assert importlib.util.find_spec("memoire_master_rl_logistique.rl.diagnostics") is None
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "rl.diagnostics" not in makefile and "losscurve" not in makefile
    assert "reports/figures " not in makefile and "FIGURES ?= reports/figures_v2" in makefile


def test_metadonnees_suffisent_a_la_courbe(campaign):
    """Le budget et la durée du poste viennent du metadata.json du run."""
    meta = json.loads((model_dir(campaign, "DQN", REGIME, 2) / "metadata.json").read_text(
        encoding="utf-8"))
    assert meta["training"]["total_timesteps"] == BUDGETS["DQN"]
    assert meta["env"]["episode_minutes"] == SCENARIOS[REGIME].episode_minutes / 8
