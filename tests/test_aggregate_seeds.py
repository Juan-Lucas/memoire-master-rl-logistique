"""Tests de T30 (M7) : agrégation sur les graines d'entraînement (IQM, bootstrap stratifié,
probabilité d'amélioration) et tests appariés avec plusieurs graines d'entraînement."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from memoire_master_rl_logistique.analysis import aggregate_seeds as ag
from memoire_master_rl_logistique.analysis import statistical_tests as st
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SPECIALISED

AGENT_LEVELS = {"Q-Learning": 3000.0, "SARSA": 2950.0, "DQN": 3400.0, "PPO": 3500.0}
HEURISTIC_LEVELS = {"Fixed": 3100.0, "ShortestPath": 3200.0}
SCENARIO_FACTOR = {"nominal": 1.0, "high_load": 1.3, "high_breakdown": 0.8}


def benchmark(train_seeds=(1, 2, 3, 4, 5), eval_seeds=range(42, 72), scenarios=MAIN_SCENARIOS):
    """Résultats synthétiques au format de run_benchmark : un effet par graine
    d'entraînement (décalage du modèle) et un bruit par graine d'évaluation."""
    rng = np.random.default_rng(0)
    rows = []
    for scenario in scenarios:
        factor = SCENARIO_FACTOR[scenario]
        noise = {s: rng.normal(0, 30) for s in eval_seeds}
        for policy, level in HEURISTIC_LEVELS.items():
            for s in eval_seeds:
                rows.append(dict(scenario=scenario, policy=policy, mode=None, train_seed=None,
                                 seed=s, truck_count=12, productivity_tph=factor * level + noise[s],
                                 specific_fuel_l_per_ton=0.4))
        for policy, level in AGENT_LEVELS.items():
            for t in train_seeds:
                offset = 20.0 * (t - 3)
                for s in eval_seeds:
                    rows.append(dict(scenario=scenario, policy=policy, mode=SPECIALISED,
                                     train_seed=t, seed=s, truck_count=12,
                                     productivity_tph=factor * level + offset + noise[s],
                                     specific_fuel_l_per_ton=0.5 - level / 1e4 + t / 1e3))
    return pd.DataFrame(rows)


# --- Statistiques de base (définitions de rliable) ----------------------------------

def test_iqm_robuste_a_un_run_aberrant():
    """Savoir expliquer : la moyenne interquartile ignore le run aberrant, pas la moyenne."""
    scores = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert ag.aggregate(scores, "iqm") == pytest.approx(3.0)
    assert ag.aggregate(scores, "mean") == pytest.approx(22.0)
    assert ag.aggregate(scores, "median") == pytest.approx(3.0)


def test_iqm_definition_de_rliable():
    """rliable.metrics.aggregate_iqm = moyenne tronquée à 25 % sur toute la matrice."""
    matrix = np.random.default_rng(1).normal(size=(5, 3))
    assert ag.aggregate(matrix) == pytest.approx(stats.trim_mean(matrix, 0.25, axis=None))


def test_bootstrap_stratifie_ne_melange_pas_les_regimes():
    matrix = np.column_stack([np.arange(5.0), 100 + np.arange(5.0)])
    samples = ag.stratified_bootstrap(matrix, n_boot=500)
    assert samples.shape == (500, 5, 2)
    assert set(np.unique(samples[:, :, 0])) <= set(matrix[:, 0])
    assert set(np.unique(samples[:, :, 1])) <= set(matrix[:, 1])


def test_intervalle_reproductible_et_autour_de_l_iqm():
    scores = [3400.0, 3420.0, 3380.0, 3500.0, 3390.0]
    first = ag.bootstrap_ci(scores, n_boot=5_000)
    assert first == ag.bootstrap_ci(scores, n_boot=5_000)
    assert first[0] <= ag.aggregate(scores) <= first[1]
    assert np.isnan(ag.bootstrap_ci([3400.0])).all()  # un seul run : rien à estimer


def test_probabilite_d_amelioration():
    better, worse = np.array([[5.0], [6.0], [7.0]]), np.array([[1.0], [2.0], [3.0]])
    assert ag.probability_of_improvement(better, worse)[0] == 1.0
    assert ag.probability_of_improvement(worse, better)[0] == 0.0
    assert ag.probability_of_improvement(better, better)[0] == 0.5

    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    # Définition de rliable : U de Mann-Whitney / (n_x n_y) par régime, moyenné.
    expected = np.mean([stats.mannwhitneyu(x[:, m], y[:, m], alternative="greater")[0] / 25
                        for m in range(3)])
    point, low, high = ag.probability_of_improvement(x, y, n_boot=2_000)
    assert point == pytest.approx(expected)
    assert low <= point <= high


def test_valeurs_de_rliable():
    """Références calculées avec rliable 1.2.0 (metrics + tirage de StratifiedBootstrap,
    installé à part : il ne s'importe pas avec pandas 3) sur ces mêmes matrices."""
    rng = np.random.default_rng(7)
    x, y = rng.normal(1.1, 0.05, size=(5, 3)), rng.normal(1.0, 0.08, size=(5, 3))
    assert ag.aggregate(x) == pytest.approx(1.0921932870060713, abs=1e-12)
    assert ag.bootstrap_ci(x) == pytest.approx((1.0745, 1.1067), abs=0.002)
    point, low, high = ag.probability_of_improvement(x, y)
    assert point == pytest.approx(0.9866666666666667, abs=1e-12)
    assert (low, high) == pytest.approx((0.9466666666666667, 1.0))
    with pytest.raises(ValueError, match="formes"):
        ag.probability_of_improvement(x, y[:4])


# --- Résultats de benchmark --------------------------------------------------------

def test_score_d_un_run_est_sa_moyenne_sur_l_evaluation():
    data = benchmark()
    scores = ag.run_scores(data, "productivity_tph")
    assert set(scores["policy"]) == set(AGENT_LEVELS)  # heuristiques exclues
    assert (scores["n_eval_seeds"] == 30).all()
    one = data[(data["scenario"] == "nominal") & (data["policy"] == "PPO")
               & (data["train_seed"] == 2)]
    row = scores[(scores["scenario"] == "nominal") & (scores["policy"] == "PPO")
                 & (scores["train_seed"] == 2)]
    assert row["score"].item() == pytest.approx(one["productivity_tph"].mean())


def test_runs_evalues_sur_des_graines_differentes_refuses():
    data = benchmark()
    data = data.drop(data[(data["policy"] == "DQN") & (data["train_seed"] == 4)
                          & (data["seed"] == 50)].index)
    with pytest.raises(ValueError, match="mêmes graines"):
        ag.run_scores(data, "productivity_tph")


def test_tableau_par_regime():
    table = ag.aggregate_table(benchmark(), "productivity_tph", n_boot=2_000)
    assert list(table["scenario"].unique()) == list(MAIN_SCENARIOS)
    assert list(table["policy"].unique()) == ["Q-Learning", "SARSA", "DQN", "PPO"]
    assert (table["n_train_seeds"] == 5).all()
    ppo = table[(table["scenario"] == "nominal") & (table["policy"] == "PPO")].iloc[0]
    # Runs = 3500 + {-40, -20, 0, 20, 40} + bruit moyen : l'IQM garde les 3 du milieu.
    assert ppo["iqm"] == pytest.approx(ppo["mean"])
    assert ppo["min"] < ppo["ci_low"] <= ppo["iqm"] <= ppo["ci_high"] < ppo["max"]


def test_scores_normalises_par_la_reference():
    data = benchmark()
    norm = ag.normalized_scores(data, "productivity_tph", reference="Fixed")
    raw = ag.run_scores(data, "productivity_tph")
    ref = data[(data["policy"] == "Fixed") & (data["scenario"] == "high_load")][
        "productivity_tph"].mean()
    pick = (raw["scenario"] == "high_load") & (raw["policy"] == "DQN") & (raw["train_seed"] == 1)
    assert norm.loc[pick.to_numpy(), "score"].item() == pytest.approx(raw[pick]["score"].item() / ref)
    with pytest.raises(ValueError, match="référence"):
        ag.normalized_scores(data, "productivity_tph", reference="MTWT")


def test_agregation_sur_tous_les_regimes():
    table = ag.aggregate_across_scenarios(benchmark(), "productivity_tph", n_boot=2_000)
    assert (table["n_scenarios"] == 3).all() and (table["n_train_seeds"] == 5).all()
    ppo = table[table["policy"] == "PPO"].iloc[0]
    assert ppo["iqm"] == pytest.approx(3500 / 3100, abs=0.02)  # PPO / Fixed, tous régimes
    assert ppo["ci_low"] <= ppo["iqm"] <= ppo["ci_high"]


def test_meme_nombre_de_runs_dans_chaque_regime():
    data = benchmark()
    data = data.drop(data[(data["scenario"] == "high_load") & (data["train_seed"] == 5)].index)
    with pytest.raises(ValueError, match="Nombre de runs"):
        ag.aggregate_across_scenarios(data, "productivity_tph", n_boot=100)


def test_probabilite_d_amelioration_dans_le_bon_sens():
    """Plus de productivité est mieux ; moins de carburant aussi."""
    data = benchmark()
    prod = ag.improvement_table(data, "productivity_tph", pairs=[("PPO", "Q-Learning")],
                                n_boot=500).set_index("scenario")
    fuel = ag.improvement_table(data, "specific_fuel_l_per_ton", pairs=[("PPO", "Q-Learning")],
                                n_boot=500).set_index("scenario")
    assert list(prod.index) == [*MAIN_SCENARIOS, "ensemble"]
    assert (prod["p_improvement"] == 1.0).all()
    assert (fuel["p_improvement"] == 1.0).all()  # PPO consomme moins par tonne


def test_commande_d_agregation(tmp_path):
    csv = tmp_path / "benchmark.csv"
    benchmark().to_csv(csv, index=False)
    tables = ag.main(csv)
    assert set(tables) == {f"{m}_{k}" for m in st.METRICS
                           for k in ("by_scenario", "overall", "improvement")}


def zero_tonnage(data, policy="DQN", train_seeds=(2, 4), scenario="nominal"):
    """Runs qui n'ont rien produit, écrits comme dans le CSV de la campagne (fuel / 1e-9)."""
    data = data.copy()
    rows = ((data["policy"] == policy) & data["train_seed"].isin(train_seeds)
            & (data["scenario"] == scenario))
    data.loc[rows, ["productivity_tph", "total_tonnage_t"]] = 0.0
    data.loc[rows, "specific_fuel_l_per_ton"] = 1.0848e12
    data["total_tonnage_t"] = data["total_tonnage_t"].fillna(1000.0)
    return data


def test_carburant_des_runs_sans_tonne_exclu_et_compte(tmp_path):
    """Ablation w4 = 0,5 de la campagne : deux DQN attendent tout le poste. Leur carburant
    par tonne n'est pas défini ; le tableau résume les 3 autres runs et compte les 2 exclus,
    au lieu d'afficher une moyenne de 4e11 L/t."""
    csv = tmp_path / "benchmark.csv"
    zero_tonnage(benchmark().assign(total_tonnage_t=1000.0)).to_csv(csv, index=False)
    data = st.load_results(csv)
    table = ag.aggregate_table(data, "specific_fuel_l_per_ton", n_boot=2_000)
    dqn = table[(table["scenario"] == "nominal") & (table["policy"] == "DQN")].iloc[0]
    assert (dqn["n_train_seeds"], dqn["n_undefined"]) == (3, 2)
    expected = np.mean([0.5 - 3400.0 / 1e4 + t / 1e3 for t in (1, 3, 5)])
    assert dqn["mean"] == pytest.approx(expected)
    assert dqn["max"] < 1.0
    # La productivité (0 t/h) reste définie : les 5 runs sont résumés.
    prod = ag.aggregate_table(data, "productivity_tph", n_boot=2_000)
    dqn = prod[(prod["scenario"] == "nominal") & (prod["policy"] == "DQN")].iloc[0]
    assert (dqn["n_train_seeds"], dqn["n_undefined"]) == (5, 0)


def test_run_sans_aucune_valeur_definie():
    data = zero_tonnage(benchmark(), train_seeds=(1, 2, 3, 4, 5))
    data = st.undefined_fuel_as_nan(data)
    table = ag.aggregate_table(data, "specific_fuel_l_per_ton", n_boot=2_000)
    dqn = table[(table["scenario"] == "nominal") & (table["policy"] == "DQN")].iloc[0]
    assert (dqn["n_train_seeds"], dqn["n_undefined"]) == (0, 5)
    assert np.isnan(dqn["iqm"])


def test_tests_apparies_refusent_un_carburant_non_defini(tmp_path):
    """Dans les régimes principaux, une valeur non définie serait ignorée en silence par la
    moyenne sur les graines d'entraînement : les tests appariés refusent."""
    csv = tmp_path / "benchmark.csv"
    zero_tonnage(benchmark()).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="non définie"):
        st.main(csv, pairs=[("PPO", "Fixed")])


# --- Tests appariés avec plusieurs graines d'entraînement ---------------------------

def test_moyenne_sur_les_graines_d_entrainement():
    data = benchmark()
    averaged = st.average_over_training_seeds(data)
    assert len(averaged) == 3 * (2 + 4) * 30  # une ligne par (régime, politique, graine)
    ppo = averaged[(averaged["scenario"] == "nominal") & (averaged["policy"] == "PPO")
                   & (averaged["seed"] == 42)]
    raw = data[(data["scenario"] == "nominal") & (data["policy"] == "PPO") & (data["seed"] == 42)]
    assert ppo["productivity_tph"].item() == pytest.approx(raw["productivity_tph"].mean())
    assert ppo["n_train_seeds"].item() == 5
    fixed = averaged[(averaged["policy"] == "Fixed") & (averaged["seed"] == 42)
                     & (averaged["scenario"] == "nominal")]
    assert fixed["n_train_seeds"].item() == 0  # heuristique : rien à moyenner

    table = st.paired_table(averaged, "nominal", "productivity_tph")
    assert table.shape == (30, 6)


def test_graine_d_entrainement_manquante_refusee():
    data = benchmark()
    data = data.drop(data[(data["policy"] == "SARSA") & (data["train_seed"] == 2)
                          & (data["seed"] == 42)].index)
    with pytest.raises(ValueError, match="manquantes"):
        st.average_over_training_seeds(data)


def test_resultats_d_avant_t26_inchanges():
    data = benchmark().drop(columns="train_seed")
    assert st.average_over_training_seeds(data) is data


def test_tests_apparies_acceptent_plusieurs_graines(tmp_path, capsys):
    """Avant T30, statistical_tests.main refusait un CSV à plusieurs graines d'entraînement."""
    csv = tmp_path / "benchmark.csv"
    benchmark().to_csv(csv, index=False)
    with pytest.raises(ValueError, match="average_over_training_seeds"):
        st.paired_table(st.load_results(csv), "nominal", "productivity_tph")
    st.main(csv, st.V1_PAIRS)  # CSV synthétique sans MTWT
    assert "Friedman" in capsys.readouterr().out
