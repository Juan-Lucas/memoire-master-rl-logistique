from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.plots.benchmark_comparisons import (
    main,
    pareto_front,
    summarise,
)

REFERENCES_CSV = (
    Path(__file__).parent.parent / "data" / "results_v1_article" / "benchmark_results.csv"
)


def test_pareto_front_points_domines():
    """On minimise le coût et on maximise le gain : seuls les compromis non battus restent."""
    cost = np.array([10, 20, 30, 25])
    gain = np.array([100, 200, 300, 150])  # le 4e point est battu par le 2e (moins cher, plus de gain)

    assert pareto_front(cost, gain).tolist() == [True, True, True, False]


def test_pareto_front_egalite_non_dominee():
    """Deux points identiques ne se dominent pas l'un l'autre."""
    assert pareto_front([5, 5], [50, 50]).tolist() == [True, True]


def test_summarise_moyenne_et_ic():
    """Une moyenne et un IC par (politique, régime) ; valeurs constantes → IC de largeur nulle."""
    df = pd.DataFrame(
        {
            "policy": ["A"] * 3 + ["B"] * 3,
            "scenario": ["nominal"] * 6,
            "productivity_tph": [10, 10, 10, 1, 2, 3],
        }
    )
    summary = summarise(df, "productivity_tph").set_index("policy")

    assert summary.loc["A", "mean"] == pytest.approx(10)
    assert (summary.loc["A", "ci_low"], summary.loc["A", "ci_high"]) == (10, 10)
    assert summary.loc["B", "ci_low"] <= 2 <= summary.loc["B", "ci_high"]


@pytest.mark.skipif(not REFERENCES_CSV.exists(), reason="CSV v1 absent")
def test_main_ecrit_les_trois_figures(tmp_path):
    """Sur le CSV v1, les trois figures sont écrites ; un second lancement refuse d'écraser."""
    outputs = main(REFERENCES_CSV, tmp_path)

    assert [p.name for p in outputs] == [
        "barplot_productivite_scenarios.png",
        "wait_time_comparison.png",
        "pareto_productivity_waiting.png",
    ]
    assert all(p.exists() for p in outputs)
    with pytest.raises(FileExistsError):
        main(REFERENCES_CSV, tmp_path)


def test_courbes_de_la_campagne(tmp_path):
    """Une figure, trois panneaux (régimes), à partir du CSV des courbes de la campagne ;
    jamais d'écrasement sans --overwrite."""
    from memoire_master_rl_logistique.plots.campaign_curves import main as curves_main

    rows = [dict(x=p, progress_pct=p, n_runs=5, mean=4000 + p, ci_low=3990 + p,
                 ci_high=4010 + p, algorithm=a, scenario=s)
            for a in ("PPO", "DQN", "Q-Learning", "SARSA")
            for s in ("nominal", "high_load", "high_breakdown") for p in (0.5, 50.5, 99.5)]
    csv = tmp_path / "courbes.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    out = curves_main(csv, tmp_path / "fig")
    assert out.exists() and out.stat().st_size > 0
    with pytest.raises(FileExistsError):
        curves_main(csv, tmp_path / "fig")


def test_figures_avec_plusieurs_graines_d_entrainement(tmp_path):
    """Campagne : un agent a 5 modèles. La barre est la moyenne de ses modèles, et l'IC
    porte sur les 30 réplications (moyennées sur les modèles), pas sur 150 épisodes."""
    from memoire_master_rl_logistique.analysis import statistical_tests as st
    from memoire_master_rl_logistique.plots import benchmark_comparisons as bc

    rows = []
    for scenario, trucks in (("nominal", 12), ("high_load", 18), ("high_breakdown", 12)):
        for s in range(42, 72):
            rows.append(dict(scenario=scenario, policy="Fixed", mode=None, train_seed=None,
                             seed=s, truck_count=trucks, productivity_tph=4000.0 + s,
                             queue_wait_min_per_truck=10.0))
            for t in range(1, 6):
                rows.append(dict(scenario=scenario, policy="PPO", mode="spécialisé",
                                 train_seed=t, seed=s, truck_count=trucks,
                                 productivity_tph=4100.0 + s + 10 * t,
                                 queue_wait_min_per_truck=12.0))
    csv = tmp_path / "campagne.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    df = st.average_over_training_seeds(st.load_results(csv))
    ppo = df[(df["policy"] == "PPO") & (df["scenario"] == "nominal")]
    assert len(ppo) == 30 and ppo["productivity_tph"].mean() == pytest.approx(4100 + 56.5 + 30)
    assert len(bc.main(csv, tmp_path / "fig")) == 3
