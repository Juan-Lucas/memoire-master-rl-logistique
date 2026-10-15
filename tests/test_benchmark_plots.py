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
