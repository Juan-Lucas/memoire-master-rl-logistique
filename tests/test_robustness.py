from pathlib import Path

import pandas as pd
import pytest

from memoire_master_rl_logistique.experiments.reporting import order_policies, prepare_outputs
from memoire_master_rl_logistique.experiments.robustness_analysis import (
    main,
    plot_robustness_heatmap,
    robustness_ratios,
    wait_change_pct,
)

REFERENCES_CSV = (
    Path(__file__).parent.parent / "data" / "results_v1_article" / "benchmark_results.csv"
)


def make_robustness_rows() -> pd.DataFrame:
    """Deux politiques fictives, une graine, les trois régimes.

    A : même production par camion en forte charge (ratio 1), −10 % en fortes pannes ;
        attente 20 → 30 (+50 %) en forte charge, 20 → 10 (−50 %) en fortes pannes.
    B : même production totale avec 18 camions au lieu de 12 (ratio 2/3) ; attente constante.
    """
    return pd.DataFrame(
        {
            "scenario": ["nominal", "high_load", "high_breakdown"] * 2,
            "policy": ["A"] * 3 + ["B"] * 3,
            "truck_count": [12, 18, 12, 12, 18, 12],
            "productivity_tph": [3600, 5400, 3240, 3600, 3600, 3600],
            "avg_wait_min_per_truck": [20, 30, 10, 20, 20, 20],
        }
    )


def test_ratio_par_camion():
    """Le ratio est calculé par camion : ajouter des camions ne suffit pas à l'augmenter."""
    ratios = robustness_ratios(make_robustness_rows())

    assert ratios.loc["A", "high_load"] == pytest.approx(1.0)
    assert ratios.loc["B", "high_load"] == pytest.approx(2 / 3)
    assert ratios.loc["A", "high_breakdown"] == pytest.approx(0.9)


def test_wait_change_pct():
    """Variation de l'attente en % du nominal, positive quand l'attente augmente."""
    wait = wait_change_pct(make_robustness_rows())

    assert wait.loc["A", "high_load"] == pytest.approx(50.0)
    assert wait.loc["A", "high_breakdown"] == pytest.approx(-50.0)
    assert wait.loc["B", "high_load"] == pytest.approx(0.0)


def test_regime_manquant_leve_une_erreur():
    """Sans régime nominal, aucun ratio n'a de sens : erreur explicite."""
    rows = make_robustness_rows()
    rows = rows[rows["scenario"] != "nominal"]

    with pytest.raises(ValueError, match="nominal"):
        robustness_ratios(rows)


def test_order_policies_garde_les_inconnues():
    """Les politiques connues suivent l'ordre de l'article ; les inconnues ne sont pas perdues."""
    assert order_policies(["PPO", "B", "Fixed", "A"]) == ["Fixed", "PPO", "A", "B"]


def test_prepare_outputs_refuse_d_ecraser(tmp_path):
    """Un fichier existant n'est pas écrasé sans --overwrite."""
    (tmp_path / "figure.png").write_text("ancienne figure")

    with pytest.raises(FileExistsError):
        prepare_outputs(tmp_path, ["figure.png"], overwrite=False)
    assert prepare_outputs(tmp_path, ["figure.png"], overwrite=True) == [tmp_path / "figure.png"]


def test_heatmap_cree_le_fichier(tmp_path):
    """La heatmap s'écrit sans erreur, y compris pour des politiques hors de l'ordre de l'article."""
    rows = make_robustness_rows()
    out = tmp_path / "heatmap.png"

    plot_robustness_heatmap(robustness_ratios(rows), wait_change_pct(rows), out)

    assert out.exists()


@pytest.mark.skipif(not REFERENCES_CSV.exists(), reason="CSV v1 absent")
def test_main_retrouve_le_tableau_13(tmp_path):
    """Sur le CSV v1, les ratios sont ceux du tableau 13 de l'article v2 et les sorties sont écrites."""
    table = main(REFERENCES_CSV, tmp_path)

    assert table.loc["Fixed", "ratio_high_load"] == pytest.approx(0.962, abs=5e-4)
    assert table.loc["Fixed", "ratio_high_breakdown"] == pytest.approx(0.938, abs=5e-4)
    assert table.loc["PPO", "ratio_high_load"] == pytest.approx(0.822, abs=5e-4)
    assert table.loc["DQN", "wait_pct_high_load"] == pytest.approx(9.3, abs=0.05)
    assert (tmp_path / "robustness_ratios.csv").exists()
    assert (tmp_path / "robustness_heatmap.png").exists()

    with pytest.raises(FileExistsError):
        main(REFERENCES_CSV, tmp_path)
