import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.analysis.statistical_tests import (
    format_p,
    friedman_test,
    interaction_test,
    load_results,
    main,
    paired_bootstrap_ci,
    paired_comparisons,
    paired_table,
)


def make_rows() -> pd.DataFrame:
    """Renvoie un DataFrame avec des données de test."""

    policies = ['PPO', 'Fixed']
    seeds = [42, 43, 44]
    policies_seeds = list(itertools.product(policies, seeds))

    data = {
        "policy" : [ps[0] for ps in policies_seeds],
        "seed" : [ps[1] for ps in policies_seeds],
        "scenario" : ["nominal" for _ in policies_seeds],
        "productivity_tph": [85.5, 87.2, 84.1, 72.0, 71.5, 73.1],
        "inference_time_ms_mean": [12.4, 12.6, 12.3, 5.1, 5.0, 5.2]
    }

    return pd.DataFrame(data)

def write_data(tmp_path: Path, data: pd.DataFrame):
    """Écrit les données dans un fichier csv et renvoie son chemin."""

    data.to_csv(tmp_path / "test.csv", index=False)
    return tmp_path / "test.csv"

def test_fichier_propre(tmp_path: Path):
    """Teste si le fichier csv contient les données attendues (lignes)."""

    data_path = write_data(tmp_path, make_rows())

    data = load_results(data_path, n_expected=3)

    assert len(data) == 6

def test_doublons_identiques_retires(tmp_path: Path):
    """Teste si les doublons identiques sont retirés."""

    data = pd.concat([make_rows(), make_rows()], ignore_index=True)

    data.loc[[6, 7, 8, 9, 10, 11], "inference_time_ms_mean"] = 3.1

    result = load_results(write_data(tmp_path, data), n_expected=3)

    assert len(result) == 6

def test_doublon_contradictoire(tmp_path: Path):
    """Teste un doublon contradictoire doit lever ValueError"""

    data = make_rows()

    row_duplicated = pd.DataFrame({
        "policy" : ["PPO"],
        "seed" : [42],
        "scenario" : ["nominal"],
        "productivity_tph": [23],
        "inference_time_ms_mean": [12.4]
    })

    data_duplicated = pd.concat([data, row_duplicated], ignore_index=True)

    data_path = write_data(tmp_path, data_duplicated)

    with pytest.raises(ValueError, match="contradictoires"):
        load_results(data_path, n_expected=3)

def test_groupe_incomplet(tmp_path: Path):
    """Teste s'il y a un groupe incomplet."""

    data = make_rows().iloc[:-1]

    data_path = write_data(tmp_path, data)

    with pytest.raises(ValueError, match="incomplets"):
        load_results(data_path, n_expected=3)

def test_paired_table_forme():
    """Teste la forme de la table pivotée."""

    data = paired_table(make_rows(), "nominal", "productivity_tph")

    assert data.shape == (3, 2)
    assert data.loc[42, "PPO"] == 85.5

def test_paired_table_graines_differentes():
    """Teste s'il y a des cases vides."""

    data = make_rows()

    data.loc[5, "seed"] = 45

    with pytest.raises(ValueError, match="Cases vides"):
        paired_table(data, "nominal", "productivity_tph")

def test_friedman_classement_parfait():
    """Un classement parfait A < B < C sur 10 graines donne χ² = 20."""
    data = {
        "A": [1] * 10,
        "B": [2] * 10,
        "C": [3] * 10,
    }

    data_table = pd.DataFrame(data)

    statistique, p_value = friedman_test(data_table)

    assert statistique == pytest.approx(20)
    assert p_value < 0.001

def make_table():
    """Renvoie un DataFrame à trois colonnes."""
    data = {
        "A": [100] * 30,
        "B": [100 + i for i in range(1, 31)],
        "C": [100 + (i * 2) for i in range(1, 31)],
    }

    return pd.DataFrame(data)

def test_wilcoxon_domination_totale():
    """B bat A sur les 30 graines : p = 2/2³⁰, le minimum possible."""

    data_table = paired_comparisons(make_table(), [("B", "A")])

    assert data_table.loc[0, "mean_diff"] == pytest.approx(15.5)
    assert data_table.loc[0, "p_value"] == pytest.approx(2/2**30)

def test_wilcoxon_bilateral():
    """Bilatéral : inverser la paire ne change pas p. """

    data_table = paired_comparisons(make_table(), [("B", "A"), ("A", "B")])

    assert data_table.loc[0, "p_value"] == data_table.loc[1, "p_value"]

def test_holm_trois_paires():
    """Holm sur 3 p-values égales : chacune devient 3p. """

    data_table = paired_comparisons(make_table(), [("B", "A"), ("C", "A"), ("C", "B")])

    assert data_table.loc[0, "p_holm"] == pytest.approx(3 * 2 / 2**30)
    assert data_table.loc[1, "p_holm"] == pytest.approx(3 * 2 / 2**30)
    assert data_table.loc[2, "p_holm"] == pytest.approx(3 * 2 / 2**30)

def test_bootstrap_differences_constantes():
    """Des différences toutes égales à 5 donnent IC = (5, 5)."""

    diff = [5] * 30
    low, high = paired_bootstrap_ci(diff)
    assert low == 5
    assert high == 5

def test_bootstrap_reproductible():
    """Même graine, même résultat."""

    table = make_table()

    diff = table["B"] - table["A"]
    first = paired_bootstrap_ci(diff)
    second = paired_bootstrap_ci(diff)
    third = paired_bootstrap_ci(diff, seed = 1)
    assert first == second
    assert first != third

def test_bootstrap_encadre_la_moyenne():
    """L'IC contient la moyenne des différences (15,5). """

    table = make_table()

    diff = table["B"] - table["A"]
    low, high = paired_bootstrap_ci(diff)
    assert low < 15.5 < high

def test_comparisons_contient_ic():
    """Chaque comparaison contient son IC bootstrap. """

    data_table = paired_comparisons(make_table(), [("B", "A")])
    assert data_table.loc[0, "ci_low"] < 15.5
    assert data_table.loc[0, "ci_high"] > 15.5

def make_interaction_rows(with_interaction):
    """Fausses données 3 régimes × 3 politiques × 5 graines, avec ou sans interaction."""

    rng = np.random.default_rng(0)
    base = {"A": 0, "B": 10, "C": 20}
    regimes = ["s1", "s2", "s3"]
    policies = ["A", "B", "C"]
    seeds = [s for s in range(5)]
    rows = []

    for reg in regimes:
        for p in policies:
            for s in seeds:
                e = base[p]
                if with_interaction and reg == "s2":
                    e = 20 - base[p]
                if with_interaction and reg == "s3":
                    e = 10
                rows.append(
                    {
                        "scenario": reg,
                        "policy": p,
                        "seed": s,
                        "truck_count": 1,
                        "productivity_tph" : 100 + e + rng.normal(0, 1)
                    }
                )
    return pd.DataFrame(rows)

def test_interaction_forte():
    """ Classement inversé selon le régime : interaction très significative."""

    result = interaction_test(make_interaction_rows(True))

    assert result["p_value"] < 0.001
    assert result["df_num"] == 4
    assert result["df_den"] == 24

def test_pas_d_interaction():
    """Même classement dans tous les régimes : pas d'interaction."""

    result = interaction_test(make_interaction_rows(False))
    assert result["p_value"] > 0.05

def test_format_p():
    """ Très petites valeurs → "p < 0,001" ; sinon 3 décimales avec une virgule."""

    assert format_p(3e-41) == "p < 0,001"
    assert format_p(0.0262) == "p = 0,026"
    assert format_p(0.5) == "p = 0,500"

REFERENCES_CSV = Path(__file__).parent.parent / "data" / "results_v1_article" / "benchmark_results.csv"

@pytest.mark.skipif(not REFERENCES_CSV.exists(), reason="Données manquantes")
def test_main_retrouve_article(capsys):
    """Sur le CSV v1, main affiche les chiffres de l'article (Friedman, interaction)."""

    main(REFERENCES_CSV)
    out = capsys.readouterr().out

    assert "χ²(7) = 207,6" in out
    assert "F(14, 609) = 430,7" in out

