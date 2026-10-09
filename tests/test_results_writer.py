import csv
import pytest

from memoire_master_rl_logistique.experiments.run_benchmark import write_results

ROWS = [
    {"scenario": "nominal", "policy": "Fixed", "seed": 42},
    {"scenario": "high_load", "policy": "Nearest", "seed": 43},
    {"scenario": "high_breakdown", "policy": "FIFO", "seed": 44},
]

ROWS_DUPLICATED = ROWS + [{"scenario": "nominal", "policy": "Fixed", "seed": 42}]

ROWS_EMPTY = []

def test_deux_identifiants_differents_deux_fichiers(tmp_path):
    """Teste si deux identifiants différents donnent deux fichiers différents."""

    path_a = write_results(ROWS, tmp_path, "a")
    path_b = write_results(ROWS, tmp_path, "b")

    assert path_a != path_b

    with open(path_a, "r") as file_a:
        reader_a = csv.DictReader(file_a)
        rows_a = list(reader_a)

        assert len(rows_a) == len(ROWS)

        for row in rows_a:
            assert row["run_id"] == "a"

def test_doublon_refuse(tmp_path):
    """Teste l'existence d'un doublon dans la liste."""

    with pytest.raises(ValueError):
        write_results(ROWS_DUPLICATED, tmp_path, "a")

def test_meme_identifiant_refuse(tmp_path):
    """Teste si un identifiant est déjà utilisé."""

    write_results(ROWS, tmp_path, "a")
    with pytest.raises(FileExistsError):
        write_results(ROWS, tmp_path, "a")

def test_liste_vide(tmp_path):
    """Teste le refus d'une liste vide."""

    with pytest.raises(ValueError):
        write_results(ROWS_EMPTY, tmp_path, "a")

    assert list(tmp_path.iterdir()) == []
