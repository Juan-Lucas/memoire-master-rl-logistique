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


def test_colonnes_absentes_de_la_premiere_ligne(tmp_path):
    """Les colonnes propres aux agents tabulaires (T20) sont écrites même si la première
    ligne (une heuristique) ne les a pas ; les cases manquantes restent vides."""
    rows = [
        {"scenario": "nominal", "policy": "Fixed", "train_seed": None, "seed": 42},
        {"scenario": "nominal", "policy": "SARSA", "train_seed": 1, "seed": 42,
         "known_state_pct": 100.0},
    ]
    path = write_results(rows, tmp_path, run_id="mix")

    with open(path, newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert written[0]["known_state_pct"] == ""
    assert written[1]["known_state_pct"] == "100.0"


def test_meme_politique_deux_graines_d_entrainement(tmp_path):
    """T26 : deux modèles (graines d'entraînement 1 et 2) sur la même graine d'évaluation
    ne sont pas un doublon ; le même modèle deux fois en est un."""
    row = {"scenario": "nominal", "policy": "PPO", "seed": 42}
    write_results([{**row, "train_seed": 1}, {**row, "train_seed": 2}], tmp_path, run_id="a")
    with pytest.raises(ValueError, match="Doublon"):
        write_results([{**row, "train_seed": 1}, {**row, "train_seed": 1}], tmp_path, run_id="b")
