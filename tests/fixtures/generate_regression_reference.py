"""Génère ``regression_reference.csv``, la référence du test de non-régression (T4).

Historique :
- T4 à T12 bis : la référence était le CSV de l'article v1 (``data/results_v1_article``).
- T13 (flux aléatoires par processus) change l'ordre des tirages : tous les chiffres
  bougent, comme prévu par le plan. La référence est alors figée ici, à partir du code
  de T13, et ce fichier est suivi par Git.
- T15 et T16 bis (moteur à événements, décision à la verse) : changement voulu de la
  simulation, référence régénérée. Fixed et FIFO en nominal, graines 42 à 46 :
  productivité moyenne 4 084,5 → 4 273,5 t/h (Fixed) et 3 293,5 → 4 263,0 t/h (FIFO).
- Option B (20 octobre 2026, coût d'opportunité de chaque cycle) : productivités et
  attentes identiques ; ``total_reward`` baisse exactement de 12 × 480 / 23,2 = 248,276
  sur chaque poste (vérifié avant de régénérer).

À relancer seulement quand un changement **voulu** de la simulation modifie les
résultats, en notant pourquoi dans le journal :
    python tests/fixtures/generate_regression_reference.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS

REFERENCE_CSV = Path(__file__).with_name("regression_reference.csv")
POLICIES = ("Fixed", "FIFO")
SEEDS = (42, 43, 44, 45, 46)
COLUMNS = ("productivity_tph", "total_reward", "total_wait_min")


def generate() -> list[dict]:
    """Rejoue Fixed et FIFO en nominal sur les graines 42 à 46."""
    scenario = SCENARIOS["nominal"]
    policies = heuristic_policies(scenario)
    rows = []
    for policy in POLICIES:
        for seed in SEEDS:
            kpis = run_single_episode(make_env(scenario), policies[policy], seed)
            rows.append({"scenario": "nominal", "policy": policy, "seed": seed,
                         **{c: repr(kpis[c]) for c in COLUMNS}})
    return rows


if __name__ == "__main__":
    rows = generate()
    with REFERENCE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Écrit : {REFERENCE_CSV} ({len(rows)} lignes)")
