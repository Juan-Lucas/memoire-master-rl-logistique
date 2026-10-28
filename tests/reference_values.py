"""Accès aux valeurs de référence de la simulation (``fixtures/regression_reference.csv``).

Partagé par le test de non-régression et par les tests qui vérifient qu'une
fonctionnalité (journal des cycles, borne au poste, chronométrage…) ne change pas
la simulation.
"""

import csv
from pathlib import Path

REFERENCE_CSV = Path(__file__).parent / "fixtures" / "regression_reference.csv"


def reference(policy: str, seed: int, scenario: str = "nominal") -> dict[str, float]:
    """KPI de référence d'un épisode : ``productivity_tph``, ``total_reward``, ``total_wait_min``."""
    with REFERENCE_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["scenario"], row["policy"], int(row["seed"])) == (scenario, policy, seed):
                return {k: float(v) for k, v in row.items() if k not in ("scenario", "policy", "seed")}
    raise KeyError(f"Aucune référence pour {scenario}, {policy}, {seed}")
