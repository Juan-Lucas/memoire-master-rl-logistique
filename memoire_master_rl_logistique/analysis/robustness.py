"""Robustesse : chaque régime perturbé comparé au régime nominal (tableau 13 de l'article).

- Ratio de robustesse = productivité **par camion** dans le régime / en nominal.
  En forte charge (18 camions au lieu de 12), un ratio sur la productivité totale
  mesurerait surtout l'ajout de 50 % de camions : une politique inchangée
  afficherait déjà 1,5. En fortes pannes, le nombre de camions est le même et les
  deux calculs coïncident.
- Variation de l'attente par camion (en %) par rapport au nominal. Depuis T10, c'est
  l'**attente en file** (``queue_wait_min_per_truck``) ; pour un CSV antérieur à T10
  (comme celui de l'article v1), seule l'immobilisation totale existe
  (``avg_wait_min_per_truck`` : file + réparations + ATTENDRE) : voir ``wait_column``.

Calculs seulement : la figure et la ligne de commande sont dans ``plots/robustness``.
"""

from __future__ import annotations

import pandas as pd

from memoire_master_rl_logistique.experiments.scenarios import PERTURBED_SCENARIOS

PRODUCTIVITY = "productivity_tph"
QUEUE_WAIT = "queue_wait_min_per_truck"  # depuis T10
TOTAL_WAIT = "avg_wait_min_per_truck"  # file + réparations + ATTENDRE (seule colonne en v1)


def wait_column(df: pd.DataFrame) -> str:
    """Colonne d'attente à utiliser : l'attente en file si elle existe, sinon l'immobilisation totale."""
    return QUEUE_WAIT if QUEUE_WAIT in df.columns else TOTAL_WAIT


def _scenario_means(df: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """Moyenne de ``values`` par politique (lignes) et par régime (colonnes).

    Lève ``ValueError`` si le régime nominal ou un régime perturbé manque.
    """
    means = values.groupby([df["policy"], df["scenario"]]).mean().unstack("scenario")
    missing = [s for s in ["nominal", *PERTURBED_SCENARIOS] if s not in means.columns]
    if missing:
        raise ValueError(f"Régimes absents des données : {missing}")
    return means


def robustness_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio de robustesse par politique : productivité par camion du régime / nominal.

    Renvoie un tableau politiques × régimes perturbés ; 1 = aucune perte.
    """
    per_truck = df[PRODUCTIVITY] / df["truck_count"]
    means = _scenario_means(df, per_truck)
    return means[PERTURBED_SCENARIOS].div(means["nominal"], axis=0)


def wait_change_pct(df: pd.DataFrame, wait_col: str | None = None) -> pd.DataFrame:
    """Variation (%) de l'attente par camion par rapport au nominal.

    ``wait_col`` : par défaut ``wait_column(df)``. Positive = l'attente augmente
    (dégradation). Tableau politiques × régimes perturbés.
    """
    wait_col = wait_col or wait_column(df)
    means = _scenario_means(df, df[wait_col])
    nominal = means["nominal"]
    return means[PERTURBED_SCENARIOS].sub(nominal, axis=0).div(nominal, axis=0) * 100


def robustness_table(ratios: pd.DataFrame, wait_pct: pd.DataFrame) -> pd.DataFrame:
    """Assemble ratios (``ratio_<régime>``) et variations d'attente (``wait_pct_<régime>``)."""
    return ratios.add_prefix("ratio_").join(wait_pct.add_prefix("wait_pct_"))
