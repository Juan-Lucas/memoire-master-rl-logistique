"""Coût du transfert : agents nominaux contre agents spécialisés, par régime (T27, M1).

Pour chaque régime perturbé et chaque agent, on compare sur les **mêmes** graines
d'évaluation l'agent entraîné dans ce régime (« spécialisé ») et l'agent entraîné en
nominal puis évalué sans réentraînement (« transféré »). L'écart transféré − spécialisé
mesure ce que coûte de ne pas réentraîner :
- spécialisé seul (article v1) : capacité d'**adaptation** de l'algorithme, puisqu'on
  réentraîne pour chaque régime ;
- transféré : **robustesse** d'une politique figée face à un régime qu'elle n'a pas vu.

Appariement : (régime, politique, graine d'entraînement, graine d'évaluation). Avec
plusieurs graines d'entraînement, les différences sont d'abord moyennées par graine
d'évaluation, puis l'IC bootstrap rééchantillonne les graines d'évaluation : il ne
contient pas la variabilité entre graines d'entraînement (traitée en T30).

Lancement : python -m memoire_master_rl_logistique.analysis.transfer RESULTATS.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from memoire_master_rl_logistique.analysis.statistical_tests import (
    load_results,
    paired_bootstrap_ci,
)
from memoire_master_rl_logistique.experiments.scenarios import SPECIALISED, TRANSFERRED


def transfer_table(data: pd.DataFrame, metric: str = "productivity_tph") -> pd.DataFrame:
    """Une ligne par (régime, politique) évaluée dans les deux modes.

    Colonnes : ``n_seeds`` (graines d'évaluation appariées), ``specialised``,
    ``transferred`` (moyennes), ``diff`` (transféré − spécialisé), ``ci_low``,
    ``ci_high`` (IC 95 % bootstrap apparié de ``diff``), ``diff_pct`` (en % du spécialisé).
    """
    if "mode" not in data.columns:
        raise ValueError("Pas de colonne « mode » : CSV antérieur à T27, aucun transfert évalué.")
    agents = data[data["mode"].isin([SPECIALISED, TRANSFERRED])]
    keys = ["scenario", "policy", "seed"] + (["train_seed"] if "train_seed" in data else [])
    wide = agents.pivot_table(index=keys, columns="mode", values=metric, aggfunc="first")
    wide = wide.dropna()  # seuls les couples évalués dans les deux modes sont comparés
    if wide.empty:
        return pd.DataFrame(columns=["scenario", "policy", "n_seeds", "specialised",
                                     "transferred", "diff", "ci_low", "ci_high", "diff_pct"])

    # Moyenne sur les graines d'entraînement, par graine d'évaluation.
    per_seed = wide.groupby(level=["scenario", "policy", "seed"]).mean()
    rows = []
    for (scenario, policy), group in per_seed.groupby(level=["scenario", "policy"]):
        diff = group[TRANSFERRED] - group[SPECIALISED]
        low, high = paired_bootstrap_ci(diff.to_numpy())
        specialised = group[SPECIALISED].mean()
        rows.append({
            "scenario": scenario,
            "policy": policy,
            "n_seeds": len(group),
            "specialised": specialised,
            "transferred": group[TRANSFERRED].mean(),
            "diff": diff.mean(),
            "ci_low": low,
            "ci_high": high,
            "diff_pct": 100 * diff.mean() / specialised,
        })
    return pd.DataFrame(rows)


def main(csv_path: str | Path) -> dict[str, pd.DataFrame]:
    """Affiche le coût du transfert pour la productivité et le carburant spécifique."""
    data = load_results(csv_path, mode=None)
    tables = {}
    for metric in ("productivity_tph", "specific_fuel_l_per_ton"):
        tables[metric] = transfer_table(data, metric)
        print(f"\n===== {metric} (transféré − spécialisé) =====")
        print(tables[metric].round(3).to_string(index=False))
    return tables


if __name__ == "__main__":
    main(sys.argv[1])
