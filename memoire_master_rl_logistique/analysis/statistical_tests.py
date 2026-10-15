"""Analyse statistique appariée des résultats de benchmark (§7.6 de l'article).

- chargement validé ;
- Friedman ;
- Wilcoxon apparié bilatéral + Holm ;
- IC bootstrap apparié ;
- d de Cohen ;
- interaction politique × régime ;
- productivité et carburant, avec le même traitement.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS

DYNAMIC_COLUMNS = ["run_id", "inference_time_ms_mean", "inference_time_ms_max"]
SCENARIOS = MAIN_SCENARIOS
PAIRS = [("PPO", "Fixed"), ("DQN", "Fixed"), ("PPO", "DQN")]
METRICS = ["productivity_tph", "specific_fuel_l_per_ton"]

def load_results(csv_path: str | Path, n_expected: int = 30) -> pd.DataFrame:
    """ Charger le CSV de résultats et vérifier qu'il est exploitable
        - Les doublons identiques sont retirés, en ignorant les colonnes de DYNAMIC_COLUMNS
        - Un doublon contradictoire lève ValueError
        - Un groupe (scenario, policy) dont l'effectif est différent de n_expected lève ValueError.
    """

    #1. Charger les données
    data = pd.read_csv(csv_path)

    #2. Retirer les doublons identiques (lignes)
    static_columns = [c for c in list(data.columns) if c not in DYNAMIC_COLUMNS]
    data_cleaned = data.drop_duplicates(subset=static_columns)

    #3. Détecter les doublons contradictoires
    nb_duplicated = data_cleaned.duplicated(subset=["scenario", "policy", "seed"]).sum()

    if nb_duplicated > 0:
        raise ValueError(f"{nb_duplicated} doublons contradictoires ont été détectés.")

    #4. Vérifier les effectifs
    nb_groups = data_cleaned.groupby(["scenario", "policy"]).size()

    mask = nb_groups != n_expected

    if mask.any():
        raise ValueError(f"Les groupes incomplets (attendu: {n_expected}) : {nb_groups[mask].to_dict()}.")

    return data_cleaned.reset_index(drop=True)

def paired_table(data: pd.DataFrame, scenario: str, metric: str) -> pd.DataFrame:
    """ Tableau apparié graine × politique pour un régime et une métrique.
    - Graines en lignes, politiques en colonnes, valeurs de metric, pour le régime demandé
    - ValueError si une graine n'a pas toutes les politiques.
    """

    data = data[data["scenario"] == scenario]

    data_pivot = pd.pivot(data, index="seed", columns="policy", values=metric)

    data_empty = data_pivot.isna()

    if data_empty.any().any():
        bad_seeds = data_pivot.index[data_empty.any(axis=1)].tolist()
        raise ValueError(f"Cases vides : les graines {bad_seeds} n'ont pas toutes les politiques.")

    return data_pivot

def friedman_test(table: pd.DataFrame) -> tuple[float, float]:
    """Test de Friedman sur un tableau graines × politiques ; renvoie (statistique, valeur p)."""

    columns = [table[col] for col in table.columns]

    result = stats.friedmanchisquare(*columns)

    return result.statistic, result.pvalue

def paired_comparisons(table: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Wilcoxon apparié bilatéral, IC bootstrap et correction de Holm pour chaque  paire (a, b). """
    rows = []

    for a, b in pairs:
        diff = table[a] - table[b]
        low, high = paired_bootstrap_ci(diff)
        res = stats.wilcoxon(table[a], table[b])
        rows.append(
            {
                "pair": f"{a}-{b}",
                "mean_diff": np.mean(diff),
                "p_value": res.pvalue,
                "ci_low": low,
                "ci_high": high
            }
        )

    result = pd.DataFrame(rows)

    result["p_holm"] = multipletests(result["p_value"], method="holm")[1]

    return result

def paired_bootstrap_ci(diff, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """IC à 95 % de la moyenne des différences appariées, par bootstrap (rééchantillonnage des graines)"""

    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(n_boot, len(diff)),
        replace=True
    )

    means = np.mean(samples, axis=1)

    low, high = np.percentile(means, [2.5, 97.5])

    return float(low), float(high)

def interaction_test(df, metric="productivity_tph"):
    """Interaction politique × régime : modèle complet contre modèle sans interaction (test F)."""

    data = df.copy()

    data["y"] = data[metric] / data["truck_count"]
    data["bloc"] = data["scenario"] + "_" + data["seed"].astype(str)

    full = ols("y ~ C(bloc) + C(policy) + C(policy):C(scenario)", data=data).fit()
    reduced = ols("y ~ C(bloc) + C(policy)", data=data).fit()

    row = sm.stats.anova_lm(reduced, full).iloc[1]

    return {
        "F": float(row["F"]),
        "df_num": int(row["df_diff"]),
        "df_den": int(row["df_resid"]),
        "p_value": float(row["Pr(>F)"]),
        "eta_p2": float(row["ss_diff"] / (row["ss_diff"] + row["ssr"]))
    }

def format_p(p: float) -> str:
    """Valeur p au format de l'article : « p < 0,001 » ou « p = 0,026 »."""

    if p < 0.001:
        return "p < 0,001"
    else:
        return f"p = {p:.3f}".replace(".", ",")

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d — taille d'effet paramétrique."""
    nx, ny = len(x), len(y)
    if nx + ny <= 2:
        return float("nan")
    pooled = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else float("nan")

def main(csv_path: str | Path) -> None:
    """Analyse appariée complète : Friedman, Wilcoxon + Holm, IC bootstrap, d, puis interaction."""

    df = load_results(csv_path)

    for metric in METRICS:
        print(f"\n===== {metric} =====")
        for scenario in SCENARIOS:
            table = paired_table(df, scenario, metric)
            stat, p = friedman_test(table)
            k = len(table.columns)
            line = f"{scenario} — Friedman : χ²({k - 1}) = {stat:.1f}, {format_p(p)}"
            print(line.replace(".", ","))
            comp = paired_comparisons(table, PAIRS)
            comp["d"] = [cohens_d(table[a], table[b]) for a, b in PAIRS]
            print(comp.to_string(index=False))

    inter = interaction_test(df)
    line = (
        f"\nInteraction politique × régime : F({inter['df_num']}, {inter['df_den']}) = "
        f"{inter['F']:.1f}, {format_p(inter['p_value'])}, η²p = {inter['eta_p2']:.2f}"
    )

    print(line.replace(".", ","))

if __name__ == "__main__":
    main(sys.argv[1])
