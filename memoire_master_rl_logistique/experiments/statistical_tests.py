from __future__ import annotations

"""Tests statistiques pour les résultats de benchmark.

Ce module reproduit l'analyse statistique décrite au chapitre 6 du mémoire.
Il réalise :

- Tests t de Welch (comparaisons par paires)
- ANOVA à un facteur sur les 8 politiques (scénario = nominal)
- Test post-hoc de Tukey HSD pour identifier les paires significativement
    différentes

Les choix méthodologiques suivent la règle de reproductibilité du projet :
le fichier d'entrée est `data/results/benchmark_results.csv` et le script
doit être exécuté dans l'environnement conda `datascience` utilisé pour
les expériences.
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Charger le CSV de résultats dans un DataFrame pandas.

    Paramètres
    ----------
    csv_path : str | Path
        Chemin vers `benchmark_results.csv`.

    Retour
    ------
    pd.DataFrame
        DataFrame contenant une ligne par réplication (policy, scenario, seed).
    """
    df = pd.read_csv(csv_path)
    return df


def print_group_stats(df: pd.DataFrame, scenario: str, policies: list[str]) -> None:
    """Afficher la taille d'échantillon, la moyenne et l'écart-type.

    Résumé compact utilisé pour vérifier rapidement les statistiques avant
    les tests inférentiels.
    """
    print(f"\n=== Statistiques de productivité — scénario {scenario} ===")
    for policy in policies:
        subset = df[(df["scenario"] == scenario) & (df["policy"] == policy)]
        n = len(subset)
        mean = subset["productivity_tph"].mean()
        std = subset["productivity_tph"].std(ddof=1)
        print(f"{policy:12s}: n={n}, mean={mean:.2f}, std={std:.2f}")


def run_t_test(df: pd.DataFrame, scenario: str, policy_a: str, policy_b: str) -> None:
    """Effectuer le test t de Welch (unilatéral) et afficher t et p.

    On utilise `scipy.stats.ttest_ind` avec `equal_var=False` (Welch) et
    `alternative='greater'` pour tester si la moyenne de `policy_a` est
    supérieure à celle de `policy_b`.
    """
    x = df[(df["scenario"] == scenario) & (df["policy"] == policy_a)]["productivity_tph"]
    y = df[(df["scenario"] == scenario) & (df["policy"] == policy_b)]["productivity_tph"]

    tstat, pvalue = stats.ttest_ind(x, y, equal_var=False, alternative="greater")

    print(
        f"\nTest t (Welch) {policy_a} > {policy_b} — scénario {scenario}:"
        f" t={tstat:.4f}, p={pvalue:.6f}, mean_{policy_a}={x.mean():.2f}, mean_{policy_b}={y.mean():.2f}"
    )


def run_anova(df: pd.DataFrame, scenario: str) -> None:
    """Réaliser une ANOVA à sens unique et un test post-hoc de Tukey.

    L'ANOVA vérifie si au moins une moyenne de politique diffère des autres.
    Si le résultat est significatif, Tukey HSD permet d'identifier les
    paires qui diffèrent tout en contrôlant l'erreur de famille.
    """
    subset = df[df["scenario"] == scenario].copy()
    print(f"\n=== ANOVA sur les 8 méthodes — scénario {scenario} ===")

    # Ajuster un modèle linéaire ordinaire : productivity ~ policy
    model = ols("productivity_tph ~ C(policy)", data=subset).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # Comparaisons post-hoc (Tukey HSD)
    print("\nPost-hoc Tukey HSD:")
    tukey = pairwise_tukeyhsd(subset["productivity_tph"], subset["policy"], alpha=0.05)
    print(tukey)


def main() -> None:
    """Point d'entrée : charger les données et lancer les tests.

    Le flux courant reproduit les analyses du chapitre 6 :
    - PPO vs Fixed (nominal)
    - DQN vs Fixed (high_breakdown)
    - ANOVA + Tukey sur le scénario nominal
    """
    results_csv = Path(__file__).resolve().parents[2] / "data" / "results" / "benchmark_results.csv"

    # Charger les résultats et afficher la source pour traçabilité
    df = load_results(results_csv)
    print("Fichier de résultats:", results_csv)

    # Tests par paires (afficher résumé puis effectuer le test)
    print_group_stats(df, "nominal", ["PPO", "Fixed"])
    run_t_test(df, "nominal", "PPO", "Fixed")

    print_group_stats(df, "high_breakdown", ["DQN", "Fixed"])
    run_t_test(df, "high_breakdown", "DQN", "Fixed")

    # ANOVA sur toutes les politiques pour le scénario nominal
    run_anova(df, "nominal")


if __name__ == "__main__":
    main()
