"""Tests statistiques pour les résultats de benchmark.

Ce module reproduit l'analyse statistique décrite au chapitre 6 du mémoire.
Il réalise :

- Tests t de Welch (comparaisons par paires)
- ANOVA à un facteur sur les 8 politiques, pour chacun des 3 scénarios
- Test post-hoc de Tukey HSD pour identifier les paires significativement
    différentes

Les choix méthodologiques suivent la règle de reproductibilité du projet :
le fichier d'entrée est `data/results/benchmark_results.csv` et le script
doit être exécuté dans l'environnement conda `datascience` utilisé pour
les expériences.
"""

from __future__ import annotations

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
    """Point d'entrée : automatisation complète de l'analyse du Chapitre 6.
    
    Réalise les tests de significativité pour les 3 scénarios du mémoire
    et compare les agents RL (PPO/DQN) aux baselines industrielles (Fixed).
    """
    # Chemin relatif robuste vers les résultats
    base_dir = Path(__file__).resolve().parents[2]
    results_csv = base_dir / "data" / "results" / "benchmark_results.csv"

    if not results_csv.exists():
        print(f"ERREUR : Le fichier {results_csv} est introuvable.")
        return

    df = load_results(results_csv)
    print(f"Analyse des résultats : {results_csv}")
    print(f"Nombre de réplications détectées : {len(df)}")

    # Liste des scénarios officiels du mémoire
    scenarios = ["nominal", "high_load", "high_breakdown"]
    
    for sc in scenarios:
        print(f"\n\n" + "="*80)
        print(f" ÉTUDE STATISTIQUE DU SCÉNARIO : {sc.upper()}")
        print("="*80)

        # 1. Statistiques descriptives
        available_policies = df[df["scenario"] == sc]["policy"].unique()
        print_group_stats(df, sc, list(available_policies))

        # 2. ANOVA Globale (Pour prouver que le choix de l'algo compte)
        run_anova(df, sc)

        # 3. Tests t de comparaison (Les preuves de l'hypothèse H1)
        print(f"\n--- Tests de supériorité (Hypothèse H1) ---")
        
        if "PPO" in available_policies and "Fixed" in available_policies:
            run_t_test(df, sc, "PPO", "Fixed")
            
        if "DQN" in available_policies and "Fixed" in available_policies:
            run_t_test(df, sc, "DQN", "Fixed")

    print("\n" + "="*80)
    print("FIN DE L'ANALYSE STATISTIQUE")
    print("="*80)

if __name__ == "__main__":
    main()