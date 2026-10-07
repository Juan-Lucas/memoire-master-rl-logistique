"""Tests statistiques sur benchmark_results.csv.

Les politiques sont traitées comme des groupes indépendants, et le CSV est lu
tel quel, sans retrait des doublons.

- Tests t de Welch unilatéraux (comparaisons par paires)
- ANOVA, Welch-ANOVA ou Kruskal–Wallis selon les prérequis, puis Tukey ou Dunn
- Test de Shapiro-Wilk (normalité par groupe)
- Test de Levene (homogénéité des variances)
- Kruskal-Wallis (alternative non paramétrique si normalité rejetée)
- Cohen's d et Hedges' g (taille d'effet paramétrique)
- Cliff's delta (taille d'effet non paramétrique)
- Intervalles de confiance à 95 % par bootstrap
- Correction de Holm pour comparaisons multiples
"""

from __future__ import annotations

from pathlib import Path

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import kruskal, levene, rankdata, shapiro
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Charger le CSV de résultats dans un DataFrame pandas."""
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Statistiques descriptives
# ---------------------------------------------------------------------------

def print_group_stats(df: pd.DataFrame, scenario: str, policies: list[str]) -> None:
    """Afficher la taille d'échantillon, la moyenne, l'écart-type et l'IC 95 %."""
    print(f"\n=== Statistiques de productivité — scénario {scenario} ===")
    for policy in policies:
        x = df[(df["scenario"] == scenario) & (df["policy"] == policy)]["productivity_tph"].values
        if len(x) == 0:
            continue
        mean = x.mean()
        std = x.std(ddof=1)
        lo, hi = bootstrap_ci(x)
        print(f"{policy:14s}: n={len(x):2d}  mean={mean:.2f}  std={std:.2f}  IC95=[{lo:.2f}, {hi:.2f}]")


# ---------------------------------------------------------------------------
# Normalité et homogénéité des variances (prérequis ANOVA)
# ---------------------------------------------------------------------------

def run_normality_tests(df: pd.DataFrame, scenario: str, policies: list[str]) -> dict[str, bool]:
    """Test de Shapiro-Wilk par groupe.

    Retourne un dict {policy: True si normale (p > 0.05)}.
    """
    print(f"\n=== Shapiro-Wilk (normalité) — scénario {scenario} ===")
    results = {}
    for policy in policies:
        x = df[(df["scenario"] == scenario) & (df["policy"] == policy)]["productivity_tph"].values
        if len(x) < 3:
            continue
        stat, pval = shapiro(x)
        normal = pval > 0.05
        results[policy] = normal
        flag = "✓ normale" if normal else "✗ non normale"
        print(f"  {policy:14s}: W={stat:.4f}  p={pval:.4f}  {flag}")
    return results


def run_levene_test(df: pd.DataFrame, scenario: str, policies: list[str]) -> bool:
    """Test de Levene — homogénéité des variances inter-groupes.

    Retourne True si les variances sont homogènes (p > 0.05).
    """
    groups = [
        df[(df["scenario"] == scenario) & (df["policy"] == p)]["productivity_tph"].values
        for p in policies
        if len(df[(df["scenario"] == scenario) & (df["policy"] == p)]) >= 2
    ]
    if len(groups) < 2:
        return True
    stat, pval = levene(*groups)
    homogeneous = pval > 0.05
    flag = "✓ homogènes" if homogeneous else "✗ hétérogènes"
    print(f"\nLevene — scénario {scenario}: W={stat:.4f}  p={pval:.4f}  {flag}")
    return homogeneous


# ---------------------------------------------------------------------------
# Tests inférentiels
# ---------------------------------------------------------------------------

def run_t_test(df: pd.DataFrame, scenario: str, policy_a: str, policy_b: str) -> None:
    """Test t de Welch unilatéral : policy_a > policy_b."""
    x = df[(df["scenario"] == scenario) & (df["policy"] == policy_a)]["productivity_tph"]
    y = df[(df["scenario"] == scenario) & (df["policy"] == policy_b)]["productivity_tph"]
    tstat, pval = stats.ttest_ind(x, y, equal_var=False, alternative="greater")
    print(
        f"\nWelch {policy_a} > {policy_b} — {scenario}:"
        f"  t={tstat:.4f}  p={pval:.2e}"
        f"  mean_diff={x.mean()-y.mean():.2f} t/h"
    )


def run_welch_anova(groups: list[np.ndarray]) -> tuple[float, float]:
    """ANOVA de Welch à un facteur (variances hétérogènes), calculée à la main."""
    k = len(groups)
    n_i = np.array([len(g) for g in groups])
    m_i = np.array([g.mean() for g in groups])
    v_i = np.array([g.var(ddof=1) for g in groups])

    w_i = n_i / v_i
    W = w_i.sum()
    m_w = (w_i * m_i).sum() / W

    sstr = (w_i * (m_i - m_w) ** 2).sum() / (k - 1)

    lambda_ = (3 / (k ** 2 - 1)) * ((1 - w_i / W) ** 2 / (n_i - 1)).sum()

    F = sstr / (1 + (2 * (k - 2) / (k ** 2 - 1)) * lambda_)
    df2 = (k ** 2 - 1) / (3 * lambda_)

    from scipy.stats import f as f_dist
    pval = 1 - f_dist.cdf(F, k - 1, df2)
    return float(F), float(pval)


def dunn_test(groups: list[np.ndarray], labels: list[str], alpha: float = 0.05) -> None:
    """Test de Dunn post-hoc après Kruskal-Wallis, avec correction de Holm-Bonferroni.

    Implémentation directe basée sur les rangs joints (pas de dépendance externe).
    """
    all_data = np.concatenate(groups)
    N = len(all_data)
    all_ranks = rankdata(all_data)

    # Correction pour les ex aequo
    _, counts = np.unique(all_data, return_counts=True)
    tie_correction = (counts ** 3 - counts).sum() / (12 * (N - 1)) if N > 1 else 0.0

    # Rang moyen par groupe
    idx = 0
    mean_ranks: list[float] = []
    ns: list[int] = []
    for g in groups:
        n = len(g)
        mean_ranks.append(all_ranks[idx:idx + n].mean())
        ns.append(n)
        idx += n

    pairs = list(itertools.combinations(range(len(groups)), 2))
    raw_pvals: list[float] = []
    pair_labels: list[str] = []
    z_stats: list[float] = []

    for i, j in pairs:
        se = np.sqrt((N * (N + 1) / 12 - tie_correction) * (1 / ns[i] + 1 / ns[j]))
        z = abs(mean_ranks[i] - mean_ranks[j]) / se if se > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(z))
        raw_pvals.append(p)
        pair_labels.append(f"{labels[i]} vs {labels[j]}")
        z_stats.append(z)

    reject, pvals_adj, _, _ = multipletests(raw_pvals, method="holm")

    print("\nDunn's test (post-hoc KW) + Holm-Bonferroni correction:")
    for label, z, p_raw, p_adj, rej in zip(pair_labels, z_stats, raw_pvals, pvals_adj, reject):
        flag = "**" if rej else "ns"
        print(f"  {label}: z={z:.3f}  p_raw={p_raw:.3e}  p_adj={p_adj:.3e}  {flag}")


def run_anova(
    df: pd.DataFrame,
    scenario: str,
    use_kruskal: bool = False,
    homogeneous_variances: bool = True,
) -> None:
    """ANOVA classique, Welch-ANOVA ou Kruskal-Wallis, puis un post-hoc.

    Règle de sélection :
    - Normalité rejetée              → Kruskal-Wallis
    - Normalité OK + variances homo  → ANOVA classique (Fisher)
    - Normalité OK + variances hétéro → Welch-ANOVA
    """
    subset = df[df["scenario"] == scenario].copy()

    if use_kruskal:
        label = "Kruskal-Wallis"
    elif not homogeneous_variances:
        label = "Welch-ANOVA"
    else:
        label = "ANOVA"

    print(f"\n=== {label} — scénario {scenario} ===")

    policy_names = list(subset["policy"].unique())
    groups = [subset[subset["policy"] == p]["productivity_tph"].values for p in policy_names]

    if use_kruskal:
        stat, pval = kruskal(*groups)
        print(f"  H={stat:.4f}  p={pval:.2e}")
        # Kruskal–Wallis → Dunn + Holm
        dunn_test(groups, policy_names)
    elif not homogeneous_variances:
        F, pval = run_welch_anova(groups)
        print(f"  F={F:.4f}  p={pval:.2e}  (Welch — variances hétérogènes)")
        # Tukey suppose des variances égales : affiché pour information seulement.
        print("\nPost-hoc Tukey HSD (reference only — pairwise Welch t-tests + Holm are the primary post-hoc):")
        tukey = pairwise_tukeyhsd(subset["productivity_tph"], subset["policy"], alpha=0.05)
        print(tukey)
    else:
        model = ols("productivity_tph ~ C(policy)", data=subset).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
        print("\nPost-hoc Tukey HSD:")
        tukey = pairwise_tukeyhsd(subset["productivity_tph"], subset["policy"], alpha=0.05)
        print(tukey)


def run_holm_correction(pvalues: list[float], labels: list[str]) -> None:
    """Correction de Holm-Bonferroni pour comparaisons multiples."""
    reject, pvals_corrected, _, _ = multipletests(pvalues, method="holm")
    print("\n=== Correction de Holm ===")
    for label, praw, padj, rej in zip(labels, pvalues, pvals_corrected, reject):
        flag = "significatif" if rej else "non significatif"
        print(f"  {label}: p_raw={praw:.2e}  p_adj={padj:.2e}  {flag}")


# ---------------------------------------------------------------------------
# Tailles d'effet
# ---------------------------------------------------------------------------

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d — taille d'effet paramétrique."""
    nx, ny = len(x), len(y)
    if nx + ny <= 2:
        return float("nan")
    pooled = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else float("nan")


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g — correction de Cohen's d pour petits échantillons."""
    d = cohens_d(x, y)
    n = len(x) + len(y)
    correction = 1 - (3 / (4 * n - 9))
    return d * correction


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta — taille d'effet non paramétrique ([-1, 1])."""
    more = sum(xi > yi for xi in x for yi in y)
    less = sum(xi < yi for xi in x for yi in y)
    return (more - less) / (len(x) * len(y))


def _interpret_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "négligeable"
    if ad < 0.5:
        return "petit"
    if ad < 0.8:
        return "moyen"
    return "grand"


def run_effect_sizes(df: pd.DataFrame, scenario: str, policy_a: str, policy_b: str) -> None:
    """Affiche Cohen's d, Hedges' g, Cliff's delta et IC bootstrap pour une paire."""
    x = df[(df["scenario"] == scenario) & (df["policy"] == policy_a)]["productivity_tph"].values
    y = df[(df["scenario"] == scenario) & (df["policy"] == policy_b)]["productivity_tph"].values
    d = cohens_d(x, y)
    g = hedges_g(x, y)
    delta = cliffs_delta(x, y)
    lo_a, hi_a = bootstrap_ci(x)
    lo_b, hi_b = bootstrap_ci(y)
    print(f"\nEffect sizes {policy_a} vs {policy_b} — {scenario}:")
    print(f"  Cohen's d={d:.3f} ({_interpret_d(d)})  Hedges' g={g:.3f}  Cliff's delta={delta:.3f}")
    print(f"  IC 95 % {policy_a}: [{lo_a:.1f}, {hi_a:.1f}] t/h")
    print(f"  IC 95 % {policy_b}: [{lo_b:.1f}, {hi_b:.1f}] t/h")


# ---------------------------------------------------------------------------
# Intervalles de confiance par bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    x: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """IC par bootstrap percentile sur la moyenne."""
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Lance l'analyse complète pour les trois régimes, puis le carburant."""
    base_dir = Path(__file__).resolve().parents[2]
    results_csv = base_dir / "data" / "results" / "benchmark_results.csv"

    if not results_csv.exists():
        print(f"ERREUR : {results_csv} introuvable.")
        return

    df = load_results(results_csv)
    print(f"Résultats chargés : {results_csv}  ({len(df)} lignes)")

    scenarios = ["nominal", "high_load", "high_breakdown"]

    for sc in scenarios:
        print(f"\n\n{'='*80}")
        print(f"  SCÉNARIO : {sc.upper()}")
        print("=" * 80)

        policies = sorted(df[df["scenario"] == sc]["policy"].unique())
        if not policies:
            print("  Aucune donnée pour ce scénario.")
            continue

        # 1. Statistiques descriptives + IC bootstrap
        print_group_stats(df, sc, policies)

        # 2. Prérequis ANOVA
        normality = run_normality_tests(df, sc, policies)
        homogeneous = run_levene_test(df, sc, policies)
        use_kruskal = not all(normality.values())

        # 3. Test global
        run_anova(df, sc, use_kruskal=use_kruskal, homogeneous_variances=homogeneous)

        # 4. Tests par paires + tailles d'effet
        print(f"\n--- Comparaisons par paires (H1) ---")
        pairs = [("PPO", "Fixed"), ("DQN", "Fixed"), ("PPO", "DQN")]
        raw_pvalues, pair_labels = [], []

        for pa, pb in pairs:
            if pa not in policies or pb not in policies:
                continue
            x = df[(df["scenario"] == sc) & (df["policy"] == pa)]["productivity_tph"].values
            y = df[(df["scenario"] == sc) & (df["policy"] == pb)]["productivity_tph"].values
            _, pval = stats.ttest_ind(x, y, equal_var=False, alternative="greater")
            raw_pvalues.append(pval)
            pair_labels.append(f"{pa}>{pb}")
            run_t_test(df, sc, pa, pb)
            run_effect_sizes(df, sc, pa, pb)

        # 5. Correction de Holm
        if raw_pvalues:
            run_holm_correction(raw_pvalues, pair_labels)

    # --- Analyse carburant ---
    print(f"\n\n{'='*80}")
    print("  ANALYSE CARBURANT — specific_fuel_l_per_ton")
    print("=" * 80)

    for sc in scenarios:
        print(f"\n--- Scénario {sc} ---")
        for pa, pb in [("PPO", "Fixed"), ("DQN", "Fixed")]:
            x = df[(df["scenario"] == sc) & (df["policy"] == pa)]["specific_fuel_l_per_ton"].values
            y = df[(df["scenario"] == sc) & (df["policy"] == pb)]["specific_fuel_l_per_ton"].values
            if len(x) == 0 or len(y) == 0:
                continue
            tstat, pval = stats.ttest_ind(x, y, equal_var=False, alternative="less")
            diff_pct = (x.mean() - y.mean()) / y.mean() * 100
            lo_a, hi_a = bootstrap_ci(x)
            lo_b, hi_b = bootstrap_ci(y)
            print(
                f"  Welch {pa} < {pb} (carburant): t={tstat:.3f}  p={pval:.2e}"
                f"  Δ={diff_pct:+.1f} %"
                f"  IC95 {pa}=[{lo_a:.4f},{hi_a:.4f}]  {pb}=[{lo_b:.4f},{hi_b:.4f}]"
            )

    print(f"\n{'='*80}")
    print("FIN DE L'ANALYSE STATISTIQUE")
    print("=" * 80)


def summarise_energy_efficiency(csv_path: str | Path = "data/results/benchmark_results.csv") -> dict[str, float | str]:
    """Compare la consommation spécifique (L/t) de DQN et de l'Affectation fixe.

    Retourne la différence en pourcentage et la valeur p d'un test de Welch
    unilatéral.
    """
    df = load_results(csv_path)
    nominal = df[(df["scenario"] == "nominal") & (df["policy"].isin(["DQN", "Fixed"]))]
    if nominal.empty:
        return {"scenario": "nominal", "policy_a": "DQN", "policy_b": "Fixed", "delta_pct": float("nan"), "p_value": float("nan")}

    dqn = nominal[nominal["policy"] == "DQN"]["specific_fuel_l_per_ton"].values
    fixed = nominal[nominal["policy"] == "Fixed"]["specific_fuel_l_per_ton"].values
    if len(dqn) == 0 or len(fixed) == 0:
        return {"scenario": "nominal", "policy_a": "DQN", "policy_b": "Fixed", "delta_pct": float("nan"), "p_value": float("nan")}

    _, pval = stats.ttest_ind(dqn, fixed, equal_var=False, alternative="less")
    diff_pct = (dqn.mean() - fixed.mean()) / fixed.mean() * 100
    print("\n=== Résumé efficacité énergétique ===")
    print(f"DQN vs Fixed (nominal, L/t): Δ={diff_pct:+.2f}%  p={pval:.3e}")
    print(f"  DQN mean={dqn.mean():.4f}  Fixed mean={fixed.mean():.4f}")

    return {
        "scenario": "nominal",
        "policy_a": "DQN",
        "policy_b": "Fixed",
        "delta_pct": float(diff_pct),
        "p_value": float(pval),
    }


if __name__ == "__main__":
    main()
