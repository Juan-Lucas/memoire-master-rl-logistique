"""Analyse de robustesse inter-scénarios (section 5.8.3 du plan Q1).

Calcule le Robustness Ratio pour chaque politique et chaque scénario :

    RR(p, s) = Performance(p, s) / Performance(p, nominal)

Un ratio proche de 1.0 indique que la politique conserve ses performances
nominales sous perturbation. Un ratio faible indique une dégradation.

Produit :
- Un tableau CSV `data/results/robustness_ratios.csv`
- Un affichage console du tableau et du Δ_breakdown
- Une heatmap (Figure 6 de l'article)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Calcul des ratios
# ---------------------------------------------------------------------------

def compute_robustness_ratios(
    csv_path: str | Path,
    metric: str = "productivity_tph",
) -> pd.DataFrame:
    """Calcule RR(p, s) = mean(p, s) / mean(p, nominal) pour chaque politique.

    Paramètres
    ----------
    csv_path : chemin vers benchmark_results.csv
    metric   : colonne KPI à utiliser (défaut : productivité)

    Retour
    ------
    DataFrame avec colonnes [policy, scenario, mean, std, robustness_ratio,
                              delta_pct, ic95_lo, ic95_hi]
    """
    df = pd.read_csv(csv_path)
    df = df[df["scenario"].isin(["nominal", "high_load", "high_breakdown"])]

    # Moyenne et écart-type par politique × scénario
    agg = (
        df.groupby(["policy", "scenario"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_kpi", "std": "std_kpi", "count": "n"})
    )

    # IC 95 % par bootstrap
    def _bootstrap_ci(series: pd.Series, n_boot: int = 10_000) -> tuple[float, float]:
        rng = np.random.default_rng(42)
        vals = series.values
        means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    ci_rows = []
    for (policy, scenario), grp in df.groupby(["policy", "scenario"]):
        lo, hi = _bootstrap_ci(grp[metric])
        ci_rows.append({"policy": policy, "scenario": scenario, "ic95_lo": lo, "ic95_hi": hi})
    ci_df = pd.DataFrame(ci_rows)

    agg = agg.merge(ci_df, on=["policy", "scenario"])

    # Robustness Ratio
    nominal = agg[agg["scenario"] == "nominal"][["policy", "mean_kpi"]].rename(
        columns={"mean_kpi": "mean_nominal"}
    )
    agg = agg.merge(nominal, on="policy")
    agg["robustness_ratio"] = agg["mean_kpi"] / agg["mean_nominal"]

    # Δ_breakdown = (nominal − s) / nominal × 100
    agg["delta_pct"] = (1 - agg["robustness_ratio"]) * 100

    return agg.sort_values(["scenario", "robustness_ratio"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def print_robustness_table(df: pd.DataFrame) -> None:
    """Affiche le tableau de robustesse dans la console."""
    print("\n=== Robustness Ratio RR(p,s) = Performance(p,s) / Performance(p,nominal) ===\n")
    cols = ["policy", "scenario", "mean_kpi", "std_kpi", "robustness_ratio", "delta_pct"]
    fmt = {
        "mean_kpi": "{:.1f}",
        "std_kpi": "{:.1f}",
        "robustness_ratio": "{:.4f}",
        "delta_pct": "{:+.2f} %",
    }
    for sc in ["nominal", "high_load", "high_breakdown"]:
        sub = df[df["scenario"] == sc][cols].reset_index(drop=True)
        print(f"--- {sc.upper()} ---")
        for _, row in sub.iterrows():
            print(
                f"  {row['policy']:14s}  mean={row['mean_kpi']:7.1f}  "
                f"RR={row['robustness_ratio']:.4f}  Δ={row['delta_pct']:+.2f} %"
            )
        print()


def plot_robustness_heatmap(
    df: pd.DataFrame,
    results_csv: str | Path | None = None,
    output_path: str | Path | None = None,
) -> None:
    """Two-block degradation heatmap (Figure 7).

    Top block  : productivity signed % change vs nominal = (RR-1)*100
                 positive (blue) = improvement, negative (red) = degradation
    Bottom block: waiting-time signed % change vs nominal = -Δ_wait
                 positive (blue) = less waiting (improvement), negative (red) = more waiting
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib / seaborn required for the heatmap.")
        return

    POLICY_ORDER = [
        "FIFO", "Fixed", "Nearest", "ShortestPath",
        "Q-Learning", "SARSA", "DQN", "PPO",
    ]
    COL_ORDER = ["nominal", "high_load", "high_breakdown"]
    COL_LABELS = {"nominal": "Nominal", "high_load": "High-load", "high_breakdown": "High-breakdown"}

    # --- Top block: (RR - 1) * 100 = productivity % change vs nominal ---
    pivot_prod = df.pivot_table(
        index="policy", columns="scenario", values="robustness_ratio", aggfunc="mean",
    )
    col_order = [c for c in COL_ORDER if c in pivot_prod.columns]
    pivot_prod = pivot_prod[col_order]
    pivot_delta_prod = (pivot_prod - 1.0) * 100.0

    # --- Bottom block: waiting-time % change vs nominal (negated so blue = better) ---
    if results_csv is not None:
        raw = pd.read_csv(results_csv)
        raw = raw[raw["scenario"].isin(COL_ORDER)]
        wait_means = (
            raw.groupby(["policy", "scenario"])["avg_wait_min_per_truck"]
            .mean()
            .unstack("scenario")
        )
        wait_nominal = wait_means["nominal"]
        delta_wait_pct = wait_means.copy()
        for sc in col_order:
            delta_wait_pct[sc] = (wait_means[sc] - wait_nominal) / wait_nominal * 100.0
        # Negate: positive = less waiting = improvement = blue
        pivot_delta_wait = -delta_wait_pct[col_order]
        has_wait = True
    else:
        has_wait = False

    # Align row order
    shared_policies = [p for p in POLICY_ORDER if p in pivot_delta_prod.index]
    pivot_delta_prod = pivot_delta_prod.loc[shared_policies]

    col_labels = [COL_LABELS.get(c, c) for c in col_order]

    if has_wait:
        pivot_delta_wait = pivot_delta_wait.loc[shared_policies]
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7, 9),
            gridspec_kw={"hspace": 0.35},
        )
    else:
        fig, ax_top = plt.subplots(figsize=(7, 5))

    # Shared vmin/vmax for symmetric diverging palette
    vabs = max(abs(pivot_delta_prod.values).max(), 20)
    sns.heatmap(
        pivot_delta_prod,
        annot=True, fmt=".1f",
        cmap="RdBu", center=0, vmin=-vabs, vmax=vabs,
        linewidths=0.5, ax=ax_top,
        xticklabels=col_labels,
        cbar_kws={"label": "Δ productivity (%)"},
    )
    ax_top.set_title("Top: Productivity change vs nominal (%)\n"
                     "(blue = improvement, red = degradation)", fontsize=10)
    ax_top.set_xlabel("")
    ax_top.set_ylabel("Policy")

    if has_wait:
        vabs_w = max(abs(pivot_delta_wait.values).max(), 20)
        sns.heatmap(
            pivot_delta_wait,
            annot=True, fmt=".1f",
            cmap="RdBu", center=0, vmin=-vabs_w, vmax=vabs_w,
            linewidths=0.5, ax=ax_bot,
            xticklabels=col_labels,
            cbar_kws={"label": "Δ waiting time (%, negated)"},
        )
        ax_bot.set_title("Bottom: Waiting-time change vs nominal (negated, %)\n"
                         "(blue = less waiting, red = more waiting)", fontsize=10)
        ax_bot.set_xlabel("Scenario")
        ax_bot.set_ylabel("Policy")

    plt.suptitle("Two-dimensional robustness heatmap — all eight policies",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved: {output_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main(
    results_csv: str = "data/results/benchmark_results.csv",
    output_csv: str = "data/results/robustness_ratios.csv",
    heatmap_path: str = "reports/figures/robustness_heatmap.png",
) -> None:
    csv_path = Path(results_csv)
    if not csv_path.exists():
        print(f"ERREUR : {csv_path} introuvable.")
        return

    df = compute_robustness_ratios(csv_path)
    print_robustness_table(df)

    df.to_csv(output_csv, index=False)
    print(f"Ratios sauvegardés : {output_csv}")

    plot_robustness_heatmap(df, results_csv=results_csv, output_path=heatmap_path)


if __name__ == "__main__":
    main()
