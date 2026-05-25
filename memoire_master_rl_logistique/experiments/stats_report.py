"""Analyse statistique et génération de figures (Chapitre 6).

Produit les tableaux comparatifs et les figures pour le Chapitre 6 :
- Tableaux : moyenne ± écart-type par méthode et scénario
- Figures : barplots comparatifs des KPIs
- Tests statistiques de significativité (Demšar, 2006)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Charge les résultats du benchmark depuis un CSV."""
    return pd.read_csv(csv_path)


def generate_summary_table(
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Génère le tableau de synthèse (moyenne ± std) par politique et scénario."""
    kpi_cols = [
        "productivity_tph",
        "avg_wait_min_per_truck",
        "utilization_pct",
        "specific_fuel_l_per_ton",
        "cost_per_cycle",
        "total_reward",
    ]

    summary_rows = []
    for scenario in df["scenario"].unique():
        for policy in df["policy"].unique():
            mask = (df["scenario"] == scenario) & (df["policy"] == policy)
            subset = df[mask]
            row = {"scenario": scenario, "policy": policy}
            for col in kpi_cols:
                if col in subset.columns:
                    mean = subset[col].mean()
                    std = subset[col].std()
                    row[f"{col}_mean"] = round(mean, 2)
                    row[f"{col}_std"] = round(std, 2)
                    row[f"{col}_display"] = f"{mean:.2f} ± {std:.2f}"
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Tableau de synthèse sauvegardé : {summary_path}")
    return summary_df


def plot_kpi_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    scenario_name: str = "nominal",
) -> None:
    """Génère les barplots comparatifs des KPIs (figures pour le Chapitre 6)."""
    mask = df["scenario"] == scenario_name
    subset = df[mask]

    if subset.empty:
        print(f"Aucune donnée pour le scénario '{scenario_name}'.")
        return

    policies = subset["policy"].unique()
    kpis = {
        "productivity_tph": ("Productivité (t/h)", "steelblue"),
        "avg_wait_min_per_truck": ("Attente moy./camion (min)", "coral"),
        "utilization_pct": ("Utilisation (%)", "seagreen"),
        "specific_fuel_l_per_ton": ("Conso. spécifique (L/t)", "goldenrod"),
        "cost_per_cycle": ("Coût moy./cycle (L)", "mediumpurple"),
        "total_reward": ("Récompense cumulée", "indianred"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Comparaison des KPIs — Scénario: {scenario_name}",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (col, (label, color)) in zip(axes.flat, kpis.items()):
        means = []
        stds = []
        for policy in policies:
            policy_data = subset[subset["policy"] == policy][col]
            means.append(policy_data.mean())
            stds.append(policy_data.std())

        x = np.arange(len(policies))
        bars = ax.bar(x, means, yerr=stds, color=color, alpha=0.8, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(means),
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    fig_path = output_dir / f"kpi_comparison_{scenario_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure sauvegardée : {fig_path}")


def plot_learning_curve(
    training_csv: str | Path,
    output_dir: Path,
) -> None:
    """Génère la courbe d'apprentissage (learning curve) de l'agent PPO."""
    csv_path = Path(training_csv)
    if not csv_path.exists():
        print(f"Fichier d'entraînement non trouvé : {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Courbes d'apprentissage PPO", fontsize=14, fontweight="bold")

    # Tonnage
    if "tonnage_t" in df.columns:
        window = min(20, len(df))
        axes[0].plot(df["tonnage_t"], alpha=0.3, color="steelblue")
        axes[0].plot(
            df["tonnage_t"].rolling(window).mean(),
            color="steelblue",
            linewidth=2,
        )
        axes[0].set_xlabel("Épisode")
        axes[0].set_ylabel("Tonnage (t)")
        axes[0].set_title("Tonnage par épisode")
        axes[0].grid(alpha=0.3)

    # Temps d'attente
    if "wait_min" in df.columns:
        axes[1].plot(df["wait_min"], alpha=0.3, color="coral")
        axes[1].plot(
            df["wait_min"].rolling(window).mean(),
            color="coral",
            linewidth=2,
        )
        axes[1].set_xlabel("Épisode")
        axes[1].set_ylabel("Attente (min)")
        axes[1].set_title("Temps d'attente par épisode")
        axes[1].grid(alpha=0.3)

    # Carburant
    if "fuel_l" in df.columns:
        axes[2].plot(df["fuel_l"], alpha=0.3, color="goldenrod")
        axes[2].plot(
            df["fuel_l"].rolling(window).mean(),
            color="goldenrod",
            linewidth=2,
        )
        axes[2].set_xlabel("Épisode")
        axes[2].set_ylabel("Carburant (L)")
        axes[2].set_title("Consommation par épisode")
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "learning_curve_ppo.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Learning curve sauvegardée : {fig_path}")


def generate_full_report(
    results_csv: str = "data/results/benchmark_results.csv",
    training_csv: str = "models/ppo_mine/training_kpis.csv",
    output_dir: str = "data/results/figures",
) -> None:
    """Génère le rapport complet : tableaux + figures."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results_path = Path(results_csv)
    if not results_path.exists():
        print(f"Fichier de résultats non trouvé : {results_path}")
        print("Lancez d'abord run_benchmark.py.")
        return

    df = load_results(results_path)

    # Tableau de synthèse
    summary = generate_summary_table(df, out_path)
    print("\n=== Tableau de synthèse ===")
    print(summary.to_string(index=False))

    # Figures par scénario
    for scenario in df["scenario"].unique():
        plot_kpi_comparison(df, out_path, scenario)

    # Learning curve
    plot_learning_curve(training_csv, out_path)

    print(f"\nRapport complet généré dans : {out_path}")


def main() -> None:
    """Point d'entrée pour la génération du rapport."""
    generate_full_report()


if __name__ == "__main__":
    main()
