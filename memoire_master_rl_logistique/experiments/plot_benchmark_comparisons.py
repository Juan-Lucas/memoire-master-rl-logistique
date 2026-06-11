"""Comparaisons de benchmark (productivité et temps d'attente) entre politiques.

Source : data/results/benchmark_results.csv
Sortie :
- reports/figures/barplot_productivite_scenarios.png
- reports/figures/wait_time_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCENARIO_COLORS = {
    "high_breakdown": "#64B5F6",
    "high_load": "#1A237E",
    "nominal": "#FF7043",
}
SCENARIO_LABELS = {
    "high_breakdown": "high_breakdown",
    "high_load": "high_load",
    "nominal": "nominal",
}

ALGO_ORDER = ["DQN", "Fixed", "PPO", "Q-Learning", "ShortestPath", "Nearest", "FIFO", "SARSA"]

WAIT_POLICY_COLORS = {
    "DQN": "#2196F3",
    "Fixed": "#1A237E",
    "PPO": "#FF5722",
}


def plot_productivity(df: pd.DataFrame, out_dir: Path) -> None:
    means = (
        df.groupby(["policy", "scenario"])["productivity_tph"]
        .mean()
        .unstack("scenario")
        .loc[ALGO_ORDER, list(SCENARIO_COLORS)]
    )

    n_algos = len(ALGO_ORDER)
    n_scenarios = len(SCENARIO_COLORS)
    x = np.arange(n_algos)
    width = 0.8 / n_scenarios

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, scenario in enumerate(SCENARIO_COLORS):
        offsets = x + (i - (n_scenarios - 1) / 2) * width
        values = means[scenario].to_numpy()
        bars = ax.bar(offsets, values, width, label=SCENARIO_LABELS[scenario], color=SCENARIO_COLORS[scenario])
        for bar, val in zip(bars, values):
            label = f"{val / 1000:.1f}K".replace(".", ",")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, label,
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_ORDER)
    ax.set_xlabel("Algorithme")
    ax.set_ylabel("Productivité (t/h)")
    ax.set_title("Productivité comparative (t/h) — 3 scénarios")
    ax.legend(title="scenario")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "barplot_productivite_scenarios.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")


def plot_wait_time(df: pd.DataFrame, out_dir: Path) -> None:
    scenarios = list(SCENARIO_COLORS)
    policies = list(WAIT_POLICY_COLORS)

    means = (
        df.groupby(["policy", "scenario"])["avg_wait_min_per_truck"]
        .mean()
        .unstack("scenario")
        .loc[policies, scenarios]
    )

    n_scenarios = len(scenarios)
    n_policies = len(policies)
    x = np.arange(n_scenarios)
    width = 0.8 / n_policies

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, policy in enumerate(policies):
        offsets = x + (i - (n_policies - 1) / 2) * width
        values = means.loc[policy].to_numpy()
        bars = ax.bar(offsets, values, width, label=policy, color=WAIT_POLICY_COLORS[policy])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.0f}",
                    ha="center", va="bottom", fontsize=9)

    ax.axhline(30, color="#42A5F5", linestyle="--", linewidth=1.5)
    ax.text(-0.45, 31, "Seuil cible : 30 min", color="#1565C0", ha="left", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_xlabel("Scénario")
    ax.set_ylabel("Temps d'attente moyen (min)")
    ax.set_title("Temps d'attente moyen (min) — Fixed / PPO / DQN")
    ax.legend(title="policy")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "wait_time_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")


def main() -> None:
    df = pd.read_csv("data/results/benchmark_results.csv")
    df = df[df["scenario"].isin(SCENARIO_COLORS)]
    out_dir = Path("reports/figures")

    plot_productivity(df, out_dir)
    plot_wait_time(df, out_dir)


if __name__ == "__main__":
    main()
