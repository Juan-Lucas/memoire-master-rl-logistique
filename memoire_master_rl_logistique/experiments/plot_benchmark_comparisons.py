"""Benchmark comparison plots (productivity and waiting time).

Source : data/results/benchmark_results.csv
Output :
- reports/figures/barplot_productivite_scenarios.png  (productivity bar chart)
- reports/figures/wait_time_comparison.png            (waiting-time bar chart)
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
    "high_breakdown": "High-breakdown",
    "high_load": "High-load",
    "nominal": "Nominal",
}

ALGO_ORDER = ["DQN", "Fixed", "PPO", "Q-Learning", "ShortestPath", "Nearest", "FIFO", "SARSA"]

WAIT_POLICY_COLORS = {
    "DQN": "#2196F3",
    "Fixed": "#1A237E",
    "PPO": "#FF5722",
}


def _bootstrap_ci(series: pd.Series, n_boot: int = 5_000, seed: int = 42) -> tuple[float, float]:
    """95 % bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    vals = series.to_numpy()
    boot_means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def plot_productivity(df: pd.DataFrame, out_dir: Path) -> None:
    means = (
        df.groupby(["policy", "scenario"])["productivity_tph"]
        .mean()
        .unstack("scenario")
        .loc[ALGO_ORDER, list(SCENARIO_COLORS)]
    )

    # Bootstrap 95 % CI per (policy, scenario)
    ci_lo = (
        df.groupby(["policy", "scenario"])["productivity_tph"]
        .apply(lambda s: _bootstrap_ci(s)[0])
        .unstack("scenario")
        .loc[ALGO_ORDER, list(SCENARIO_COLORS)]
    )
    ci_hi = (
        df.groupby(["policy", "scenario"])["productivity_tph"]
        .apply(lambda s: _bootstrap_ci(s)[1])
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
        lo = ci_lo[scenario].to_numpy()
        hi = ci_hi[scenario].to_numpy()
        yerr = np.array([values - lo, hi - values])
        bars = ax.bar(
            offsets, values, width,
            label=SCENARIO_LABELS[scenario],
            color=SCENARIO_COLORS[scenario],
        )
        ax.errorbar(
            offsets, values, yerr=yerr,
            fmt="none", ecolor="black", capsize=3, linewidth=0.8,
        )
        for bar, val in zip(bars, values):
            label = f"{val / 1000:.1f}K".replace(".", ",")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                label,
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_ORDER)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Productivity (t/h)")
    ax.set_title("Mean productivity (t/h) per policy and scenario — 3 scenarios")
    ax.legend(title="Scenario")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "barplot_productivite_scenarios.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def plot_wait_time(df: pd.DataFrame, out_dir: Path) -> None:
    scenarios = list(SCENARIO_COLORS)
    policies = list(WAIT_POLICY_COLORS)

    means = (
        df.groupby(["policy", "scenario"])["avg_wait_min_per_truck"]
        .mean()
        .unstack("scenario")
        .loc[policies, scenarios]
    )

    # Bootstrap 95 % CI per (policy, scenario)
    ci_lo_df = (
        df.groupby(["policy", "scenario"])["avg_wait_min_per_truck"]
        .apply(lambda s: _bootstrap_ci(s)[0])
        .unstack("scenario")
        .loc[policies, scenarios]
    )
    ci_hi_df = (
        df.groupby(["policy", "scenario"])["avg_wait_min_per_truck"]
        .apply(lambda s: _bootstrap_ci(s)[1])
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
        lo = ci_lo_df.loc[policy].to_numpy()
        hi = ci_hi_df.loc[policy].to_numpy()
        yerr = np.array([values - lo, hi - values])
        bars = ax.bar(
            offsets, values, width,
            label=policy,
            color=WAIT_POLICY_COLORS[policy],
        )
        ax.errorbar(
            offsets, values, yerr=yerr,
            fmt="none", ecolor="black", capsize=3, linewidth=0.8,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.axhline(30, color="#42A5F5", linestyle="--", linewidth=1.5)
    ax.text(-0.45, 31, "Target threshold: 30 min", color="#1565C0",
            ha="left", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean waiting time (min)")
    ax.set_title("Mean waiting time (min) — Fixed Assignment, DQN and PPO")
    ax.legend(title="Policy")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "wait_time_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def main() -> None:
    df = pd.read_csv("data/results/benchmark_results.csv")
    df = df[df["scenario"].isin(SCENARIO_COLORS)]
    out_dir = Path("reports/figures")

    plot_productivity(df, out_dir)
    plot_wait_time(df, out_dir)


if __name__ == "__main__":
    main()
