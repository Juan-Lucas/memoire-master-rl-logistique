"""Courbes de perte (Policy Loss / Value Loss) pour DQN et PPO -- les 3
scenarios (nominal, high_load, high_breakdown).

Source :
- data/results/dqn_{scenario}_losscurve/sb3_logs/progress.csv
- data/results/ppo_{scenario}_losscurve/sb3_logs/progress.csv

Sortie :
- reports/figures/loss_dqn_{nominal,highload,highbreakdown}.png
- reports/figures/loss_ppo_{nominal,highload,highbreakdown}.png
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

WINDOW = 30

SCENARIOS = [
    ("nominal", "nominal", "scénario nominal"),
    ("high_load", "highload", "scénario high_load"),
    ("high_breakdown", "highbreakdown", "scénario high_breakdown"),
]


def plot_dqn(out_dir: Path, scenario_dir: str, suffix: str, scenario_label: str) -> None:
    df = pd.read_csv(f"data/results/dqn_{scenario_dir}_losscurve/sb3_logs/progress.csv").dropna(subset=["train/loss"])
    steps = df["time/total_timesteps"] / 1e6
    loss = df["train/loss"]
    loss_smooth = loss.rolling(WINDOW, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, loss, color="#90CAF9", linewidth=0.6, label="Perte (brute)")
    ax.plot(steps, loss_smooth, color="#2196F3", linewidth=2, label=f"Moyenne mobile ({WINDOW} points)")
    ax.set_xlabel("Steps d'entraînement (millions)")
    ax.set_ylabel("Perte du réseau Q (train/loss)")
    ax.set_title(f"DQN — Perte du réseau Q au cours de l'entraînement ({scenario_label})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"loss_dqn_{suffix}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")


def plot_ppo(out_dir: Path, scenario_dir: str, suffix: str, scenario_label: str) -> None:
    df = pd.read_csv(f"data/results/ppo_{scenario_dir}_losscurve/sb3_logs/progress.csv").dropna(subset=["train/policy_gradient_loss"])
    steps = df["time/total_timesteps"] / 1e6
    policy_loss = df["train/policy_gradient_loss"].rolling(WINDOW, min_periods=1).mean()
    value_loss = df["train/value_loss"].rolling(WINDOW, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(steps, policy_loss, color="#FF5722", linewidth=1.5)
    axes[0].set_ylabel("Policy Loss\n(moy. mobile)")
    axes[0].set_title(f"PPO — Policy Loss et Value Loss au cours de l'entraînement ({scenario_label})")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, value_loss, color="#FF9800", linewidth=1.5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Value Loss\n(moy. mobile, échelle log)")
    axes[1].set_xlabel("Steps d'entraînement (millions)")
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    out_path = out_dir / f"loss_ppo_{suffix}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")


def main() -> None:
    out_dir = Path("reports/figures")
    for scenario_dir, suffix, scenario_label in SCENARIOS:
        plot_dqn(out_dir, scenario_dir, suffix, scenario_label)
        plot_ppo(out_dir, scenario_dir, suffix, scenario_label)


if __name__ == "__main__":
    main()
