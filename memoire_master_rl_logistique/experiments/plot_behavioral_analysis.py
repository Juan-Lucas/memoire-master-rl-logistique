"""Analyse comportementale : dynamique des files d'attente par pelle.

Compare Nearest Shovel et PPO sur un seul épisode de 480 min (forte charge,
graine unique). Nearest Shovel envoie toute la flotte à la pelle 1 dans tous
les régimes, car le camion reste à la pelle qu'il vient de servir : la file
observée n'est donc pas un effet propre à la forte charge.
Génère :
  - reports/figures/behavioral_analysis.png

Figure à 3 panneaux :
  (A) Queue backlog (min) par pelle — Nearest Shovel
  (B) Queue backlog (min) par pelle — PPO
  (C) Entropie d'affectation : Nearest (hot spot) vs PPO (équilibrage)
"""

from __future__ import annotations

from math import log2
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from stable_baselines3 import PPO

from memoire_master_rl_logistique.baselines.nearest_policy import NearestShovelPolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv

SEED = 42
SHOVEL_COUNT = 3
DUMP_COUNT = 2
TRUCK_COUNT = 18          # high-load fleet
EPISODE_MIN = 480.0
BREAKDOWN_PROB = 0.02     # high-load : faible taux de panne

SHOVEL_COLORS = ["#1565C0", "#AD1457", "#2E7D32"]
SHOVEL_LABELS = [f"Shovel {i+1}" for i in range(SHOVEL_COUNT)]


def _pick_action_sb3(policy, obs: np.ndarray) -> int:
    """Appelle SB3 policy.predict() et retourne l'action entière."""
    result = policy.predict(obs, deterministic=True)
    return int(result[0]) if isinstance(result, tuple) else int(result)


def _run_episode(policy, env: MineEnv, is_nearest: bool = False) -> dict:
    """Exécute un épisode complet et enregistre les métriques à chaque step.

    Parameters
    ----------
    is_nearest : si True, passe truck_location à policy.predict()
                 (requis par NearestShovelPolicy).
    """
    obs, info = env.reset(seed=SEED)
    times: list[float] = []
    backlogs: list[list[float]] = []    # [shovel_0, shovel_1, shovel_2]
    assignments: list[int] = []         # index pelle (0-2), -1 si hors plage
    disruptions: list[float] = []       # temps où une panne est détectée
    prev_total_wait = 0.0

    terminated, truncated = False, False
    while not (terminated or truncated):
        t_now = info["current_time_min"]
        shovel_avail = info["shovel_available_at"]
        backlog = [max(0.0, sa - t_now) for sa in shovel_avail]

        if isinstance(policy, PPO):
            action = _pick_action_sb3(policy, obs)
        elif is_nearest:
            truck_idx = env.current_truck_idx
            truck_loc = env.truck_locations[truck_idx]
            action = policy.predict(obs, info=info, truck_location=truck_loc)
        else:
            result = policy.predict(obs, info=info)
            action = int(result) if not isinstance(result, tuple) else int(result[0])

        shovel_idx = action // DUMP_COUNT if action < SHOVEL_COUNT * DUMP_COUNT else -1

        times.append(t_now)
        backlogs.append(backlog)
        assignments.append(shovel_idx)

        obs, _, terminated, truncated, info = env.step(action)

        # Détecte une panne : saut dans total_wait_min > 8 min en un seul step
        total_wait = info["total_wait_min"]
        if (total_wait - prev_total_wait) > 8.0:
            disruptions.append(t_now)
        prev_total_wait = total_wait

    return {
        "times": np.array(times),
        "backlogs": np.array(backlogs),   # (N, 3)
        "assignments": np.array(assignments),
        "disruptions": disruptions,
    }


def _assignment_dist(assignments: np.ndarray, window: int = 50) -> np.ndarray:
    """Fraction des affectations par pelle sur une fenêtre glissante."""
    n = len(assignments)
    dist = np.zeros((n, SHOVEL_COUNT))
    for i in range(n):
        start = max(0, i - window)
        window_vals = assignments[start:i+1]
        valid = window_vals[window_vals >= 0]
        if len(valid) > 0:
            for s in range(SHOVEL_COUNT):
                dist[i, s] = np.sum(valid == s) / len(valid)
    return dist


def _entropy(dist_row: np.ndarray) -> float:
    """Entropie de Shannon normalisée (0=imbalancé, 1=parfait équilibre)."""
    row = dist_row[dist_row > 0]
    if len(row) == 0 or SHOVEL_COUNT <= 1:
        return 0.0
    h = -sum(p * log2(p) for p in row)
    return h / log2(SHOVEL_COUNT)


def main() -> None:
    ppo_path = "data/results/ppo_high_load/ppo_mine_agent.zip"
    ppo_model = PPO.load(ppo_path)

    env_kwargs = dict(
        truck_count=TRUCK_COUNT,
        shovel_count=SHOVEL_COUNT,
        dump_count=DUMP_COUNT,
        episode_minutes=EPISODE_MIN,
        seed=SEED,
        breakdown_probability=BREAKDOWN_PROB,
        reward_weights=(1.0, 0.1, 0.05, 0.3),
    )
    # Build environment first so NearestShovelPolicy can access env.graph
    env_nearest = MineEnv(**env_kwargs)
    env_nearest.reset(seed=SEED)   # initialise self.graph

    shovel_node_ids = [f"shovel_{i+1}" for i in range(SHOVEL_COUNT)]
    dump_node_ids   = [f"dump_{i+1}"   for i in range(DUMP_COUNT)]
    nearest = NearestShovelPolicy(
        graph=env_nearest.graph,
        shovel_node_ids=shovel_node_ids,
        dump_node_ids=dump_node_ids,
    )

    env_ppo = MineEnv(**env_kwargs)

    print("Running Nearest Shovel episode...")
    data_nearest = _run_episode(nearest, env_nearest, is_nearest=True)
    print("Running PPO episode...")
    data_ppo     = _run_episode(ppo_model, env_ppo)

    # --- Figure ---
    # Panels A and B share both x and y axes for direct comparison
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.42)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a, sharey=ax_a)
    ax_c = fig.add_subplot(gs[2], sharex=ax_a)
    axes = [ax_a, ax_b, ax_c]

    # Shared y-limits for panels A and B (direct comparison)
    all_backlogs = np.concatenate([data_nearest["backlogs"], data_ppo["backlogs"]])
    y_max_ab = np.ceil(np.nanmax(all_backlogs) * 1.12)

    def _plot_backlogs(ax, data: dict, title: str) -> None:
        times = data["times"]
        bl = data["backlogs"]
        for s in range(SHOVEL_COUNT):
            ax.plot(times, bl[:, s], color=SHOVEL_COLORS[s],
                    label=SHOVEL_LABELS[s], linewidth=1.2, alpha=0.9)
        ax.set_ylabel("Queue backlog (min)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlim(0, EPISODE_MIN)
        ax.set_ylim(-0.5, y_max_ab)

    _plot_backlogs(ax_a, data_nearest,
                   "(A) Nearest Shovel — shovel queue backlog over time (high-load, 18 trucks)")
    _plot_backlogs(ax_b, data_ppo,
                   "(B) PPO — shovel queue backlog over time (same y-scale as A)")
    plt.setp(ax_a.get_xticklabels(), visible=False)

    # Panel C: assignment entropy comparison
    ax = ax_c
    dist_nearest = _assignment_dist(data_nearest["assignments"])
    dist_ppo     = _assignment_dist(data_ppo["assignments"])
    ent_nearest  = np.array([_entropy(dist_nearest[i]) for i in range(len(dist_nearest))])
    ent_ppo      = np.array([_entropy(dist_ppo[i])     for i in range(len(dist_ppo))])

    ax.plot(data_nearest["times"], ent_nearest, color="#37474F", linewidth=1.3,
            alpha=0.9, label="Nearest Shovel (greedy → hot spot, H≈0)")
    ax.plot(data_ppo["times"],     ent_ppo,     color="#AD1457", linewidth=1.3,
            alpha=0.9, label="PPO (adaptive load balancing, H>0)")
    ax.axhline(y=1.0, color="#2E7D32", linestyle=":", linewidth=1.0, alpha=0.7,
               label="Uniform assignment (H = 1)")
    ax.set_xlabel("Simulation time (min)", fontsize=9)
    ax.set_ylabel("Assignment entropy\n(normalised, 50-step window)", fontsize=9)
    ax.set_title(
        "(C) Assignment entropy: Nearest Shovel concentrates flow (hot spot); "
        "PPO distributes across shovels",
        fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(0, EPISODE_MIN)

    scenario_note = (
        f"High-load scenario: {TRUCK_COUNT} trucks, {BREAKDOWN_PROB*100:.0f}% breakdown probability, seed {SEED}."
    )
    fig.text(0.5, 0.005, scenario_note, ha="center", fontsize=8,
             color="#546E7A", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    # tight_layout conflicts with shared axes; re-apply xlim
    for ax in [ax_a, ax_b, ax_c]:
        ax.set_xlim(0, EPISODE_MIN)

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "behavioral_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardée : {out_path}")


if __name__ == "__main__":
    main()