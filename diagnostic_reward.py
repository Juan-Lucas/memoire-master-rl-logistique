"""Script utilitaire pour diagnostiquer la récompense moyenne par action.

Ce script exécute un petit test en comparant la distribution des récompenses
pour des actions fixes (toujours choisir la même pelle). Utile pour vérifier
les effets de la fonction de récompense et détecter des régressions.

Usage rapide :
  python diagnostic_reward.py
"""

from memoire_master_rl_logistique.env.mine_env import MineEnv
import numpy as np


def _run_test(action: int, env: MineEnv, n_steps: int = 300) -> list[float]:
    obs, _info = env.reset(seed=42)
    rewards = []
    for _ in range(n_steps):
        obs, r, term, trunc, _info = env.step(action)
        rewards.append(r)
        if term or trunc:
            break
    return rewards


def main() -> None:
    env = MineEnv()

    print("=== Test action 0 (toujours pelle 0) ===")
    rewards_a0 = _run_test(0, env)
    print(f"  Min={min(rewards_a0):.3f}, Max={max(rewards_a0):.3f}")
    print(f"  Moy={np.mean(rewards_a0):.3f}, Std={np.std(rewards_a0):.3f}")

    print("=== Test action 1 (toujours pelle 1) ===")
    rewards_a1 = _run_test(1, env)
    print(f"  Min={min(rewards_a1):.3f}, Max={max(rewards_a1):.3f}")
    print(f"  Moy={np.mean(rewards_a1):.3f}, Std={np.std(rewards_a1):.3f}")

    print("=== Test action 2 (toujours pelle 2) ===")
    rewards_a2 = _run_test(2, env)
    print(f"  Min={min(rewards_a2):.3f}, Max={max(rewards_a2):.3f}")
    print(f"  Moy={np.mean(rewards_a2):.3f}, Std={np.std(rewards_a2):.3f}")

    print("\n=== Différence entre actions ===")
    print(f"  Action 0 vs Action 1 : {np.mean(rewards_a0)-np.mean(rewards_a1):.4f}")
    print(f"  Action 0 vs Action 2 : {np.mean(rewards_a0)-np.mean(rewards_a2):.4f}")


if __name__ == "__main__":
    main()