"""Essai 2 : récompense estimée à la décision (1b) avec un coût d'opportunité des minutes en file.

Même configuration que la meilleure du réglage, même graine d'entraînement (1), même budget
(200 000 pas), même critère (productivité, nominal, graines de sélection 20000-20009).
Usage : python reward_queue_test.py {PPO,DQN} DOSSIER
"""
from collections import Counter
import functools
from pathlib import Path
import sys

import numpy as np

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.policies import agent_policy_fn, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl import env_factory
from memoire_master_rl_logistique.rl.model_loading import load_model
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS, TUNING_SEEDS

algo, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
# Environnements d'entraînement avec la récompense estimée à la décision.
env_factory.MineEnv = functools.partial(MineEnv, reward_timing="decision", queue_opportunity=True)

s = SCENARIOS["nominal"]
common = dict(total_timesteps=200_000, seed=TRAIN_SEEDS[0], output_dir=str(out),
              progress_bar=False, truck_count=s.truck_count, episode_minutes=s.episode_minutes,
              breakdown_probability=s.breakdown_probability, reward_weights=s.reward_weights)
if algo == "PPO":
    train_ppo(ent_coef=0.0, learning_rate=1e-4, **common)
else:
    train_dqn(exploration_fraction=0.1, learning_rate=3e-4, **common)

# Évaluation sur l'environnement normal : seule la productivité compte.
policy = agent_policy_fn(load_model(out, algo, make_env(s)), algo)
tph, shares, waits = [], Counter(), []
for seed in TUNING_SEEDS:
    env = make_env(s, log_cycles=True)
    k = run_single_episode(env, policy, seed)
    tph.append(k["productivity_tph"]); waits.append(k["wait_share_pct"])
    shares.update(c.shovel_id for c in env.cycle_log if c.kind == "cycle")
tot = sum(shares.values())
print(f"RESULT {algo} décision + file : {np.mean(tph):.1f} ± {np.std(tph, ddof=1):.1f} t/h, "
      f"ATTENDRE {np.mean(waits):.1f} %, pelles "
      + " ".join(f"{sid} {100 * shares[sid] / tot:.0f}%" for sid in ("S1", "S2", "S3")), flush=True)
