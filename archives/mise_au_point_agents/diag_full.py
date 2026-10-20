"""Diagnostic au budget complet : PPO, 2 000 000 pas, meilleure configuration du réglage
(entropie 0, apprentissage 1e-4), graine d'entraînement 1, nominal.
Usage : python diag_full.py {cycle_end,decision} DOSSIER
Évaluation : productivité sur les graines de sélection 20000-20009 ; courbe d'apprentissage
résumée en 10 intervalles du budget (productivité pendant l'entraînement).
"""
from collections import Counter
import functools
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.policies import agent_policy_fn, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl import env_factory
from memoire_master_rl_logistique.rl.model_loading import load_model
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS, TUNING_SEEDS

timing, out = sys.argv[1], Path(sys.argv[2]) / f"ppo_{sys.argv[1]}"
TIMESTEPS = 2_000_000
env_factory.MineEnv = functools.partial(MineEnv, reward_timing=timing)

s = SCENARIOS["nominal"]
train_ppo(total_timesteps=TIMESTEPS, seed=TRAIN_SEEDS[0], output_dir=str(out), progress_bar=False,
          ent_coef=0.0, learning_rate=1e-4, truck_count=s.truck_count,
          episode_minutes=s.episode_minutes, breakdown_probability=s.breakdown_probability,
          reward_weights=s.reward_weights)

log = pd.read_csv(out / "training_kpis.csv")
bins = np.minimum((log["timesteps"] - 1) * 10 // TIMESTEPS, 9)
curve = (log["tonnage_t"] / 8).groupby(bins).mean().round(0).astype(int).tolist()
print(f"COURBE {timing} (t/h par dixième du budget) : {curve}", flush=True)

policy = agent_policy_fn(load_model(out, "PPO", make_env(s)), "PPO")
tph, shares, waits = [], Counter(), []
for seed in TUNING_SEEDS:
    env = make_env(s, log_cycles=True)
    k = run_single_episode(env, policy, seed)
    tph.append(k["productivity_tph"]); waits.append(k["wait_share_pct"])
    shares.update(c.shovel_id for c in env.cycle_log if c.kind == "cycle")
tot = sum(shares.values())
print(f"RESULT PPO {timing} 2M : {np.mean(tph):.1f} ± {np.std(tph, ddof=1):.1f} t/h, "
      f"ATTENDRE {np.mean(waits):.1f} %, pelles "
      + " ".join(f"{sid} {100 * shares[sid] / tot:.0f}%" for sid in ("S1", "S2", "S3")), flush=True)
