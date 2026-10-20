from collections import Counter
from pathlib import Path
import numpy as np
from memoire_master_rl_logistique.experiments.policies import agent_policy_fn, heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.model_loading import load_model
from memoire_master_rl_logistique.seeds import TUNING_SEEDS
s = SCENARIOS["nominal"]
pols = heuristic_policies(s)
rng = np.random.default_rng(0)
pols["Hasard (sans ATTENDRE)"] = lambda o, i, e: int(rng.integers(e.action_space.n - 1))
root = Path("data/results_v2/tuning")
for algo, cfg in (("PPO", "ent_coef=0_learning_rate=0.0001"), ("DQN", "exploration_fraction=0.1_learning_rate=0.0003")):
    env = make_env(s)
    pols[f"{algo} meilleur"] = agent_policy_fn(load_model(root / algo / cfg, algo, env), algo)
for name, pol in pols.items():
    k, shares, waits = [], Counter(), []
    for seed in TUNING_SEEDS:
        env = make_env(s, log_cycles=True)
        r = run_single_episode(env, pol, seed); k.append(r["productivity_tph"]); waits.append(r["wait_share_pct"])
        shares.update(c.shovel_id for c in env.cycle_log if c.kind == "cycle")
    tot = sum(shares.values())
    print(f"{name:24s} {np.mean(k):7.1f} t/h   ATTENDRE {np.mean(waits):5.1f} %   pelles " + " ".join(f"{sid} {100*shares[sid]/tot:3.0f}%" for sid in ("S1","S2","S3")))
