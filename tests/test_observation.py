"""Tests de T18 (observation de taille fixe, variables agrégées) et de T17 (aucune
composante constante ni saturée)."""

import numpy as np
import pytest

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.env.observation_builder import (
    MAX_TRUCKS,
    observation_layout,
    observation_size,
)
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS
from memoire_master_rl_logistique.rl.discretize import _discretize_obs
from memoire_master_rl_logistique.rl.model_loading import load_model
from memoire_master_rl_logistique.rl.model_metadata import EnvConfig
from memoire_master_rl_logistique.rl.train_ppo import train_ppo

LAYOUT = observation_layout(3, 2)


def episode_observations(scenario: str, policy: str, seed: int) -> np.ndarray:
    """Toutes les observations d'un épisode (politique heuristique ou aléatoire)."""
    s = SCENARIOS[scenario]
    env = make_env(s)
    rng = np.random.default_rng(seed)
    policy_fn = (heuristic_policies(s)[policy] if policy != "aléatoire"
                 else lambda obs, info, e: int(rng.integers(e.action_space.n)))
    obs, info = env.reset(seed=seed)
    seen = [obs]
    while True:
        obs, _, terminated, truncated, info = env.step(policy_fn(obs, info, env))
        seen.append(obs)
        if terminated or truncated:
            return np.array(seen)


@pytest.fixture(scope="module", params=MAIN_SCENARIOS)
def observations(request):
    """10 épisodes par régime (Fixed, FIFO, aléatoire), comme le demande T17."""
    return np.concatenate([
        episode_observations(request.param, policy, seed)
        for policy in ("Fixed", "FIFO", "aléatoire")
        for seed in SCENARIOS[request.param].seeds[:10]
    ])


# --- T18 : taille fixe ---------------------------------------------------------------

def test_meme_taille_quel_que_soit_le_nombre_de_camions():
    """Test du plan : ``observation_space.shape`` identique pour 12 et 18 camions (et 15)."""
    shapes = {MineEnv(truck_count=n).observation_space.shape for n in (12, 15, 18, MAX_TRUCKS)}

    assert shapes == {(observation_size(3, 2),)} == {(18,)}


def test_disposition_couvre_tout_le_vecteur():
    """Les blocs se suivent sans trou ni chevauchement : 4S + D + 4 composantes."""
    blocks = sorted(LAYOUT.values(), key=lambda b: b.start)

    assert blocks[0].start == 0
    assert all(a.stop == b.start for a, b in zip(blocks, blocks[1:]))
    assert blocks[-1].stop == 4 * 3 + 2 + 4
    assert observation_size(5, 3) == 4 * 5 + 3 + 4


def test_contenu_des_blocs():
    """Au début du poste : tout le monde au parc, pelles et verses libres, rien de produit."""
    env = MineEnv()
    obs, _ = env.reset(seed=42)

    assert (obs[LAYOUT["shovel_wait"]] == 0).all() and (obs[LAYOUT["dump_wait"]] == 0).all()
    assert (obs[LAYOUT["shovel_trucks"]] == 0).all()  # aucun camion encore rattaché à une pelle
    assert list(obs[LAYOUT["position"]]) == [1.0, 0.0, 0.0, 0.0]  # au parc
    assert obs[LAYOUT["tonnage"]][0] == obs[LAYOUT["fuel"]][0] == obs[LAYOUT["time"]][0] == 0


def test_camions_par_pelle():
    """Après un cycle, le camion est rattaché à sa pelle : 1/24 sur la pelle choisie."""
    env = MineEnv()
    env.reset(seed=42)
    obs, *_ = env.step(2)  # pelle 2, verse 1

    assert obs[LAYOUT["shovel_trucks"]][1] == pytest.approx(1 / MAX_TRUCKS)
    assert obs[LAYOUT["shovel_trucks"]].sum() == pytest.approx(1 / MAX_TRUCKS)


def test_etat_tabulaire_inchange():
    """Q-Learning et SARSA lisent les mêmes valeurs qu'avant T18 (attentes et distance la
    plus courte) : leur état discret ne dépend pas du nombre de camions."""
    env = MineEnv()
    obs, _ = env.reset(seed=42)

    assert _discretize_obs(obs, 8) == (0, 0, 0, 0, 0, 0)  # tout est libre, pelle 1 la plus proche


def test_ppo_nominal_s_execute_en_forte_charge(tmp_path):
    """Test du plan : un PPO entraîné brièvement avec 12 camions s'exécute sans erreur dans
    un environnement à 18 camions (régime d'entraînement déclaré : nominal)."""
    train_ppo(total_timesteps=1024, output_dir=str(tmp_path), log_freq=10_000,
              progress_bar=False)
    high_load = SCENARIOS["high_load"]
    env = make_env(high_load)
    model = load_model(tmp_path, "PPO", env, trained_on=EnvConfig.from_scenario(SCENARIOS["nominal"]))

    kpis = run_single_episode(env, lambda obs, info, e: int(model.predict(obs)[0]), seed=42)

    assert kpis["truck_count"] == 18 and kpis["productivity_tph"] > 0


# --- T17 : aucune composante constante ni saturée --------------------------------------

def test_aucune_composante_constante(observations):
    """Test du plan : sur 10 épisodes, aucune composante n'a un écart-type nul."""
    assert (observations.std(axis=0) > 0).all()


def test_aucune_variable_continue_saturee(observations):
    """Test du plan : les variables continues (pas le one-hot de position) valent exactement 1
    moins de 5 % du temps. Le temps du poste n'atteint 1 qu'à la dernière décision."""
    continuous = [j for name, block in LAYOUT.items() if name != "position"
                  for j in range(block.start, block.stop)]

    assert ((observations[:, continuous] == 1.0).mean(axis=0) < 0.05).all()
