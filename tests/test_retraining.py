"""Réentraînements de diagnostic (rapport du 17/11/2026 : M5, M6, T17).

Variantes de PPO et de DQN (``rl/agent_variants.py``), carburant dans la récompense
(``MineEnv.fuel_cost_scale_l``) et lanceur ``experiments/retraining.py``.
"""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import DQN
import torch as th

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.env.observation_builder import observation_layout
from memoire_master_rl_logistique.experiments import retraining as rt
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl import agent_variants as av
from memoire_master_rl_logistique.rl.model_paths import model_dir
from memoire_master_rl_logistique.rl.time_discount import TimeDiscountReplayBuffer
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.seeds import EVAL_SEEDS, EXTENSION_EVAL_SEEDS, TUNING_SEEDS

NOMINAL = SCENARIOS["nominal"]


def _fixed_episode(seed, **env_kwargs):
    env = make_env(NOMINAL, **env_kwargs)
    kpis = run_single_episode(env, heuristic_policies(NOMINAL)["Fixed"], seed, "Fixed")
    return env, kpis


# --- Configuration des variantes ---------------------------------------------------------

def test_six_variantes_chacune_ne_change_qu_un_element():
    variants = av.load_variants()
    assert list(variants) == ["PPO-no-route-wait", "PPO-fuel", "PPO-w3-0", "PPO-w3-0.1",
                              "DQN-no-wait", "Double-DQN"]
    nominal = tuple(NOMINAL.reward_weights)
    flags = ("drop_route_wait", "no_wait_action", "double_q", "fuel_in_reward")
    for v in variants.values():
        changes = sum(getattr(v, f) for f in flags) + (v.reward_weights != nominal)
        assert changes == 1, v
    assert variants["PPO-w3-0"].reward_weights[2] == 0.0
    assert variants["PPO-w3-0.1"].reward_weights[2] == 0.1
    with pytest.raises(ValueError):
        av.AgentVariant("x", "PPO", nominal, double_q=True)


def test_trente_entrainements_dqn_d_abord():
    jobs = rt.variant_jobs()
    assert len(jobs) == 30 and len(set(jobs)) == 30
    algos = [j.algorithm() for j in jobs]
    assert algos == ["DQN"] * 10 + ["PPO"] * 20
    assert jobs[0].label() == "DQN-no-wait_seed1"


def test_evaluation_sur_les_graines_72_a_101():
    assert list(EXTENSION_EVAL_SEEDS) == list(range(72, 102))
    assert not set(EXTENSION_EVAL_SEEDS) & set(EVAL_SEEDS)
    assert not set(EXTENSION_EVAL_SEEDS) & set(TUNING_SEEDS)


# --- Carburant dans la récompense --------------------------------------------------------

def test_sans_echelle_la_recompense_de_la_campagne_est_inchangee():
    _, base = _fixed_episode(101)
    env, same = _fixed_episode(101, fuel_cost_scale_l=None)
    assert same["total_reward"] == base["total_reward"]
    # cumuls des cycles cohérents avec la récompense de distance
    w3 = NOMINAL.reward_weights[2]
    assert env.episode_reward_terms["distance"] == pytest.approx(-w3 * env.episode_cycle_km / 5)


def test_variante_carburant_remplace_la_distance_par_le_carburant_du_cycle():
    scale = 23.53
    env, kpis = _fixed_episode(101, fuel_cost_scale_l=scale)
    _, base = _fixed_episode(101)
    w3 = NOMINAL.reward_weights[2]
    assert env.episode_reward_terms["distance"] == pytest.approx(-w3 * env.episode_cycle_fuel_l / scale)
    # même épisode (la récompense n'agit pas sur la simulation), autre récompense
    assert kpis["productivity_tph"] == base["productivity_tph"]
    assert kpis["total_reward"] != base["total_reward"]
    with pytest.raises(ValueError):
        MineEnv(fuel_cost_scale_l=0.0)


def test_echelle_du_carburant_etalonnee_sur_les_graines_de_selection():
    fuel = km = 0.0
    for seed in TUNING_SEEDS:
        env, _ = _fixed_episode(seed)
        fuel, km = fuel + env.episode_cycle_fuel_l, km + env.episode_cycle_km
    scale = fuel / (km / 5)
    assert scale == pytest.approx(av.retraining_config()["fuel_cost_scale_l"], abs=0.01)


# --- Wrappers ----------------------------------------------------------------------------

def test_observation_sans_les_attentes_prevues():
    env = av.wrap_env(MineEnv(), av.load_variants()["PPO-no-route-wait"])
    obs, _ = env.reset(seed=101)
    assert obs.shape == (17,) and env.observation_space.shape == (17,)
    full, _ = MineEnv().reset(seed=101)
    route = observation_layout(3, 2)["route_wait"]
    assert np.array_equal(obs, np.delete(full, np.arange(route.start, route.stop)))
    kept = av.kept_observation_indices()
    assert len(kept) == 17 and not set(range(route.start, route.stop)) & set(kept)


def test_actions_sans_attendre():
    env = av.wrap_env(MineEnv(), av.load_variants()["DQN-no-wait"])
    assert env.action_space.n == 6
    env.reset(seed=101)
    for action in range(6):
        env.step(action)
    assert env.unwrapped.wait_action_count == 0
    with pytest.raises(ValueError):
        env.step(6)


def test_politique_d_evaluation_reduit_l_observation():
    variant = av.load_variants()["PPO-no-route-wait"]
    seen = []
    model = SimpleNamespace(predict=lambda obs, deterministic: (seen.append(obs) or 3, None))
    full, _ = MineEnv().reset(seed=101)
    assert av.variant_policy_fn(model, variant)(full, {}, None) == 3
    assert seen[0].shape == (17,)


# --- Double DQN --------------------------------------------------------------------------

def test_cible_double_dqn(monkeypatch):
    """La cible choisit l'action suivante avec le réseau en ligne et l'évalue avec le réseau
    cible, avec le facteur γ_min^Δt de la transition."""
    model = av.DoubleDQN("MlpPolicy", MineEnv(episode_minutes=60.0), learning_starts=50,
                         seed=0, replay_buffer_class=TimeDiscountReplayBuffer,
                         replay_buffer_kwargs={"gamma_per_minute": 0.997})
    model.learn(200)
    batch = model.replay_buffer.sample(16)
    monkeypatch.setattr(model.replay_buffer, "sample", lambda *a, **k: batch)
    with th.no_grad():
        online = model.q_net(batch.next_observations)
        target = model.q_net_target(batch.next_observations)
        expected = batch.rewards + (1 - batch.dones) * batch.discounts * \
            th.gather(target, 1, online.argmax(dim=1, keepdim=True))
    captured = {}

    def loss(current, tgt):
        captured["target"] = tgt
        return th.nn.functional.smooth_l1_loss(current, tgt)
    monkeypatch.setattr(av, "F", SimpleNamespace(smooth_l1_loss=loss))
    model.train(gradient_steps=1, batch_size=16)
    assert th.allclose(captured["target"], expected)
    assert batch.discounts is not None and not th.all(batch.discounts == model.gamma)


# --- Entraînement et évaluation de bout en bout ------------------------------------------

def _tiny(variant_name, train_seed, root):
    scenario = replace(NOMINAL, total_timesteps=1_024, episode_minutes=60.0)
    rt.train_variant(av.load_variants()[variant_name], scenario, train_seed,
                     rt.VariantJob(variant_name, train_seed).directory(root))


@pytest.mark.parametrize("name", ["PPO-no-route-wait", "PPO-fuel", "DQN-no-wait", "Double-DQN"])
def test_variante_entrainee_porte_sa_description(tmp_path, name):
    _tiny(name, 1, tmp_path)
    directory = rt.VariantJob(name, 1).directory(tmp_path)
    variant = av.load_variants()[name]
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert meta["training"]["agent_variant"] == variant.as_dict()
    assert meta["observation_size"] == (17 if variant.drop_route_wait else 23)
    rt.check_variant_metadata(directory, variant)
    other = av.load_variants()["PPO-w3-0"]
    with pytest.raises(ValueError):
        rt.check_variant_metadata(directory, other)
    if name == "Double-DQN":
        assert DQN.load(str(directory / "dqn_mine_agent.zip")).__class__.__name__ in ("DQN", "DoubleDQN")


def test_campagne_inchangee_sans_variante(tmp_path):
    """Sans variante, ni la signature ni les métadonnées ne gagnent de champ."""
    train_ppo(total_timesteps=1_024, episode_minutes=60.0, output_dir=str(tmp_path),
              progress_bar=False)
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert "agent_variant" not in meta["training"] and "fuel_cost_scale_l" not in meta["training"]


def test_evaluation_et_analyse(tmp_path, monkeypatch):
    """Toutes les variantes (budget minuscule), références de la campagne, 2 graines."""
    monkeypatch.setattr(rt, "regime", lambda: replace(NOMINAL, episode_minutes=60.0))
    root, camp = tmp_path / "variantes", tmp_path / "campagne"
    for job in rt.variant_jobs(train_seeds=[1]):
        _tiny(job.variant, 1, root)
    monkeypatch.setattr(rt, "TRAIN_SEEDS", [1])
    monkeypatch.setattr(rt, "variant_jobs", lambda: [rt.VariantJob(n, 1) for n in av.load_variants()])
    with pytest.raises(FileNotFoundError):
        rt.evaluate(root, camp, workers=1, seeds=[72, 73])
    train_ppo(total_timesteps=1_024, episode_minutes=60.0, progress_bar=False,
              output_dir=str(model_dir(camp, "PPO", "nominal", 1)))
    train_dqn(total_timesteps=1_024, episode_minutes=60.0, progress_bar=False,
              output_dir=str(model_dir(camp, "DQN", "nominal", 1)))
    out = rt.evaluate(root, camp, workers=1, seeds=[72, 73])
    with pytest.raises(FileExistsError):
        rt.evaluate(root, camp, workers=1, seeds=[72, 73])
    results = pd.read_csv(out)
    assert len(results) == (2 + 2 + 6) * 2
    assert set(results["seed"]) == {72, 73}
    nowait = results[results["policy"] == "DQN-no-wait"]
    assert (nowait["wait_share_pct"] == 0).all()
    shares = results[["share_S1", "share_S2", "share_S3"]].sum(axis=1)
    assert np.allclose(shares[results["productivity_tph"] > 0], 100)
    summary = rt.analyze(root)
    assert set(summary["policy"]) == set(results["policy"])
    row = summary.set_index("policy").loc["PPO-fuel"]
    assert "productivity_tph_vs_ref_pct" in row and "specific_fuel_l_per_ton_vs_mtwt_pct" in row
    assert rt.reference_of("Double-DQN") == "DQN" and rt.reference_of("MTWT") == "MTWT"
