"""Tests de la piste 2 d'amélioration des agents : récompense normalisée et γ de PPO."""

import json

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import VecNormalize

from memoire_master_rl_logistique.rl.train_ppo import REWARD_CLIP, VECNORMALIZE_FILE, train_ppo

COMMON = dict(total_timesteps=2_048, episode_minutes=60.0, progress_bar=False)


def test_recompense_normalisee_seulement_pendant_l_entrainement(tmp_path):
    """Avec ``normalize_reward``, PPO apprend sur une récompense normalisée (VecNormalize,
    observations intactes, coupure déclarée) ; les journaux gardent la récompense brute ;
    les valeurs utilisées sont dans metadata.json et les statistiques sont enregistrées."""
    model = train_ppo(output_dir=str(tmp_path), gamma=0.995, normalize_reward=True, **COMMON)

    env = model.get_env()
    assert isinstance(env, VecNormalize) and env.norm_reward and not env.norm_obs
    assert env.gamma == model.gamma == 0.995 and env.clip_reward == REWARD_CLIP
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["training"]["normalize_reward"] is True and meta["training"]["gamma"] == 0.995
    assert meta["training"]["reward_clip"] == REWARD_CLIP
    assert (tmp_path / VECNORMALIZE_FILE).exists()

    # Le journal des épisodes contient les récompenses brutes de Monitor (sous VecNormalize),
    # épisode par épisode, et non les récompenses normalisées.
    raw = env.venv.envs[0].get_episode_rewards()
    logged = pd.read_csv(tmp_path / "training_kpis.csv")["reward"]
    np.testing.assert_allclose(logged, raw, atol=1e-4)


def test_sans_normalisation_comme_la_v1(tmp_path):
    """Options désactivées (protocole v1 : γ = 0,99, récompense brute) : pas de
    VecNormalize, et deux entraînements identiques donnent les mêmes poids."""
    v1 = dict(gamma=0.99, normalize_reward=False, **COMMON)
    one = train_ppo(output_dir=str(tmp_path / "a"), **v1)
    two = train_ppo(output_dir=str(tmp_path / "b"), **v1)

    assert not isinstance(one.get_env(), VecNormalize) and one.gamma == 0.99
    for a, b in zip(one.policy.state_dict().values(), two.policy.state_dict().values(),
                    strict=True):
        assert np.array_equal(a.cpu().numpy(), b.cpu().numpy())
    meta = json.loads((tmp_path / "a" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["training"]["normalize_reward"] is False and "reward_clip" not in meta["training"]
    assert not (tmp_path / "a" / VECNORMALIZE_FILE).exists()
