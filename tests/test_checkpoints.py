"""Sauvegardes en cours d'entraînement et reprise après une interruption (rl/checkpoints.py).

Une interruption (machine à court de mémoire, arrêt du processus) est simulée par une
exception levée au milieu de l'entraînement.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import VecNormalize

from memoire_master_rl_logistique.rl import train_q_learning as q_learning_module
from memoire_master_rl_logistique.rl import train_sarsa as sarsa_module
from memoire_master_rl_logistique.rl.callbacks import KPILoggerCallback
from memoire_master_rl_logistique.rl.checkpoints import (
    CHECKPOINT_DIR,
    REPLAY_BUFFER_FILE,
    VECNORMALIZE_FILE,
    read_state,
)
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.seeds import training_episode_seed


class Interruption(Exception):
    """Arrêt brutal simulé."""


def interrupt_sb3_after(monkeypatch, timesteps):
    """Fait échouer l'entraînement PPO/DQN dès qu'il dépasse ``timesteps`` pas."""
    original = KPILoggerCallback._on_step

    def failing(self):
        if self.num_timesteps > timesteps:
            raise Interruption
        return original(self)

    monkeypatch.setattr(KPILoggerCallback, "_on_step", failing)


def metadata(path):
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


# --- PPO, DQN ----------------------------------------------------------------------------

PPO_RUN = dict(total_timesteps=3_072, episode_minutes=60.0, progress_bar=False,
               checkpoint_freq=1_024)  # PPO collecte 1 024 pas à la fois (n_steps)


def test_ppo_reprend_a_la_derniere_sauvegarde(tmp_path, monkeypatch):
    """Interrompu après 2 048 pas, PPO repart de sa sauvegarde à 2 048 pas, avec les
    statistiques de normalisation de la récompense, et finit son budget. La reprise est
    écrite dans metadata.json ; la sauvegarde est supprimée à la fin."""
    interrupt_sb3_after(monkeypatch, 2_500)
    with pytest.raises(Interruption):
        train_ppo(output_dir=str(tmp_path), **PPO_RUN)
    state = read_state(tmp_path)
    assert state["num_timesteps"] == 2_048
    assert (tmp_path / CHECKPOINT_DIR / VECNORMALIZE_FILE).exists()
    assert not (tmp_path / "ppo_mine_agent.zip").exists()
    episodes_before = pd.read_csv(tmp_path / CHECKPOINT_DIR / "training_kpis.csv")

    monkeypatch.undo()
    model = train_ppo(output_dir=str(tmp_path), resume=True, **PPO_RUN)

    assert model.num_timesteps == 3_072
    env = model.get_env()
    assert isinstance(env, VecNormalize) and env.ret_rms.count > 2_048  # statistiques reprises
    assert metadata(tmp_path)["training"]["resumed_from_timesteps"] == [2_048]
    assert not (tmp_path / CHECKPOINT_DIR).exists()
    log = pd.read_csv(tmp_path / "training_kpis.csv")
    pd.testing.assert_frame_equal(log.iloc[:len(episodes_before)], episodes_before)
    assert log["timesteps"].is_monotonic_increasing and log["timesteps"].max() <= 3_072
    assert log["episode"].tolist() == list(range(len(log)))


def test_dqn_reprend_avec_sa_memoire_de_rejeu(tmp_path, monkeypatch):
    run = dict(total_timesteps=1_200, episode_minutes=60.0, progress_bar=False,
               checkpoint_freq=400)
    interrupt_sb3_after(monkeypatch, 900)
    with pytest.raises(Interruption):
        train_dqn(output_dir=str(tmp_path), **run)
    assert read_state(tmp_path)["num_timesteps"] == 800
    assert (tmp_path / CHECKPOINT_DIR / REPLAY_BUFFER_FILE).exists()

    monkeypatch.undo()
    model = train_dqn(output_dir=str(tmp_path), resume=True, **run)
    assert model.num_timesteps == 1_200
    assert model.replay_buffer.size() == 1_200  # 800 transitions reprises + 400 nouvelles
    assert metadata(tmp_path)["training"]["resumed_from_timesteps"] == [800]


def test_la_sauvegarde_ne_change_pas_l_entrainement(tmp_path):
    """Sans interruption, sauvegarder en cours de route donne exactement le même modèle
    (et les mêmes métadonnées, sans reprise) que ne pas sauvegarder."""
    run = dict(total_timesteps=2_048, episode_minutes=60.0, progress_bar=False)
    saved = train_ppo(output_dir=str(tmp_path / "a"), checkpoint_freq=1_024, **run)
    plain = train_ppo(output_dir=str(tmp_path / "b"), checkpoint_freq=None, **run)
    for a, b in zip(saved.policy.state_dict().values(), plain.policy.state_dict().values(),
                    strict=True):
        assert np.array_equal(a.cpu().numpy(), b.cpu().numpy())
    assert "resumed_from_timesteps" not in metadata(tmp_path / "a")["training"]


def test_garde_fous(tmp_path, monkeypatch):
    run = dict(total_timesteps=2_048, episode_minutes=60.0, progress_bar=False,
               checkpoint_freq=1_024)
    with pytest.raises(FileNotFoundError, match="Aucune sauvegarde"):
        train_ppo(output_dir=str(tmp_path), resume=True, **run)

    interrupt_sb3_after(monkeypatch, 1_500)
    with pytest.raises(Interruption):
        train_ppo(output_dir=str(tmp_path), **run)
    monkeypatch.undo()

    # Un nouvel entraînement ne démarre pas par-dessus un entraînement interrompu…
    with pytest.raises(FileExistsError, match="interrompu"):
        train_ppo(output_dir=str(tmp_path), **run)
    # … et une reprise refuse une autre configuration.
    with pytest.raises(ValueError, match="learning_rate"):
        train_ppo(output_dir=str(tmp_path), resume=True, learning_rate=1e-3, **run)
    with pytest.raises(ValueError, match="total_timesteps"):
        train_ppo(output_dir=str(tmp_path), resume=True, **{**run, "total_timesteps": 4_096})
    with pytest.raises(ValueError, match="n_envs"):
        train_ppo(output_dir=str(tmp_path), resume=True, n_envs=2, **run)
    # overwrite : on recommence de zéro.
    train_ppo(output_dir=str(tmp_path), overwrite=True, **run)
    assert "resumed_from_timesteps" not in metadata(tmp_path)["training"]


# --- Q-Learning, SARSA : reprise exacte ---------------------------------------------------

TABULAR_RUN = dict(n_episodes=7, episode_minutes=60.0, checkpoint_every=2)


@pytest.mark.parametrize("module, train, table", [
    (q_learning_module, q_learning_module.train_q_learning, "q_table.pkl"),
    (sarsa_module, sarsa_module.train_sarsa, "sarsa_table.pkl"),
])
def test_reprise_tabulaire_identique(tmp_path, monkeypatch, module, train, table):
    """Interrompu pendant le 6e épisode, l'entraînement repart de la sauvegarde faite après
    le 4e et donne exactement la même Q-table et les mêmes journaux qu'un entraînement
    sans interruption."""
    train(output_dir=str(tmp_path / "ref"), **TABULAR_RUN)

    def failing_seed(seed, episode=0):
        if episode == 5:
            raise Interruption
        return training_episode_seed(seed, episode)

    monkeypatch.setattr(module, "training_episode_seed", failing_seed)
    with pytest.raises(Interruption):
        train(output_dir=str(tmp_path / "cut"), **TABULAR_RUN)
    assert read_state(tmp_path / "cut")["episodes"] == 4
    monkeypatch.undo()
    train(output_dir=str(tmp_path / "cut"), resume=True, **TABULAR_RUN)

    ref, cut = (pickle.loads((tmp_path / d / table).read_bytes())["q_table"]
                for d in ("ref", "cut"))
    assert ref.keys() == cut.keys()
    assert all(np.array_equal(ref[s], cut[s]) for s in ref)
    for csv in ("training_diagnostics.csv", "training_rewards.csv"):
        assert (tmp_path / "ref" / csv).read_text() == (tmp_path / "cut" / csv).read_text()
    assert metadata(tmp_path / "cut")["training"]["resumed_from_episodes"] == [4]
    assert "resumed_from_episodes" not in metadata(tmp_path / "ref")["training"]
    assert not (tmp_path / "cut" / CHECKPOINT_DIR).exists()


def test_reprise_tabulaire_refuse_une_autre_configuration(tmp_path, monkeypatch):
    def failing_seed(seed, episode=0):
        if episode == 3:
            raise Interruption
        return training_episode_seed(seed, episode)

    monkeypatch.setattr(q_learning_module, "training_episode_seed", failing_seed)
    with pytest.raises(Interruption):
        q_learning_module.train_q_learning(output_dir=str(tmp_path), **TABULAR_RUN)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="params"):
        q_learning_module.train_q_learning(output_dir=str(tmp_path), resume=True, alpha=0.1,
                                           **TABULAR_RUN)
