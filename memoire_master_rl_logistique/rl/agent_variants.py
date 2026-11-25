"""Variantes d'agents réentraînées après la campagne (rapport du 17/11/2026, M5, M6, T17).

Chaque variante change un seul élément de PPO ou de DQN, en régime nominal, et se compare
aux modèles nominaux de la campagne :

- ``drop_route_wait`` (T17, P2) : l'observation perd le bloc ``route_wait``, les 6 attentes
  prévues par route calculées comme pour MTWT (23 → 17 composantes). Mesure ce que PPO
  apprend sans l'estimation de MTWT ;
- ``no_wait_action`` (M6) : l'action ATTENDRE est retirée (7 → 6 actions) ;
- ``double_q`` (M6) : Double DQN, l'action suivante est choisie par le réseau en ligne et
  évaluée par le réseau cible (van Hasselt et al., 2016), avec le facteur γ_min^Δt de chaque
  transition comme dans la campagne ;
- ``fuel_in_reward`` (M5) : le terme pondéré par w3 porte le carburant du cycle au lieu de
  sa distance (``MineEnv.fuel_cost_scale_l``) ;
- ``reward_weights`` (M5) : autres poids, pour la sensibilité à w3.

Les variantes sont décrites dans ``configs/experiment_config.yaml`` (``retraining``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import torch as th
from torch.nn import functional as F

from memoire_master_rl_logistique.config import experiment_config
from memoire_master_rl_logistique.env.observation_builder import observation_layout


@dataclass(frozen=True)
class AgentVariant:
    """Une variante d'agent : ce qui change par rapport à l'agent de la campagne."""

    name: str
    algorithm: str
    reward_weights: tuple[float, ...]
    drop_route_wait: bool = False
    no_wait_action: bool = False
    double_q: bool = False
    fuel_in_reward: bool = False

    def __post_init__(self) -> None:
        if self.algorithm not in ("PPO", "DQN"):
            raise ValueError(f"Variante {self.name!r} : algorithme {self.algorithm!r} (PPO ou DQN)")
        if self.double_q and self.algorithm != "DQN":
            raise ValueError(f"Variante {self.name!r} : double_q suppose DQN")
        object.__setattr__(self, "reward_weights", tuple(float(w) for w in self.reward_weights))

    @property
    def needs_wrapper(self) -> bool:
        return self.drop_route_wait or self.no_wait_action

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "algorithm": self.algorithm,
                "reward_weights": list(self.reward_weights),
                "drop_route_wait": self.drop_route_wait, "no_wait_action": self.no_wait_action,
                "double_q": self.double_q, "fuel_in_reward": self.fuel_in_reward}


def retraining_config() -> dict[str, Any]:
    return experiment_config()["retraining"]


def load_variants(config: dict[str, Any] | None = None) -> dict[str, AgentVariant]:
    """Variantes de ``retraining.variants`` ; poids nominaux sauf s'ils sont donnés."""
    config = config or experiment_config()
    nominal = config["scenario_defaults"]["reward_weights"]
    variants = {}
    for name, values in config["retraining"]["variants"].items():
        values = {"reward_weights": nominal, **values}
        try:
            variants[name] = AgentVariant(name=name, **values)
        except TypeError as error:
            raise ValueError(f"Variante {name!r} mal décrite : {error}") from None
    return variants


def kept_observation_indices(num_shovels: int = 3, num_dumps: int = 2,
                             drop_route_wait: bool = True) -> np.ndarray:
    """Indices de l'observation gardés par la variante (toutes sauf ``route_wait``)."""
    layout = observation_layout(num_shovels, num_dumps)
    size = layout["time"].stop
    dropped = set(range(size)[layout["route_wait"]]) if drop_route_wait else set()
    return np.array([i for i in range(size) if i not in dropped], dtype=np.int64)


class DropRouteWait(gym.ObservationWrapper):
    """Observation sans les attentes prévues par route (T17)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        unwrapped = env.unwrapped
        self.kept = kept_observation_indices(unwrapped.shovel_count, unwrapped.dump_count)
        space = env.observation_space
        self.observation_space = gym.spaces.Box(low=space.low[self.kept],
                                                high=space.high[self.kept], dtype=space.dtype)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[self.kept]


class NoWaitAction(gym.ActionWrapper):
    """Actions sans ATTENDRE (M6) : les 6 couples pelle–verse, mêmes indices que l'env."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.wait_action = env.action_space.n - 1  # ATTENDRE est la dernière action
        self.action_space = gym.spaces.Discrete(self.wait_action)

    def action(self, action: Any) -> int:
        action = int(action)
        if not 0 <= action < self.wait_action:
            raise ValueError(f"Action {action} hors de [0, {self.wait_action - 1}]")
        return action


def wrap_env(env: gym.Env, variant: AgentVariant | None) -> gym.Env:
    """Applique à ``env`` les wrappers de la variante (aucun si ``None``)."""
    if variant is None:
        return env
    if variant.drop_route_wait:
        env = DropRouteWait(env)
    if variant.no_wait_action:
        env = NoWaitAction(env)
    return env


def variant_policy_fn(model: Any, variant: AgentVariant) -> Callable[[Any, dict, Any], int]:
    """Politique d'évaluation sur un ``MineEnv`` non enveloppé : l'observation complète est
    réduite comme à l'entraînement ; les actions de NoWaitAction ont les indices de l'env."""
    kept = kept_observation_indices() if variant.drop_route_wait else None

    def policy_fn(obs, info, env):  # noqa: ANN001, ANN202
        if kept is not None:
            obs = obs[kept]
        return int(model.predict(obs, deterministic=True)[0])
    return policy_fn


class DoubleDQN(DQN):
    """Double DQN : cible r + (1 - done) · γ_t · Q_cible(s', argmax_a Q_en ligne(s', a)).

    Identique à ``DQN.train`` de Stable-Baselines3 2.8 sauf le choix de l'action suivante ;
    ``replay_data.discounts`` porte γ_min^Δt de chaque transition (``TimeDiscountReplayBuffer``).
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            with th.no_grad():
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                next_q_values = th.gather(self.q_net_target(replay_data.next_observations),
                                          dim=1, index=next_actions)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = th.gather(self.q_net(replay_data.observations), dim=1,
                                         index=replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
