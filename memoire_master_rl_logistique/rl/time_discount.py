"""Actualisation par minute : formulation semi-markovienne (décision du 21 octobre 2026).

**Le problème.** Les quatre algorithmes actualisent les récompenses futures par γ **à
chaque décision**. Mais une décision dure de 0,2 min (ATTENDRE) à plus de 30 min (un
cycle vers la pelle la plus éloignée). L'agent compte donc le futur en nombre de
décisions, pas en minutes. Deux défauts mesurés en découlent (``docs/journal_corrections.md``,
19 et 21 octobre) :

- un trajet long ne retarde pas la production suivante, en décisions : DQN, Q-Learning
  et SARSA dérivaient vers la pelle la plus éloignée ;
- quand un cycle rapportait en moyenne une valeur négative (option B, abandonnée),
  enchaîner des ATTENDRE de 0,2 min repoussait ce coût de nombreuses décisions pour
  quelques minutes seulement : PPO attendait tout le temps.

**La correction.** Une récompense obtenue après une décision qui a duré Δt minutes est
actualisée par γ_min^Δt (Bradtke et Duff, 1995 ; Sutton, Precup et Singh, 1999). Le
temps qui compte devient celui de la mine. La récompense, elle, ne change pas.

**Aucun paramètre nouveau.** γ_min = γ^(1 / ``REFERENCE_DECISION_INTERVAL_MIN``), où γ est
la valeur de chaque algorithme (``configs/``) et 1,89 min l'intervalle moyen entre deux
décisions de l'Affectation fixe en nominal. En nominal, l'horizon reste en moyenne celui
d'avant (γ = 0,995 : environ 200 décisions, soit environ 377 min). Il ne dépend plus de
la taille de la flotte.

**Ce qui est actualisé, et les approximations, déclarées :**

- Q-Learning et SARSA : cible r + γ_min^Δt · Q(s′, ·) (``step_discount``) ;
- DQN : la mémoire de rejeu garde γ_min^Δt de chaque transition
  (``TimeDiscountReplayBuffer``), et Stable-Baselines3 l'utilise dans la cible ;
- PPO : avantages GAE avec un facteur γ_min^Δt par pas (``TimeDiscountRolloutBuffer``).
  Le prolongement de la valeur à la coupure de fin de poste (Stable-Baselines3 ajoute
  γ · V(s_final) à la récompense) utilise aussi γ_min^Δt (``TimeDiscountCallback``) ;
- **approximation 1** : la récompense d'un pas (cycles terminés pendant la décision) est
  comptée au début du pas, sans actualisation à l'intérieur du pas (usuel en SMDP à
  récompense par transition) ;
- **approximation 2** : pour PPO, la normalisation de la récompense (``VecNormalize``)
  estime l'échelle des retours avec le γ par décision. Elle ne sert qu'à mettre la
  récompense à l'échelle, pas à la cible.

Ce module ne touche ni à l'environnement ni à la récompense : il lit seulement
``info["elapsed_min"]``, la durée de chaque décision donnée par ``MineEnv``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

# Intervalle moyen entre deux décisions de l'Affectation fixe en nominal (480 min / nombre
# de décisions, 30 graines d'évaluation) : vérifié par tests/test_reward.py.
REFERENCE_DECISION_INTERVAL_MIN = 1.89
ELAPSED_KEY = "elapsed_min"


def per_minute_gamma(gamma: float) -> float:
    """γ_min tel qu'une décision de durée de référence soit actualisée par ``gamma``."""
    if not 0 < gamma <= 1:
        raise ValueError(f"gamma doit être dans ]0, 1] : {gamma}")
    return gamma ** (1.0 / REFERENCE_DECISION_INTERVAL_MIN)


def step_discount(gamma_per_minute: float, elapsed_min: float) -> float:
    """Facteur d'actualisation d'une décision qui a duré ``elapsed_min`` minutes."""
    if elapsed_min < 0:
        raise ValueError(f"Durée négative : {elapsed_min}")
    return gamma_per_minute ** elapsed_min


def transition_discount(gamma: float, gamma_per_minute: float | None,
                        info: dict[str, Any]) -> float:
    """Facteur d'une transition : ``gamma`` (par décision) si ``gamma_per_minute`` vaut
    ``None``, sinon γ_min^Δt avec la durée de la décision lue dans ``info``."""
    if gamma_per_minute is None:
        return gamma
    return step_discount(gamma_per_minute, _elapsed(info))


def _elapsed(info: dict[str, Any]) -> float:
    if ELAPSED_KEY not in info:
        raise KeyError(f"info[{ELAPSED_KEY!r}] manquant : l'environnement doit donner la "
                       "durée de chaque décision (MineEnv).")
    return float(info[ELAPSED_KEY])


class TimeDiscountReplayBuffer(ReplayBuffer):
    """Mémoire de rejeu de DQN qui garde, pour chaque transition, γ_min^Δt.

    Stable-Baselines3 (≥ 2.4) utilise ``ReplayBufferSamples.discounts`` à la place de γ
    dans la cible de DQN quand ce champ est rempli.
    """

    def __init__(self, *args: Any, gamma_per_minute: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gamma_per_minute = gamma_per_minute
        self.discounts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:  # noqa: ANN001
        # Écrit à la position courante, avant que ``super().add`` ne l'avance.
        self.discounts[self.pos] = [step_discount(self.gamma_per_minute, _elapsed(info))
                                    for info in infos]
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray,
                     env: VecNormalize | None = None) -> ReplayBufferSamples:
        # Même tirage de l'environnement que la classe de base (un seul environnement ici).
        if self.n_envs != 1:
            raise ValueError("L'actualisation par minute de DQN suppose n_envs = 1.")
        samples = super()._get_samples(batch_inds, env)
        discounts = self.to_torch(self.discounts[batch_inds, 0].reshape(-1, 1))
        return samples._replace(discounts=discounts)


class TimeDiscountRolloutBuffer(RolloutBuffer):
    """Tampon de PPO dont les avantages GAE utilisent γ_min^Δt à chaque pas.

    δ_t = r_t + d_t · V(s_{t+1}) · (1 − fin_t) − V(s_t) et
    A_t = δ_t + d_t · λ · (1 − fin_t) · A_{t+1}, avec d_t = γ_min^Δt de la décision t.
    Avec Δt constant égal à la durée de référence, on retrouve exactement le GAE usuel.
    Le facteur de chaque pas est écrit par ``TimeDiscountCallback`` avant l'ajout du pas.
    """

    def __init__(self, *args: Any, gamma_per_minute: float, **kwargs: Any) -> None:
        self.gamma_per_minute = gamma_per_minute
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.discounts = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)

    def set_step_discounts(self, infos: list[dict[str, Any]]) -> np.ndarray:
        """Facteurs du pas en cours (position ``pos``), lus dans les ``infos``."""
        discounts = np.array([step_discount(self.gamma_per_minute, _elapsed(info))
                              for info in infos], dtype=np.float32)
        self.discounts[self.pos] = discounts
        return discounts

    def compute_returns_and_advantage(self, last_values, dones: np.ndarray) -> None:  # noqa: ANN001
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            discount = self.discounts[step]
            delta = self.rewards[step] + discount * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + discount * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values


class TimeDiscountCallback(BaseCallback):
    """Pour PPO : écrit le facteur γ_min^Δt de chaque pas dans le tampon, et règle le γ
    que Stable-Baselines3 utilise pour prolonger la valeur à la coupure de fin de poste.

    Stable-Baselines3 appelle les callbacks après ``env.step`` et **avant** de prolonger la
    valeur et d'ajouter le pas au tampon : c'est le seul moment où les ``infos`` du pas
    sont disponibles. Un seul environnement (``n_envs = 1``) : le γ de prolongement est
    un scalaire.

    ``gamma`` : le γ de la configuration. ``model.gamma`` change à chaque pas pendant
    l'entraînement (une sauvegarde en cours de route en garde une valeur quelconque) ; il
    est remis à ``gamma`` à la fin, pour que le modèle enregistré porte la configuration.
    """

    def __init__(self, gamma: float) -> None:
        super().__init__(verbose=0)
        self._gamma = gamma

    def _on_training_start(self) -> None:
        if self.model.n_envs != 1:
            raise ValueError("L'actualisation par minute de PPO suppose n_envs = 1.")
        if not isinstance(self.model.rollout_buffer, TimeDiscountRolloutBuffer):
            raise TypeError("PPO doit utiliser TimeDiscountRolloutBuffer.")

    def _on_step(self) -> bool:
        discounts = self.model.rollout_buffer.set_step_discounts(self.locals["infos"])
        self.model.gamma = float(discounts[0])
        return True

    def _on_training_end(self) -> None:
        self.model.gamma = self._gamma  # γ de la configuration, enregistré avec le modèle
