"""Sauvegardes en cours d'entraînement et reprise après une interruption.

Un entraînement complet dure jusqu'à 2 h 45. Deux essais de la mise au point des agents
ont été perdus, arrêtés à 59 % et 97 % faute de mémoire sur la machine, sans rien à
évaluer. Ce module enregistre régulièrement l'état de l'entraînement dans
``<dossier du modèle>/checkpoint/`` et permet de reprendre là où il s'était arrêté.

**Agents tabulaires (Q-Learning, SARSA) : reprise exacte.** La sauvegarde, faite entre
deux épisodes, contient tout ce dont dépend la suite : Q-table, état du générateur
aléatoire, ε, journaux. Chaque épisode part de sa propre graine
(``training_episode_seed``). Un entraînement repris donne donc la même Q-table qu'un
entraînement sans interruption (vérifié par ``tests/test_checkpoints.py``).

**Agents profonds (PPO, DQN) : reprise proche, mais pas identique, et déclarée.** La
sauvegarde, faite à la fin d'une collecte, contient le modèle, la mémoire de rejeu (DQN),
les statistiques de normalisation de la récompense (PPO) et les journaux des épisodes
terminés. Trois choses ne sont pas restaurées :

- l'épisode en cours, qui est rejoué depuis le début, avec sa graine ;
- l'état des générateurs aléatoires de PyTorch ;
- la collecte en cours de PPO.

Le modèle final n'est donc pas celui qu'aurait donné un entraînement sans interruption.
Chaque reprise est écrite dans ``metadata.json`` (``resumed_from_timesteps``), pour être
déclarée. La reprise n'est possible qu'avec un seul environnement (``n_envs = 1``, valeur
de la campagne) : avec plusieurs, les épisodes sont entrelacés entre les environnements.

**Garde-fous :**

- les fichiers sont écrits sous un nom temporaire, puis renommés, et ``state.json`` en
  dernier : une sauvegarde interrompue n'est jamais prise pour une sauvegarde complète ;
- une reprise refuse une configuration différente de celle de la sauvegarde
  (algorithme, graine, budget, hyperparamètres, environnement) ;
- un nouvel entraînement refuse de démarrer s'il trouve la sauvegarde d'un entraînement
  interrompu, sauf ``overwrite`` ;
- à la fin d'un entraînement complet, la sauvegarde est supprimée, puisque le modèle
  final la remplace.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import pickle
import shutil
import time
from typing import Any

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from memoire_master_rl_logistique.rl.callbacks import KPILoggerCallback

CHECKPOINT_DIR = "checkpoint"
STATE_FILE = "state.json"
MODEL_FILE = "model.zip"
REPLAY_BUFFER_FILE = "replay_buffer.pkl"
VECNORMALIZE_FILE = "vecnormalize.pkl"
KPI_FILE = "training_kpis.csv"
TABULAR_STATE_FILE = "state.pkl"

# Fréquences par défaut : 20 sauvegardes pour 2 000 000 pas, 30 pour 30 000 épisodes.
DEFAULT_CHECKPOINT_FREQ = 100_000  # pas (PPO, DQN)
DEFAULT_CHECKPOINT_EVERY = 1_000   # épisodes (Q-Learning, SARSA)


def checkpoint_dir(out_path: str | Path) -> Path:
    """Dossier de sauvegarde d'un entraînement écrit dans ``out_path``."""
    return Path(out_path) / CHECKPOINT_DIR


def training_signature(**fields: Any) -> dict[str, Any]:
    """Ce qui doit être identique pour reprendre un entraînement, sous forme JSON.

    Les tuples deviennent des listes, pour que la signature écrite dans ``state.json``
    et celle recalculée à la reprise se comparent directement.
    """
    return json.loads(json.dumps(fields))


def prepare_checkpoint(out_path: str | Path, resume: bool, overwrite: bool) -> Path:
    """Vérifie qu'on peut démarrer (ou reprendre) l'entraînement écrit dans ``out_path``.

    - ``resume`` sans sauvegarde : ``FileNotFoundError`` ;
    - sans ``resume``, avec la sauvegarde d'un entraînement interrompu : ``FileExistsError``,
      sauf si ``overwrite`` (la sauvegarde est alors supprimée).

    Renvoie le dossier de sauvegarde.
    """
    folder = checkpoint_dir(out_path)
    present = (folder / STATE_FILE).exists()
    if resume and not present:
        raise FileNotFoundError(f"Aucune sauvegarde à reprendre dans {folder}")
    if not resume and folder.exists():
        if not overwrite:
            raise FileExistsError(
                f"Sauvegarde d'un entraînement interrompu dans {folder} : passer resume=True "
                "pour le reprendre, ou overwrite=True pour recommencer."
            )
        shutil.rmtree(folder)
    return folder


def check_signature(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Refuse de reprendre avec une configuration différente de celle de la sauvegarde."""
    if dict(saved) != dict(current):
        flat_saved, flat_current = _flatten(saved), _flatten(current)
        changed = sorted(k for k in set(flat_saved) | set(flat_current)
                         if flat_saved.get(k) != flat_current.get(k))
        raise ValueError(
            f"La configuration diffère de celle de la sauvegarde (champs : {', '.join(changed)}) : "
            "reprise refusée."
        )


def _flatten(fields: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """{"params": {"gamma": 0.99}} → {"params.gamma": 0.99}, pour nommer le champ qui diffère."""
    flat = {}
    for key, value in fields.items():
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def clear_checkpoint(out_path: str | Path) -> None:
    """Supprime la sauvegarde, une fois le modèle final enregistré."""
    shutil.rmtree(checkpoint_dir(out_path), ignore_errors=True)


def _write_state(folder: Path, state: Mapping[str, Any]) -> None:
    tmp = folder / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, folder / STATE_FILE)


def read_state(out_path: str | Path) -> dict[str, Any]:
    """Contenu de ``state.json`` : pas écoulés, épisodes terminés, signature, durée, reprises."""
    return json.loads((checkpoint_dir(out_path) / STATE_FILE).read_text(encoding="utf-8"))


# --- PPO, DQN ----------------------------------------------------------------------------

class CheckpointCallback(BaseCallback):
    """Sauvegarde l'entraînement tous les ``freq`` pas, à la fin d'une collecte.

    La fin d'une collecte est le seul moment où l'état de PPO est complet (pas de collecte
    à moitié faite) ; pour DQN, elle arrive tous les ``train_freq`` pas.
    """

    def __init__(
        self,
        folder: Path,
        freq: int,
        kpi_callback: KPILoggerCallback,
        signature: Mapping[str, Any],
        elapsed_before_s: float = 0.0,
        resumed_from: list[int] | None = None,
    ) -> None:
        super().__init__(verbose=0)
        if freq < 1:
            raise ValueError(f"La fréquence de sauvegarde doit être ≥ 1 : {freq}")
        self.folder = folder
        self.freq = freq
        self.kpi_callback = kpi_callback
        self.signature = dict(signature)
        self.elapsed_before_s = elapsed_before_s
        self.resumed_from = list(resumed_from or [])
        self._last_saved = 0
        self._start = time.perf_counter()

    def _on_training_start(self) -> None:
        self._last_saved = self.num_timesteps
        self._start = time.perf_counter()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_saved >= self.freq:
            self.save()

    def elapsed_s(self) -> float:
        """Durée d'entraînement cumulée, reprises comprises (sans les temps d'arrêt)."""
        return self.elapsed_before_s + time.perf_counter() - self._start

    def save(self) -> None:
        """Écrit les fichiers sous un nom temporaire, les renomme, puis ``state.json``."""
        self.folder.mkdir(parents=True, exist_ok=True)
        renames = []

        tmp = self.folder / "model.tmp.zip"
        self.model.save(str(tmp))
        renames.append((tmp, self.folder / MODEL_FILE))
        if isinstance(self.model, OffPolicyAlgorithm):
            tmp = self.folder / "replay_buffer.tmp.pkl"
            self.model.save_replay_buffer(str(tmp))
            renames.append((tmp, self.folder / REPLAY_BUFFER_FILE))
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            tmp = self.folder / (VECNORMALIZE_FILE + ".tmp")
            env.save(str(tmp))
            renames.append((tmp, self.folder / VECNORMALIZE_FILE))
        tmp = self.folder / (KPI_FILE + ".tmp")
        self.kpi_callback.write(tmp)
        renames.append((tmp, self.folder / KPI_FILE))

        for src, dst in renames:
            os.replace(src, dst)
        _write_state(self.folder, {
            "num_timesteps": int(self.num_timesteps),
            "episodes": self.kpi_callback.episodes,
            "elapsed_s": self.elapsed_s(),
            "resumed_from_timesteps": self.resumed_from,
            "signature": self.signature,
        })
        self._last_saved = self.num_timesteps


def load_sb3_checkpoint(
    algorithm: type[BaseAlgorithm], out_path: str | Path, env: Any, signature: Mapping[str, Any],
) -> tuple[BaseAlgorithm, Any, dict[str, Any]]:
    """Recharge modèle, mémoire de rejeu et normalisation de la récompense.

    ``env`` est l'environnement d'entraînement **sans** ``VecNormalize`` : si la
    sauvegarde en contient, il est recréé autour de ``env`` avec les statistiques
    enregistrées. Renvoie ``(modèle, environnement, état)``.
    """
    folder = checkpoint_dir(out_path)
    state = read_state(out_path)
    check_signature(state["signature"], signature)

    if (folder / VECNORMALIZE_FILE).exists():
        env = VecNormalize.load(str(folder / VECNORMALIZE_FILE), env)
        env.training = True
    model = algorithm.load(str(folder / MODEL_FILE), env=env)
    if (folder / REPLAY_BUFFER_FILE).exists():
        model.load_replay_buffer(str(folder / REPLAY_BUFFER_FILE))
    if model.num_timesteps != state["num_timesteps"]:
        raise RuntimeError(
            f"Sauvegarde incohérente dans {folder} : modèle à {model.num_timesteps} pas, "
            f"state.json à {state['num_timesteps']}."
        )
    return model, env, state


def resume_sb3(
    algorithm: type[BaseAlgorithm],
    out_path: str | Path,
    env: Any,
    signature: Mapping[str, Any],
    train_seed: int,
    kpi_callback: KPILoggerCallback,
) -> tuple[BaseAlgorithm, Any, dict[str, Any]]:
    """Reprise complète d'un entraînement PPO ou DQN interrompu.

    Recharge la sauvegarde (``load_sb3_checkpoint``), fait rejouer l'épisode interrompu
    depuis sa graine et remet les épisodes déjà terminés dans le journal. Renvoie
    ``(modèle, environnement, état)`` ; l'état contient la liste des reprises, celle-ci
    comprise (``resumed_from_timesteps``).
    """
    # Import local : env_factory importe l'environnement, inutile pour les agents tabulaires.
    from memoire_master_rl_logistique.rl.env_factory import reseed_training_vec_env

    model, env, state = load_sb3_checkpoint(algorithm, out_path, env, signature)
    # ``load`` ré-ensemence l'environnement avec la graine du modèle (comme à la création,
    # T26) : on le remet sur le premier épisode non terminé.
    reseed_training_vec_env(env, train_seed, state["episodes"])
    kpi_callback.restore(checkpoint_dir(out_path) / KPI_FILE)
    state["resumed_from_timesteps"] = [*state["resumed_from_timesteps"], state["num_timesteps"]]
    return model, env, state


# --- Q-Learning, SARSA -------------------------------------------------------------------

def save_tabular_checkpoint(folder: Path, state: Mapping[str, Any]) -> None:
    """Enregistre l'état complet d'un entraînement tabulaire (``state.pkl``), puis ``state.json``.

    ``state`` contient au moins ``next_episode``, ``signature`` et ``elapsed_s``.
    """
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / (TABULAR_STATE_FILE + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(dict(state), f)
    os.replace(tmp, folder / TABULAR_STATE_FILE)
    _write_state(folder, {
        "episodes": state["next_episode"],
        "elapsed_s": state["elapsed_s"],
        "signature": state["signature"],
    })


def load_tabular_checkpoint(out_path: str | Path, signature: Mapping[str, Any]) -> dict[str, Any]:
    """État enregistré par ``save_tabular_checkpoint``, après vérification de la signature."""
    folder = checkpoint_dir(out_path)
    with (folder / TABULAR_STATE_FILE).open("rb") as f:
        state = pickle.load(f)  # noqa: S301 (fichier écrit par ce module)
    check_signature(state["signature"], signature)
    if state["next_episode"] != read_state(out_path)["episodes"]:
        raise RuntimeError(f"Sauvegarde incohérente dans {folder} : state.pkl et state.json diffèrent.")
    return state
