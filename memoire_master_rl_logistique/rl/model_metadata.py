"""Traçabilité des modèles entraînés (T8, points N6 et N11).

À chaque entraînement, un ``metadata.json`` est écrit à côté du modèle : algorithme,
configuration de l'environnement (dont les poids de la récompense), taille de
l'observation, graine, paramètres d'entraînement, date et commit Git.

Au chargement, ``check_model_compatibility`` compare ces métadonnées à la
configuration demandée et refuse un modèle périmé : on ne peut plus évaluer, sans
le savoir, un agent entraîné avec une autre récompense ou une autre observation.

Ce module remplace ``model_version.py``, dont les fonctions n'étaient jamais appelées.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
from typing import Any
import warnings

METADATA_FILENAME = "metadata.json"
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]


class MissingMetadataError(FileNotFoundError):
    """Le dossier du modèle ne contient pas de ``metadata.json`` (modèle antérieur à T8)."""


class ModelCompatibilityError(ValueError):
    """Les métadonnées du modèle ne correspondent pas à la configuration demandée."""


@dataclass(frozen=True)
class EnvConfig:
    """Paramètres de l'environnement qui déterminent ce que l'agent a appris.

    Deux modèles ne sont comparables que s'ils partagent cette configuration et
    la même taille d'observation.
    """

    truck_count: int
    shovel_count: int
    dump_count: int
    episode_minutes: float
    breakdown_probability: float
    reward_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        # Types normalisés : 480 et 480.0, [1, 0.1] et (1.0, 0.1) donnent la même config.
        object.__setattr__(self, "truck_count", int(self.truck_count))
        object.__setattr__(self, "shovel_count", int(self.shovel_count))
        object.__setattr__(self, "dump_count", int(self.dump_count))
        object.__setattr__(self, "episode_minutes", float(self.episode_minutes))
        object.__setattr__(self, "breakdown_probability", float(self.breakdown_probability))
        object.__setattr__(self, "reward_weights", tuple(float(w) for w in self.reward_weights))

    @classmethod
    def from_env(cls, env: Any) -> EnvConfig:
        """Configuration d'un ``MineEnv`` existant (poids lus dans ``w1`` à ``w4``)."""
        return cls(
            truck_count=env.truck_count,
            shovel_count=env.shovel_count,
            dump_count=env.dump_count,
            episode_minutes=env.episode_minutes,
            breakdown_probability=env.breakdown_probability,
            reward_weights=(env.w1, env.w2, env.w3, env.w4),
        )

    @classmethod
    def from_scenario(cls, scenario: Any) -> EnvConfig:
        """Construit la configuration à partir d'un ``Scenario`` (ou d'un objet équivalent)."""
        return cls(
            truck_count=scenario.truck_count,
            shovel_count=scenario.shovel_count,
            dump_count=scenario.dump_count,
            episode_minutes=scenario.episode_minutes,
            breakdown_probability=scenario.breakdown_probability,
            reward_weights=scenario.reward_weights,
        )


@dataclass(frozen=True)
class ModelMetadata:
    """Contenu de ``metadata.json``."""

    algorithm: str
    env: EnvConfig
    observation_size: int
    seed: int
    training: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    git_dirty: bool | None = None
    created_at: str = ""
    schema_version: int = SCHEMA_VERSION
    # Durée de l'entraînement en secondes (T29) ; absente des modèles plus anciens.
    training_time_s: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> ModelMetadata:
        data = json.loads(text)
        return cls(env=EnvConfig(**data.pop("env")), **data)


def git_state(repo_dir: Path = REPO_ROOT) -> tuple[str | None, bool | None]:
    """Commit courant et présence de modifications non commitées (``None`` si Git est absent).

    ``git_dirty = True`` signale qu'un modèle a été entraîné avec du code qui ne
    correspond exactement à aucun commit : c'est précisément ce qui rendait N11
    impossible à prouver.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def build_metadata(
    algorithm: str,
    env: EnvConfig,
    observation_size: int,
    seed: int,
    training: dict[str, Any] | None = None,
    training_time_s: float | None = None,
) -> ModelMetadata:
    """Métadonnées d'un modèle qui vient d'être entraîné, avec la date et l'état Git."""
    commit, dirty = git_state()
    return ModelMetadata(
        algorithm=algorithm,
        env=env,
        observation_size=int(observation_size),
        seed=int(seed),
        training=dict(training or {}),
        git_commit=commit,
        git_dirty=dirty,
        created_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        training_time_s=None if training_time_s is None else round(float(training_time_s), 3),
    )


def write_metadata(model_dir: str | Path, metadata: ModelMetadata) -> Path:
    """Écrit ``metadata.json`` dans le dossier du modèle et renvoie son chemin."""
    path = Path(model_dir) / METADATA_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metadata.to_json() + "\n", encoding="utf-8")
    return path


def read_metadata(model_dir: str | Path) -> ModelMetadata:
    """Lit ``metadata.json`` ; lève ``MissingMetadataError`` s'il n'existe pas."""
    path = Path(model_dir) / METADATA_FILENAME
    if not path.exists():
        raise MissingMetadataError(f"Pas de {METADATA_FILENAME} dans {path.parent}")
    return ModelMetadata.from_json(path.read_text(encoding="utf-8"))


def _differences(
    metadata: ModelMetadata, algorithm: str, env: EnvConfig, observation_size: int
) -> list[str]:
    """Liste lisible des écarts entre le modèle et la configuration demandée."""
    diffs = []
    if metadata.algorithm != algorithm:
        diffs.append(f"algorithm : modèle {metadata.algorithm!r}, demandé {algorithm!r}")
    if metadata.observation_size != observation_size:
        diffs.append(
            f"observation_size : modèle {metadata.observation_size}, demandé {observation_size}"
        )
    for name, expected in asdict(env).items():
        actual = getattr(metadata.env, name)
        if name == "reward_weights":
            same = len(actual) == len(expected) and all(
                math.isclose(a, e, abs_tol=1e-12) for a, e in zip(actual, expected)
            )
        elif isinstance(expected, float):
            same = math.isclose(actual, expected, abs_tol=1e-12)
        else:
            same = actual == expected
        if not same:
            diffs.append(f"{name} : modèle {actual!r}, demandé {expected!r}")
    return diffs


def check_model_compatibility(
    model_dir: str | Path,
    algorithm: str,
    env: EnvConfig,
    observation_size: int,
    allow_legacy: bool = False,
) -> ModelMetadata | None:
    """Vérifie qu'un modèle peut être évalué avec la configuration demandée.

    - Métadonnées absentes : ``MissingMetadataError``, sauf si ``allow_legacy`` est
      vrai (avertissement, renvoie ``None``) — pour réévaluer en connaissance de
      cause les modèles de l'article v1, antérieurs à T8.
    - Écart sur l'algorithme, la configuration ou la taille d'observation :
      ``ModelCompatibilityError`` listant chaque écart.

    La graine et le commit ne sont pas comparés : on évalue sur d'autres graines,
    et un nouveau commit ne rend pas un modèle faux.
    """
    try:
        metadata = read_metadata(model_dir)
    except MissingMetadataError:
        if not allow_legacy:
            raise
        warnings.warn(
            f"Modèle sans métadonnées chargé sans contrôle (allow_legacy) : {model_dir}",
            stacklevel=2,
        )
        return None

    diffs = _differences(metadata, algorithm, env, observation_size)
    if diffs:
        raise ModelCompatibilityError(
            f"Modèle incompatible ({model_dir}) :\n  - " + "\n  - ".join(diffs)
        )
    return metadata
