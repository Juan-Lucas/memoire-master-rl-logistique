"""Compatibilité des artefacts entraînés avec la version de l'environnement."""

from __future__ import annotations

from pathlib import Path

MODEL_COMPAT_VERSION = "reward_wait_v2"
MODEL_COMPAT_FILENAME = "model_compat_version.txt"


def model_marker_path(model_path: str | Path) -> Path:
    """Retourne le chemin du marqueur de compatibilité pour un modèle."""
    path = Path(model_path)
    directory = path if path.is_dir() else path.parent
    return directory / MODEL_COMPAT_FILENAME


def is_model_compatible(model_path: str | Path) -> bool:
    """Vérifie si un modèle a été entraîné avec la version courante."""
    marker = model_marker_path(model_path)
    if not marker.exists():
        return False
    return marker.read_text(encoding="utf-8").strip() == MODEL_COMPAT_VERSION


def write_model_compat_marker(model_path: str | Path) -> None:
    """Écrit le marqueur de compatibilité dans le dossier du modèle."""
    marker = model_marker_path(model_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(MODEL_COMPAT_VERSION + "\n", encoding="utf-8")
