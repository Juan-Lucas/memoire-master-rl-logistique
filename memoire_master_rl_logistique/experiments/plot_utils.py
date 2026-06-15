"""Utilitaires partagés pour les scripts de génération de figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, out_path: Path, dpi: int = 150) -> None:
    """Sauvegarde une figure, la ferme et affiche le chemin de sortie."""
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")