"""Conventions communes aux figures de l'article.

- Ordre, libellés français et couleurs des politiques et des régimes, pour que
  toutes les figures restent cohérentes entre elles.
- Écriture des sorties sans écrasement accidentel (par exemple des figures de
  l'article v1 dans ``reports/figures``) : même principe que ``write_results`` (T5).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt

from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS

# Ordre de lecture : heuristiques, puis RL tabulaire, puis RL profond.
POLICY_ORDER = [
    "FIFO", "Fixed", "Nearest", "ShortestPath", "MTWT", "Q-Learning", "SARSA", "DQN", "PPO",
]

POLICY_LABELS = {
    "FIFO": "FIFO",
    "Fixed": "Affectation fixe",
    "Nearest": "Pelle la plus proche",
    "ShortestPath": "Plus court chemin",
    "MTWT": "Temps d'attente minimal",
    "Q-Learning": "Q-Learning",
    "SARSA": "SARSA",
    "DQN": "DQN",
    "PPO": "PPO",
}

SCENARIO_ORDER = MAIN_SCENARIOS
SCENARIO_LABELS = {
    "nominal": "Nominal",
    "high_load": "Forte charge",
    "high_breakdown": "Fortes pannes",
}
# Libellé de l'attente selon la colonne utilisée (voir analysis.robustness.wait_column).
WAIT_LABELS = {
    "queue_wait_min_per_truck": "Attente en file par camion (min par poste)",
    "avg_wait_min_per_truck": "Immobilisation totale par camion (min par poste)",
}

SCENARIO_COLORS = {
    "nominal": "#FF7043",
    "high_load": "#1A237E",
    "high_breakdown": "#64B5F6",
}


def order_policies(policies: Iterable[str]) -> list[str]:
    """Trie les politiques selon POLICY_ORDER ; les inconnues suivent, par ordre alphabétique.

    Une politique absente de POLICY_ORDER n'est jamais perdue silencieusement.
    """
    present = set(policies)
    known = [p for p in POLICY_ORDER if p in present]
    unknown = sorted(present - set(POLICY_ORDER))
    return known + unknown


def policy_label(policy: str) -> str:
    """Libellé français d'une politique (le nom brut si elle est inconnue)."""
    return POLICY_LABELS.get(policy, policy)


def prepare_outputs(out_dir: str | Path, filenames: Iterable[str], overwrite: bool) -> list[Path]:
    """Crée ``out_dir`` et renvoie les chemins de sortie.

    Lève ``FileExistsError`` si l'un des fichiers existe déjà et que ``overwrite``
    est faux, avant d'avoir écrit quoi que ce soit.
    """
    out_dir = Path(out_dir)
    paths = [out_dir / name for name in filenames]
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        names = ", ".join(str(p) for p in existing)
        raise FileExistsError(f"Fichiers déjà présents (utiliser --overwrite) : {names}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return paths


def save_figure(fig: plt.Figure, out_path: Path, dpi: int = 150) -> None:
    """Sauvegarde une figure, la ferme et affiche le chemin de sortie."""
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")
