"""Conventions communes aux figures de l'article.

- Ordre, libellés français et couleurs des politiques et des régimes, pour que
  toutes les figures restent cohérentes entre elles.
- Écriture des sorties sans écrasement accidentel (par exemple des figures de
  l'article v1 dans ``reports/figures``) : même principe que ``write_results`` (T5).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS

# Ordre de lecture : heuristiques, puis RL tabulaire, puis RL profond.
POLICY_ORDER = [
    "FIFO", "Fixed", "Nearest", "ShortestPath", "MTWT", "LP", "Q-Learning", "SARSA", "DQN",
    "PPO",
]

POLICY_LABELS = {
    "FIFO": "FIFO",
    "Fixed": "Affectation fixe",
    "Nearest": "Pelle la plus proche",
    "ShortestPath": "Plus court chemin",
    # Sigles des tableaux de l'article v3 (définis dans la section 5.7).
    "MTWT": "MTWT",
    "LP": "LP",
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

# Libellés anglais (article RL_article.tex) ; ``set_language("en")`` les met en place.
_LABELS = {
    "fr": {"policy": dict(POLICY_LABELS), "scenario": dict(SCENARIO_LABELS),
           "wait": dict(WAIT_LABELS)},
    "en": {
        "policy": {**POLICY_LABELS, "Fixed": "Fixed Assignment", "Nearest": "Nearest Shovel",
                   "ShortestPath": "Shortest Path"},
        "scenario": {"nominal": "Nominal", "high_load": "High load",
                     "high_breakdown": "Frequent breakdowns"},
        "wait": {"queue_wait_min_per_truck": "Queueing time per truck (min per shift)",
                 "avg_wait_min_per_truck": "Total idle time per truck (min per shift)"},
    },
}
LANGUAGES = tuple(_LABELS)
_language = "fr"


def set_language(lang: str) -> None:
    """Langue des figures : ``fr`` (article français, par défaut) ou ``en`` (article
    anglais). Les dictionnaires de libellés sont mis à jour sur place, pour que les
    modules qui les ont importés voient la même langue."""
    global _language
    if lang not in _LABELS:
        raise ValueError(f"Langue inconnue : {lang} (attendu : {LANGUAGES})")
    _language = lang
    for target, key in ((POLICY_LABELS, "policy"), (SCENARIO_LABELS, "scenario"),
                        (WAIT_LABELS, "wait")):
        target.clear()
        target.update(_LABELS[lang][key])


def tr(fr: str, en: str) -> str:
    """Texte d'une figure dans la langue choisie par ``set_language``."""
    return en if _language == "en" else fr


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
    """Libellé d'une politique dans la langue des figures (le nom brut si elle est inconnue)."""
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


# --- Courbes d'entraînement multi-graines (T30) ---------------------------------------

# Couleurs des agents, celles des figures v1.
AGENT_COLORS = {
    "Q-Learning": "#4CAF50",
    "SARSA": "#9C27B0",
    "DQN": "#2196F3",
    "PPO": "#FF5722",
}

# Suffixe des noms de fichiers de figures par régime (ex. learning_curves_highload.png).
SCENARIO_SUFFIXES = {"nominal": "nominal", "high_load": "highload",
                     "high_breakdown": "highbreakdown"}


def scenario_suffix(scenario: str) -> str:
    return SCENARIO_SUFFIXES.get(scenario, scenario)


def plot_band(ax: plt.Axes, curve: pd.DataFrame, color: str, label: str,
              x: str = "x", scale: float = 1.0, linestyle: str = "-") -> None:
    """Moyenne des graines d'entraînement (trait) et son IC à 95 % (bande).

    ``curve`` vient de ``analysis.learning_curves.learning_curve``. Les intervalles sans
    aucune valeur (pas d'épisode terminé, pas de ligne de journal) sont sautés : le trait
    relie les points voisins au lieu de s'interrompre.
    """
    curve = curve[curve["mean"].notna()]
    xs = curve[x] / scale
    n_runs = int(curve["n_runs"].max())
    ax.plot(xs, curve["mean"], color=color, linewidth=1.8, linestyle=linestyle,
            label=tr(f"{label} (moyenne de {n_runs} graines, IC 95 %)",
                     f"{label} (mean of {n_runs} seeds, 95% CI)"))
    band = curve["ci_low"].notna()
    ax.fill_between(xs[band], curve["ci_low"][band], curve["ci_high"][band], color=color,
                    alpha=0.2, linewidth=0)


def training_figures_parser(description: str) -> argparse.ArgumentParser:
    """Arguments communs des figures d'entraînement : modèles, sortie, régimes, graines."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("results_dir", type=Path, help="dossier des modèles (campagne)")
    parser.add_argument("out_dir", type=Path, help="dossier de sortie (créé si besoin)")
    parser.add_argument("--overwrite", action="store_true", help="autoriser l'écrasement")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIO_ORDER))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    return parser
