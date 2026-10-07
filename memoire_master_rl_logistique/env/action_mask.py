"""Masque des actions valides.

Espace d'action : Discrete(|P| × |D| + 1). Les |P| × |D| premières actions
encodent une paire (pelle, dump) ; la dernière est ATTENDRE.

Le masque actuel autorise toutes les actions, et il n'est appelé nulle part
dans le projet.
"""

from __future__ import annotations

import numpy as np

from memoire_master_rl_logistique.simulation.entities import DumpSite, Shovel


def compute_action_mask(
    shovels: list[Shovel],
    dumps: list[DumpSite],
    current_time_min: float,
    include_wait: bool = True,
) -> np.ndarray:
    """Renvoie un masque où toutes les actions, ATTENDRE comprise, sont valides."""
    n_pairs = len(shovels) * max(len(dumps), 1)
    n_actions = n_pairs + (1 if include_wait else 0)
    mask = np.ones(n_actions, dtype=np.int8)
    return mask
