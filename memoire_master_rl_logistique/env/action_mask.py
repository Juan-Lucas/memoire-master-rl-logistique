"""Masquage dynamique des actions invalides.

Conformément à la Section 3.4.2 du mémoire :
- L'espace d'action est A = Discrete(|P| × |D| + 1)
- Chaque action encode une paire (pelle, dump) par division entière
- Si a_t = |P| × |D| : action ATTENDRE
- Toutes les paires (pelle, dump) sont considérées valides car l'agent
  peut choisir d'attendre la disponibilité des ressources
- L'action ATTENDRE est toujours valide

Cette approche simplifie l'espace de recherche tout en préservant
la flexibilité décisionnelle requise pour le dispatching minier.
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
    """Calcule le masque d'actions valides (Section 3.4.2 du mémoire).

    Conformément au mémoire, toutes les paires (pelle, dump) sont
    considérées valides. ATTENDRE est toujours valide.
    """
    n_pairs = len(shovels) * max(len(dumps), 1)
    n_actions = n_pairs + (1 if include_wait else 0)
    mask = np.ones(n_actions, dtype=np.int8)
    return mask
