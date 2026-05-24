"""Masquage dynamique des actions invalides.

Conformément à la Section 3.4.2 du mémoire :
- Un camion ne peut être assigné qu'à une paire (pelle, dump) disponible.
- L'action ATTENDRE est toujours valide.

Le masquage empêche l'agent de choisir des actions irréalistes et accélère
l'apprentissage (cf. Costa & Ontañón, 2020).
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
    """Calcule le masque d'actions valides.

    Actions : [pelle_0×dump_0, ..., pelle_n-1×dump_m-1, ATTENDRE]
    Le masque vaut 1 si l'action est valide, 0 sinon.

    Toutes les paires (pelle, dump) sont considérées valides (l'agent
    peut choisir d'attendre la disponibilité). ATTENDRE est toujours valide.
    """
    n_pairs = len(shovels) * max(len(dumps), 1)
    n_actions = n_pairs + (1 if include_wait else 0)
    mask = np.ones(n_actions, dtype=np.int8)
    return mask
