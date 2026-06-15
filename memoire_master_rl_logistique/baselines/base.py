"""Interface commune aux politiques baseline (Section 4.7.1 du mémoire).

Toutes les baselines exposent la même signature ``predict(observation,
info, truck_location)`` afin que ``run_benchmark.py`` et
``evaluate_agent.py`` puissent les piloter de manière uniforme, et
encodent leurs actions (pelle, dump) avec le même schéma que
``MineEnv._decode_action``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class BaselinePolicy(Protocol):
    """Interface commune des politiques baseline."""

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str | None = None,
    ) -> int:
        """Retourne l'action discrète choisie pour le camion courant."""
        ...


def encode_action(shovel_idx: int, dump_idx: int, num_dumps: int) -> int:
    """Encode une paire (pelle, dump) en action discrète.

    Schéma identique à ``MineEnv._decode_action`` :
    ``shovel_idx = action // num_dumps`` et ``dump_idx = action % num_dumps``.
    """
    return shovel_idx * num_dumps + dump_idx
