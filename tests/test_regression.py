"""Test de non-régression (T4).

Rejoue Fixed et FIFO en nominal (graines 42 à 46) et compare aux références figées.

- Jusqu'à T12 bis, la référence était le CSV de l'article v1 : le code reproduisait
  exactement la v1.
- Depuis T13 (un flux aléatoire par processus), l'ordre des tirages change et tous
  les chiffres bougent, comme le plan l'annonçait. La référence est désormais
  ``fixtures/regression_reference.csv`` (script ``fixtures/generate_regression_reference.py``).

Si ce test échoue, soit une modification a changé la simulation sans le vouloir
(à corriger), soit le changement est voulu : régénérer la référence et le noter
dans le journal.
"""

import pytest
from reference_values import reference

from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS


@pytest.mark.parametrize("policy_name", ["Fixed", "FIFO"])
@pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
def test_non_regression(policy_name, seed):
    """Productivité, récompense et immobilisation identiques à la référence, à 1e-6 près."""
    scenario = SCENARIOS["nominal"]
    policy_fn = heuristic_policies(scenario)[policy_name]

    kpis = run_single_episode(make_env(scenario), policy_fn, seed)
    expected = reference(policy_name, seed)

    for key, value in expected.items():
        assert kpis[key] == pytest.approx(value, abs=1e-6), key
