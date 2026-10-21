"""Tests de T20 : représentation tabulaire documentée et couverture de la Q-table."""

import numpy as np
import pytest

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.env.observation_builder import observation_layout
from memoire_master_rl_logistique.experiments.policies import (
    agent_policy_fn,
    heuristic_policies,
    make_env,
)
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.discretize import (
    DEFAULT_N_BINS,
    WAIT_SATURATION_MIN,
    WAIT_SCALE,
    _discretize_obs,
    state_space_size,
)
from memoire_master_rl_logistique.rl.tabular_training import DEFAULT_ACTION, TabularPolicy

LAYOUT = observation_layout(3, 2)


def test_valeurs_inchangees():
    """Les constantes nommées gardent les valeurs d'entraînement (sinon il faudrait réentraîner)."""
    assert WAIT_SCALE == 50.0 and DEFAULT_N_BINS == 8 and DEFAULT_ACTION == 0
    assert WAIT_SATURATION_MIN == pytest.approx(9.6)


def test_taille_de_l_espace_d_etats():
    """« Savoir expliquer » : 8⁵ attentes discrétisées × 3 pelles possibles = 98 304 états."""
    assert state_space_size() == 8**5 * 3 == 98_304


def test_discretisation_identique_a_l_ancienne_formule():
    """Sur des observations aléatoires, l'état est le même qu'avec les constantes écrites en dur."""
    rng = np.random.default_rng(0)
    for obs in rng.random((200, 18)).astype(np.float32):
        waits = np.clip(np.concatenate([obs[LAYOUT["shovel_wait"]], obs[LAYOUT["dump_wait"]]]) * 50, 0, 1)
        expected = (*np.minimum((waits * 8).astype(int), 7), int(np.argmin(obs[LAYOUT["distance_to_shovel"]])))

        assert _discretize_obs(obs) == expected


@pytest.mark.parametrize("trucks", [12, 18])
def test_etat_independant_du_nombre_de_camions(trucks):
    """Au début du poste, l'état discret ne dépend pas du nombre de camions."""
    obs, _ = MineEnv(truck_count=trucks).reset(seed=42)

    assert _discretize_obs(obs) == (0, 0, 0, 0, 0, 0)


def test_couverture_etats_connus_et_action_par_defaut():
    """État connu : action apprise ; état inconnu : action par défaut, comptée."""
    known_obs, _ = MineEnv().reset(seed=42)
    policy = TabularPolicy({_discretize_obs(known_obs): np.array([0, 0, 5.0, 0, 0, 0, 0])})
    unknown_obs = known_obs.copy()
    unknown_obs[LAYOUT["shovel_wait"]] = 0.02  # attente de 9,6 min : intervalle 7, jamais vu

    assert policy.predict(known_obs) == 2
    assert policy.predict(unknown_obs) == DEFAULT_ACTION
    assert policy.coverage() == {"known_state_pct": 50.0, "default_action_pct": 50.0}
    policy.reset_coverage()
    assert policy.coverage() == {"known_state_pct": 0.0, "default_action_pct": 0.0}


def test_couverture_rapportee_par_episode():
    """Le benchmark rapporte, pour Q-Learning et SARSA, la part d'états connus de chaque épisode."""
    known_obs, _ = MineEnv().reset(seed=42)
    agent = TabularPolicy({_discretize_obs(known_obs): np.zeros(7)})
    policy_fn = agent_policy_fn(agent, "Q-Learning")
    scenario = SCENARIOS["nominal"]

    first = run_single_episode(make_env(scenario), policy_fn, seed=42)
    second = run_single_episode(make_env(scenario), policy_fn, seed=43)

    for kpis in (first, second):
        assert kpis["known_state_pct"] + kpis["default_action_pct"] == pytest.approx(100.0)
        assert 0 < kpis["known_state_pct"] < 100  # seul l'état du début du poste est connu
    assert agent.decisions == second["decisions"]  # remis à zéro à chaque épisode


def test_pas_de_couverture_pour_les_autres_politiques():
    """Les heuristiques et les agents profonds n'ont pas de Q-table : pas de colonne de couverture."""
    kpis = run_single_episode(make_env(SCENARIOS["nominal"]),
                              heuristic_policies(SCENARIOS["nominal"])["Fixed"], seed=42)

    assert "known_state_pct" not in kpis


# --- Variante de l'état : route de plus petite attente prévue (règle, point 9) --------------

def test_variante_route_de_plus_petite_attente_prevue():
    """``best_route`` : dernière variable = route (dans l'ordre des actions) de plus petite
    attente prévue, la même prévision que MTWT (``MineEnv.route_waits``) ; 6 valeurs
    possibles au lieu de 3."""
    assert state_space_size(state_feature="best_route") == 8**5 * 6
    scenario = SCENARIOS["nominal"]
    env = make_env(scenario)
    obs, info = env.reset(seed=42)
    policy = heuristic_policies(scenario)["MTWT"]
    seen = set()
    for _ in range(300):
        waits = env.route_waits()
        state = _discretize_obs(obs, state_feature="best_route")
        capped = np.minimum(np.asarray(waits) / 30.0, 1.0)  # bloc route_wait de l'observation
        assert state[:5] == _discretize_obs(obs)[:5]  # attentes : inchangées
        assert state[5] == int(np.argmin(capped))
        seen.add(state[5])
        obs, _, _, _, info = env.step(policy(obs, info, env))
    assert len(seen) > 1  # la variable change au cours du poste
    with pytest.raises(ValueError, match="state_feature"):
        _discretize_obs(obs, state_feature="inconnu")


def test_la_politique_garde_sa_representation(tmp_path):
    """La Q-table enregistrée garde ``state_feature`` ; une Q-table d'avant le point 9 (sans
    cette clé) est lue avec la pelle la plus proche."""
    import pickle

    from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning

    train_q_learning(n_episodes=2, episode_minutes=60.0, output_dir=str(tmp_path),
                     state_feature="best_route", checkpoint_every=None)
    data = pickle.loads((tmp_path / "q_table.pkl").read_bytes())
    assert data["state_feature"] == "best_route"
    assert {state[5] for state in data["q_table"]} <= set(range(6))
    assert TabularPolicy.load(tmp_path / "q_table.pkl").state_feature == "best_route"

    del data["state_feature"]
    (tmp_path / "old.pkl").write_bytes(pickle.dumps(data))
    assert TabularPolicy.load(tmp_path / "old.pkl").state_feature == "nearest_shovel"
