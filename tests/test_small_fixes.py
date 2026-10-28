"""Tests de T9 (petites corrections) : chemins des modèles, écart-type, mesure du temps, scénarios."""

import inspect
import math

import pytest
from reference_values import reference

from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.main import training_dir
from memoire_master_rl_logistique.rl.evaluate_agent import evaluate_policy
from memoire_master_rl_logistique.rl.model_paths import RESULTS_DIR, ensure_new_model, model_dir
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS


def fixed_policy_fn():
    policy = FixedAssignmentPolicy(num_shovels=3, num_dumps=2)

    def policy_fn(obs, info, env):
        return policy.predict(obs, info, truck_location=env.truck_locations[env.current_truck_idx])

    return policy_fn


# --- N8 : un seul dossier par modèle, partagé par l'entraînement et le benchmark ---------

def test_model_dir():
    """Le dossier d'un modèle suit la convention relue par le benchmark."""
    assert model_dir("data/results", "PPO", "nominal").as_posix() == "data/results/ppo_nominal"
    assert model_dir("r", "Q-Learning", "high_load").as_posix() == "r/q_learning_high_load"
    with pytest.raises(ValueError, match="inconnu"):
        model_dir("r", "A2C", "nominal")


@pytest.mark.parametrize(
    ("train", "algorithm"),
    [(train_ppo, "PPO"), (train_dqn, "DQN"), (train_q_learning, "Q-Learning"), (train_sarsa, "SARSA")],
)
def test_entrainement_par_defaut_la_ou_le_benchmark_relit(train, algorithm):
    """Par défaut, un entraînement écrit dans le dossier nominal que le benchmark relit."""
    default = inspect.signature(train).parameters["output_dir"].default
    assert default == str(model_dir(RESULTS_DIR, algorithm, "nominal", TRAIN_SEEDS[0]))


def test_training_dir_de_la_ligne_de_commande():
    """Flotte nominale → dossier du benchmark ; autre flotte → dossier séparé, jamais mélangé."""
    assert training_dir("PPO", 12, 3, 2) == str(model_dir(RESULTS_DIR, "PPO", "nominal", 1))
    assert training_dir("PPO", 15, 3, 2) == str(model_dir(RESULTS_DIR, "PPO", "custom_15t_3s_2d", 1))


def test_ensure_new_model(tmp_path):
    """Un modèle existant (avec ou sans .zip ajouté par SB3) n'est pas écrasé sans overwrite."""
    ensure_new_model(tmp_path / "ppo_mine_agent", overwrite=False)  # rien n'existe : OK
    (tmp_path / "ppo_mine_agent.zip").write_bytes(b"modele")

    with pytest.raises(FileExistsError):
        ensure_new_model(tmp_path / "ppo_mine_agent", overwrite=False)
    ensure_new_model(tmp_path / "ppo_mine_agent", overwrite=True)


def test_un_entrainement_ne_remplace_pas_un_modele(tmp_path):
    """Relancer un entraînement dans le même dossier échoue avant de calculer quoi que ce soit."""
    train_q_learning(n_episodes=1, output_dir=str(tmp_path))

    with pytest.raises(FileExistsError):
        train_q_learning(n_episodes=1, output_dir=str(tmp_path))
    train_q_learning(n_episodes=1, output_dir=str(tmp_path), overwrite=True)


# --- N9 : écart-type d'échantillon ------------------------------------------------------

def test_evaluate_policy_ecart_type_d_echantillon():
    """L'écart-type sur 2 épisodes est celui d'un échantillon (ddof=1) : |a − b| / √2."""
    policy = FixedAssignmentPolicy(num_shovels=3, num_dumps=2)
    env = MineEnv()

    def policy_fn(obs, info):
        return policy.predict(obs, info, truck_location=env.truck_locations[env.current_truck_idx])

    result = evaluate_policy(env, policy_fn, n_episodes=2)
    a, b = (run_single_episode(MineEnv(), fixed_policy_fn(), seed)["productivity_tph"]
            for seed in (42, 43))

    assert result["productivity_tph_std"] == pytest.approx(abs(a - b) / math.sqrt(2))
    assert evaluate_policy(env, policy_fn, n_episodes=1)["productivity_tph_std"] == 0.0


# --- m20 : décision et construction de l'observation mesurées séparément -----------------

def test_temps_de_decision_et_d_observation_separes():
    """Un épisode rapporte les deux durées, positives, et plus l'ancienne mesure mélangée."""
    kpis = run_single_episode(MineEnv(), fixed_policy_fn(), seed=42)

    for name in ("decision_time_ms", "obs_time_ms"):
        assert kpis[f"{name}_mean"] > 0
        assert kpis[f"{name}_max"] >= kpis[f"{name}_mean"]
    assert "inference_time_ms_mean" not in kpis


def test_la_mesure_ne_change_pas_les_resultats():
    """Chronométrer l'observation ne modifie pas la simulation (référence de non-régression)."""
    kpis = run_single_episode(MineEnv(), fixed_policy_fn(), seed=42)

    assert kpis["productivity_tph"] == pytest.approx(reference("Fixed", 42)["productivity_tph"])


# --- Scénarios non vus ------------------------------------------------------------------

def test_scenarios_non_vus_lisibles_par_les_modeles_nominaux():
    """Chaque scénario « unseen_* » (évalué avec les modèles nominaux) a la même observation
    que le nominal. unseen_medium_load (15 camions), retiré en T9, est revenu avec T18."""
    nominal_size = MineEnv(truck_count=SCENARIOS["nominal"].truck_count).observation_space.shape
    unseen = [s for name, s in SCENARIOS.items() if name.startswith("unseen_")]

    assert "unseen_medium_load" in SCENARIOS
    assert unseen
    for scenario in unseen:
        assert MineEnv(truck_count=scenario.truck_count).observation_space.shape == nominal_size
