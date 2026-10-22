"""Tests de T28 : fichiers de configuration réellement utilisés."""

import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from memoire_master_rl_logistique import config
from memoire_master_rl_logistique.config import (
    CONFIG_DIR_ENV,
    CONFIG_FILES,
    experiment_config,
    hyperparameters,
    snapshot_configs,
)
from memoire_master_rl_logistique.experiments import run_benchmark as rb
from memoire_master_rl_logistique.experiments.scenarios import (
    MAIN_SCENARIOS,
    SCENARIOS,
    _build_scenarios,
)
from memoire_master_rl_logistique.rl.discretize import DEFAULT_N_BINS
from memoire_master_rl_logistique.rl.time_discount import per_minute_gamma
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa
from memoire_master_rl_logistique.seeds import EVAL_SEEDS, TRAIN_SEEDS, TUNING_SEEDS

# Valeurs d'avant T28 (recopiées en dur dans scenarios.py) : la lecture des YAML doit
# les redonner exactement. (truck_count, breakdown_probability, w4) ; le reste est commun.
EXPECTED_SCENARIOS = {
    "nominal": (12, 0.02, 0.3),
    "nominal_asymétrique": (13, 0.02, 0.3),
    "high_load": (18, 0.02, 0.3),
    "high_breakdown": (12, 0.10, 0.3),
    "ablation_w4_00": (12, 0.02, 0.0),
    "ablation_w4_01": (12, 0.02, 0.1),
    "ablation_w4_05": (12, 0.02, 0.5),
    "unseen_medium_load": (15, 0.02, 0.3),
    "unseen_medium_breakdown": (12, 0.05, 0.3),
}


@pytest.fixture
def config_copy(tmp_path):
    """Copie modifiable de configs/ ; renvoie (dossier, fonction d'édition d'un fichier)."""
    directory = tmp_path / "configs"
    directory.mkdir()
    for name in CONFIG_FILES:
        shutil.copy(config.DEFAULT_CONFIG_DIR / name, directory / name)

    def edit(name, change):
        path = directory / name
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        change(content)
        path.write_text(yaml.safe_dump(content, allow_unicode=True), encoding="utf-8")

    return directory, edit


# --- Les valeurs n'ont pas changé en passant dans les YAML ---------------------------

def test_scenarios_identiques_a_avant_t28():
    assert list(SCENARIOS) == list(EXPECTED_SCENARIOS)
    for name, (trucks, breakdown, w4) in EXPECTED_SCENARIOS.items():
        s = SCENARIOS[name]
        assert (s.name, s.truck_count, s.breakdown_probability) == (name, trucks, breakdown)
        assert s.reward_weights == (1.0, 0.1, 0.05, w4) and isinstance(s.reward_weights, tuple)
        assert (s.shovel_count, s.dump_count, s.episode_minutes) == (3, 2, 480.0)
        assert s.total_timesteps == 2_000_000 and s.seeds == list(range(42, 72))
    assert MAIN_SCENARIOS == ["nominal", "high_load", "high_breakdown"]


def test_graines_identiques_a_t26():
    assert EVAL_SEEDS == tuple(range(42, 72))
    assert TUNING_SEEDS == tuple(range(20_000, 20_010))
    assert TRAIN_SEEDS == (1, 2, 3, 4, 5)


def test_hyperparametres_de_la_campagne():
    """Valeurs de la campagne v2 (mise au point des agents, 17 octobre 2026). Par rapport à
    la v1 : PPO learning_rate 3e-4 → 1e-4, ent_coef 0,02 → 0, gamma 0,99 → 0,995,
    normalize_reward false → true ; DQN exploration_fraction 0,3 → 0,1, gamma 0,99 → 0,995.
    Depuis le 21 octobre 2026, actualisation par minute pour les quatre algorithmes
    (time_discount, rl/time_discount.py)."""
    assert hyperparameters("PPO") == {
        "learning_rate": 1e-4, "gamma": 0.995, "batch_size": 64, "gae_lambda": 0.95,
        "clip_range": 0.2, "n_steps": 1024, "n_epochs": 10, "ent_coef": 0.0,
        "net_arch": [128, 128], "n_envs": 1,  # n_envs : T29, 1 = protocole v1
        "normalize_reward": True, "time_discount": True}
    assert hyperparameters("DQN") == {
        "learning_rate": 3e-4, "gamma": 0.995, "batch_size": 64, "buffer_size": 200_000,
        "learning_starts": 1000, "target_update_interval": 1000,
        "exploration_fraction": 0.1, "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02, "net_arch": [128, 128], "time_discount": True}
    # Les valeurs retenues font partie des grilles du réglage (T25 bis).
    grids = experiment_config()["tuning"]["grids"]
    assert hyperparameters("PPO")["ent_coef"] in grids["PPO"]["ent_coef"]
    assert hyperparameters("PPO")["learning_rate"] in grids["PPO"]["learning_rate"]
    assert hyperparameters("DQN")["exploration_fraction"] in grids["DQN"]["exploration_fraction"]
    assert hyperparameters("DQN")["learning_rate"] in grids["DQN"]["learning_rate"]
    # Agents tabulaires : réglage du 22 octobre 2026 (essais 16 à 23), α par algorithme,
    # 6 intervalles au lieu de 8 ; variante de l'état rejetée (essais 24 et 25).
    tabular = {"n_episodes": 30_000, "gamma": 0.99, "epsilon_start": 1.0,
               "epsilon_min": 0.01, "n_bins": 6, "time_discount": True,
               "state_feature": "nearest_shovel"}
    assert hyperparameters("Q-Learning") == tabular | {"alpha": 0.2}
    assert hyperparameters("SARSA") == tabular | {"alpha": 0.1}


def test_valeurs_par_defaut_du_code_coherentes_avec_la_configuration():
    """Ce qui reste écrit dans les signatures (environnement, budget, discrétisation)
    est égal à la configuration : une divergence ferait échouer ce test."""
    nominal = SCENARIOS[MAIN_SCENARIOS[0]]
    env_fields = ["truck_count", "shovel_count", "dump_count", "episode_minutes",
                  "breakdown_probability", "reward_weights"]
    for trainer in (train_q_learning, train_sarsa, train_dqn, train_ppo):
        defaults = {k: p.default for k, p in inspect.signature(trainer).parameters.items()}
        assert {f: defaults[f] for f in env_fields} == {f: getattr(nominal, f) for f in env_fields}
        assert defaults["seed"] == TRAIN_SEEDS[0]
    for trainer in (train_dqn, train_ppo):
        assert inspect.signature(trainer).parameters["total_timesteps"].default == \
            nominal.total_timesteps
    assert DEFAULT_N_BINS == hyperparameters("Q-Learning")["n_bins"]


# --- Les scripts lisent vraiment la configuration -----------------------------------

def test_entraineurs_tabulaires_utilisent_la_configuration(tmp_path):
    """metadata.json contient les hyperparamètres du YAML ; un argument donné les remplace.
    Avec l'actualisation par minute, γ_min calculé à partir de γ est déclaré en plus."""
    train_q_learning(n_episodes=1, episode_minutes=60.0, output_dir=str(tmp_path))
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    config = hyperparameters("Q-Learning")
    assert metadata["training"] == config | {
        "n_episodes": 1, "gamma_per_minute": per_minute_gamma(config["gamma"])}


def test_ppo_et_dqn_utilisent_la_configuration(tmp_path):
    """Le modèle construit a les hyperparamètres du YAML (et non des valeurs en dur)."""
    ppo = train_ppo(total_timesteps=1_024, episode_minutes=60.0, output_dir=str(tmp_path / "p"),
                    progress_bar=False)
    assert (ppo.n_steps, ppo.batch_size, ppo.n_epochs, ppo.ent_coef) == (1024, 64, 10, 0.0)
    assert ppo.gamma == 0.995
    assert ppo.policy.net_arch == [128, 128]

    dqn = train_dqn(total_timesteps=1_100, episode_minutes=60.0, output_dir=str(tmp_path / "d"),
                    progress_bar=False, exploration_fraction=0.3)
    assert dqn.exploration_fraction == 0.3  # argument donné (configuration : 0,1)
    assert (dqn.buffer_size, dqn.learning_starts, dqn.exploration_final_eps) == (200_000, 1000, 0.02)


def test_variable_d_environnement(config_copy):
    """Une autre configuration (MINE_RL_CONFIG_DIR) change vraiment les régimes et les graines."""
    directory, edit = config_copy
    edit("experiment_config.yaml", lambda c: (
        c["scenarios"]["high_load"].update(truck_count=20),
        c["seeds"]["evaluation"].update(start=100),
    ))
    code = ("from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS;"
            "print(SCENARIOS['high_load'].truck_count, SCENARIOS['nominal'].seeds[0])")
    env = {**os.environ, CONFIG_DIR_ENV: str(directory)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, check=True).stdout.split()
    assert out == ["20", "100"]


# --- Validation stricte -------------------------------------------------------------

def test_cle_inconnue_ou_manquante_refusee(config_copy):
    directory, edit = config_copy
    edit("ppo_hyperparameters.yaml", lambda c: c.update(ent_coeff=0.01))  # faute de frappe
    with pytest.raises(ValueError, match="ent_coeff"):
        hyperparameters("PPO", directory)
    edit("dqn_hyperparameters.yaml", lambda c: c.pop("buffer_size"))
    with pytest.raises(ValueError, match="buffer_size"):
        hyperparameters("DQN", directory)


def test_fichier_d_un_autre_algorithme_refuse(config_copy):
    directory, edit = config_copy
    edit("ppo_hyperparameters.yaml", lambda c: c.update(algorithm="DQN"))
    with pytest.raises(ValueError, match="ne correspond pas"):
        hyperparameters("PPO", directory)


def test_valeur_par_algorithme_tabulaire(config_copy):
    """Une valeur tabulaire peut dépendre de l'algorithme ; elle doit alors nommer
    Q-Learning et SARSA, sans faute de frappe. Une valeur commune reste acceptée (fichiers
    archivés avant le réglage)."""
    directory, edit = config_copy

    def set_value(**values):
        edit("tabular_hyperparameters.yaml", lambda c: c.update(values))
        config._read_yaml.cache_clear()  # le fichier est relu après chaque modification

    set_value(n_bins={"Q-Learning": 4, "SARSA": 10})
    assert hyperparameters("Q-Learning", directory)["n_bins"] == 4
    assert hyperparameters("SARSA", directory)["n_bins"] == 10
    assert hyperparameters("SARSA", directory, n_bins=8)["n_bins"] == 8  # argument donné
    set_value(alpha=0.3)
    assert hyperparameters("Q-Learning", directory)["alpha"] == \
        hyperparameters("SARSA", directory)["alpha"] == 0.3
    set_value(alpha={"Q-Learning": 0.2, "Sarsa": 0.1})
    with pytest.raises(ValueError, match="Sarsa"):
        hyperparameters("Q-Learning", directory)
    set_value(alpha={"Q-Learning": 0.2})
    with pytest.raises(ValueError, match="SARSA"):
        hyperparameters("Q-Learning", directory)


def test_remplacement_d_hyperparametres():
    assert hyperparameters("PPO", ent_coef=0.02)["ent_coef"] == 0.02
    assert hyperparameters("PPO", ent_coef=None)["ent_coef"] == 0.0  # None = non fourni
    with pytest.raises(ValueError, match="inconnus"):
        hyperparameters("PPO", exploration_fraction=0.1)
    with pytest.raises(ValueError, match="inconnu"):
        hyperparameters("A2C")


def test_scenario_mal_decrit_refuse(config_copy):
    directory, edit = config_copy
    edit("experiment_config.yaml", lambda c: c["scenarios"]["high_load"].update(truck_cont=18))
    with pytest.raises(ValueError, match="high_load"):
        _build_scenarios(experiment_config(directory))


def test_algorithmes_par_scenario():
    """Les quatre agents partout, sauf l'ablation sur w4 : PPO et DQN seulement
    (décision du 22 octobre 2026)."""
    for name, scenario in SCENARIOS.items():
        expected = ("DQN", "PPO") if name.startswith("ablation_w4") else \
            ("Q-Learning", "SARSA", "DQN", "PPO")
        assert scenario.algorithms == expected, name


def test_algorithmes_mal_decrits_refuses_ou_absents(config_copy):
    """Un nom inconnu ou une liste vide est refusé ; sans la clé (configuration archivée
    avant le 22 octobre 2026), les quatre agents, comme avant."""
    directory, edit = config_copy
    for wrong in (["PPO", "A2C"], []):
        edit("experiment_config.yaml",
             lambda c, wrong=wrong: c["scenarios"]["ablation_w4_00"].update(algorithms=wrong))
        config._read_yaml.cache_clear()
        with pytest.raises(ValueError, match="ablation_w4_00"):
            _build_scenarios(experiment_config(directory))

    def remove_key(c):
        c["scenario_defaults"].pop("algorithms")
        for scenario in c["scenarios"].values():
            scenario.pop("algorithms", None)

    edit("experiment_config.yaml", remove_key)
    config._read_yaml.cache_clear()
    assert all(s.algorithms == ("Q-Learning", "SARSA", "DQN", "PPO")
               for s in _build_scenarios(experiment_config(directory)).values())


def test_regime_principal_inconnu_refuse(config_copy):
    directory, edit = config_copy
    edit("experiment_config.yaml", lambda c: c.update(main_scenarios=["nominal", "hihg_load"]))
    with pytest.raises(ValueError, match="hihg_load"):
        experiment_config(directory)


def test_configuration_retournee_est_une_copie():
    experiment_config()["seeds"]["training"].append(99)
    hyperparameters("PPO")["net_arch"].append(7)
    assert experiment_config()["seeds"]["training"] == [1, 2, 3, 4, 5]
    assert hyperparameters("PPO")["net_arch"] == [128, 128]


# --- Archivage et commande unique ---------------------------------------------------

def test_archive_de_la_configuration(tmp_path):
    archive = snapshot_configs(tmp_path / "configs_run")
    for name in CONFIG_FILES:
        assert (archive / name).read_bytes() == (config.DEFAULT_CONFIG_DIR / name).read_bytes()
    snapshot_configs(archive)  # reprise avec la même configuration : accepté

    (archive / "ppo_hyperparameters.yaml").write_text("modifié", encoding="utf-8")
    with pytest.raises(FileExistsError):
        snapshot_configs(archive)


def test_campagne_archive_sa_configuration(monkeypatch, tmp_path):
    """Le benchmark archive la configuration à côté du CSV, avec le même run_id."""
    monkeypatch.setattr(rb, "run_benchmark", lambda scenario, *a, **k: [
        {"scenario": scenario.name, "policy": "Fixed", "mode": None, "train_seed": None,
         "seed": 42, "productivity_tph": 1.0}])

    csv_path = rb.run_all_benchmarks(str(tmp_path), scenario_names=["nominal", "high_load"],
                                     run_id="r1")

    assert csv_path == tmp_path / "benchmark_r1.csv"
    assert sorted(p.name for p in (tmp_path / "configs_r1").iterdir()) == sorted(CONFIG_FILES)
    with pytest.raises(ValueError, match="inconnus"):
        rb.run_all_benchmarks(str(tmp_path), scenario_names=["hihg_load"], run_id="r2")


def test_main_scenarios_par_defaut_de_la_commande(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(rb, "run_all_benchmarks", lambda out, scenario_names, **k: seen.append(
        list(scenario_names)) or Path(out))
    rb.main([str(tmp_path)])
    assert seen == [MAIN_SCENARIOS]
