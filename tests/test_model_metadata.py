import json

import pytest

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.model_metadata import (
    METADATA_FILENAME,
    EnvConfig,
    MissingMetadataError,
    ModelCompatibilityError,
    build_metadata,
    check_model_compatibility,
    git_state,
    read_metadata,
    write_metadata,
)
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning

NOMINAL = EnvConfig.from_scenario(SCENARIOS["nominal"])
OBS_SIZE = 147


def write_fake_model(model_dir, env=NOMINAL, algorithm="DQN", observation_size=OBS_SIZE):
    """Un « modèle » réduit à son metadata.json : c'est tout ce que le contrôle regarde."""
    write_metadata(model_dir, build_metadata(algorithm, env, observation_size, seed=42))


def test_aller_retour_json(tmp_path):
    """Ce qui est écrit est relu à l'identique, avec date et état Git."""
    original = build_metadata("PPO", NOMINAL, OBS_SIZE, seed=42, training={"total_timesteps": 10})
    write_metadata(tmp_path, original)

    assert read_metadata(tmp_path) == original
    assert original.created_at
    assert json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))["algorithm"] == "PPO"


def test_envconfig_normalise_les_types():
    """480 et 480.0, une liste ou un tuple de poids : même configuration."""
    a = EnvConfig(12, 3, 2, 480, 0.02, [1, 0.1, 0.05, 0.3])
    b = EnvConfig(12, 3, 2, 480.0, 0.02, (1.0, 0.1, 0.05, 0.3))

    assert a == b
    assert a == NOMINAL


def test_modele_compatible_accepte(tmp_path):
    """Même algorithme, même configuration, même observation : le modèle est accepté."""
    write_fake_model(tmp_path)

    metadata = check_model_compatibility(tmp_path, "DQN", NOMINAL, OBS_SIZE)

    assert metadata is not None and metadata.algorithm == "DQN"


def test_recompense_perimee_refusee(tmp_path):
    """Test du plan : un modèle entraîné avec d'anciens poids de récompense est refusé."""
    ancienne_recompense = EnvConfig(12, 3, 2, 480.0, 0.02, (1.0, 0.1, 0.05, 1.0))  # w4 = 1,0 (avant mai)
    write_fake_model(tmp_path, env=ancienne_recompense)

    with pytest.raises(ModelCompatibilityError, match="reward_weights"):
        check_model_compatibility(tmp_path, "DQN", NOMINAL, OBS_SIZE)


def test_observation_differente_refusee(tmp_path):
    """Une autre taille d'observation (138 avant le 30 mai) est refusée."""
    write_fake_model(tmp_path, observation_size=138)

    with pytest.raises(ModelCompatibilityError, match="observation_size"):
        check_model_compatibility(tmp_path, "DQN", NOMINAL, OBS_SIZE)


def test_autre_algorithme_refuse(tmp_path):
    """Un modèle PPO ne peut pas être chargé à la place d'un DQN."""
    write_fake_model(tmp_path, algorithm="PPO")

    with pytest.raises(ModelCompatibilityError, match="algorithm"):
        check_model_compatibility(tmp_path, "DQN", NOMINAL, OBS_SIZE)


def test_tous_les_ecarts_sont_listes(tmp_path):
    """Le message liste chaque écart, pas seulement le premier."""
    forte_charge = EnvConfig.from_scenario(SCENARIOS["high_load"])
    write_fake_model(tmp_path, env=forte_charge, observation_size=138)

    with pytest.raises(ModelCompatibilityError) as error:
        check_model_compatibility(tmp_path, "DQN", NOMINAL, OBS_SIZE)
    assert "truck_count" in str(error.value) and "observation_size" in str(error.value)


def test_modele_sans_metadonnees(tmp_path):
    """Modèle antérieur à T8 : refusé par défaut, accepté avec avertissement si demandé."""
    with pytest.raises(MissingMetadataError):
        check_model_compatibility(tmp_path, "DQN", NOMINAL, OBS_SIZE)

    with pytest.warns(UserWarning, match="sans métadonnées"):
        assert check_model_compatibility(tmp_path, "DQN", NOMINAL, OBS_SIZE, allow_legacy=True) is None


def test_git_state_dans_le_depot():
    """Dans le dépôt, le commit est un hash de 40 caractères."""
    commit, dirty = git_state()
    if commit is None:
        pytest.skip("Git indisponible")
    assert len(commit) == 40
    assert isinstance(dirty, bool)


def test_entrainement_ecrit_des_metadonnees_compatibles(tmp_path):
    """Un vrai entraînement (1 épisode) écrit un metadata.json accepté pour le régime nominal."""
    train_q_learning(n_episodes=1, output_dir=str(tmp_path))

    metadata = read_metadata(tmp_path)
    observation_size = MineEnv().observation_space.shape[0]

    assert metadata.algorithm == "Q-Learning"
    assert metadata.training["n_episodes"] == 1
    assert metadata.observation_size == observation_size
    check_model_compatibility(tmp_path, "Q-Learning", NOMINAL, observation_size)
